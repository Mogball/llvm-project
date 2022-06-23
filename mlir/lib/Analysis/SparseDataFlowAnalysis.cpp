//===- SparseDataFlowAnalysis.cpp - Sparse data-flow analysis -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SparseDataFlowAnalysis.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dataflow"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AbstractSparseLattice
//===----------------------------------------------------------------------===//

void AbstractSparseLattice::onUpdate(DataFlowSolver *solver) const {
  // Push all users of the value to the queue.
  for (Operation *user : point.get<Value>().getUsers())
    for (DataFlowAnalysis *analysis : useDefSubscribers)
      solver->enqueue({user, analysis});
}

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

ChangeResult Executable::setToLive() {
  if (live)
    return ChangeResult::NoChange;
  live = true;
  return ChangeResult::Change;
}

void Executable::print(raw_ostream &os) const {
  os << (live ? "live" : "dead");
}

void Executable::onUpdate(DataFlowSolver *solver) const {
  if (auto *block = point.dyn_cast<Block *>()) {
    // Re-invoke the analyses on the block itself.
    for (DataFlowAnalysis *analysis : subscribers)
      solver->enqueue({block, analysis});
    // Re-invoke the analyses on all operations in the block.
    for (DataFlowAnalysis *analysis : subscribers)
      for (Operation &op : *block)
        solver->enqueue({&op, analysis});
  } else if (auto *programPoint = point.dyn_cast<GenericProgramPoint *>()) {
    // Re-invoke the analysis on the successor block.
    if (auto *edge = dyn_cast<CFGEdge>(programPoint))
      for (DataFlowAnalysis *analysis : subscribers)
        solver->enqueue({edge->getTo(), analysis});
  }
}

//===----------------------------------------------------------------------===//
// ConstantValue
//===----------------------------------------------------------------------===//

void ConstantValue::print(raw_ostream &os) const {
  if (constant)
    return constant.print(os);
  os << "<NO VALUE>";
}

//===----------------------------------------------------------------------===//
// PredecessorState
//===----------------------------------------------------------------------===//

void PredecessorState::print(raw_ostream &os) const {
  if (allPredecessorsKnown())
    os << "(all) ";
  os << "predecessors:\n";
  for (Operation *op : getKnownPredecessors())
    os << "  " << *op << "\n";
}

ChangeResult PredecessorState::join(Operation *predecessor) {
  return knownPredecessors.insert(predecessor) ? ChangeResult::Change
                                               : ChangeResult::NoChange;
}

ChangeResult PredecessorState::join(Operation *predecessor, ValueRange inputs) {
  ChangeResult result = join(predecessor);
  if (!inputs.empty()) {
    ValueRange &curInputs = successorInputs[predecessor];
    if (curInputs != inputs) {
      curInputs = inputs;
      result |= ChangeResult::Change;
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// CFGEdge
//===----------------------------------------------------------------------===//

Location CFGEdge::getLoc() const {
  return FusedLoc::get(
      getFrom()->getParent()->getContext(),
      {getFrom()->getParent()->getLoc(), getTo()->getParent()->getLoc()});
}

void CFGEdge::print(raw_ostream &os) const {
  getFrom()->print(os);
  os << "\n -> \n";
  getTo()->print(os);
}

//===----------------------------------------------------------------------===//
// DeadCodeAnalysis
//===----------------------------------------------------------------------===//

DeadCodeAnalysis::DeadCodeAnalysis(DataFlowSolver &solver)
    : DataFlowAnalysis(solver) {
  registerPointKind<CFGEdge>();
}

LogicalResult DeadCodeAnalysis::initialize(Operation *top) {
  // Mark the top-level blocks as executable.
  for (Region &region : top->getRegions()) {
    if (region.empty())
      continue;
    auto *state = getOrCreate<Executable>(&region.front());
    propagateIfChanged(state, state->setToLive());
  }

  // Mark as overdefined the predecessors of symbol callables with potentially
  // unknown predecessors.
  initializeSymbolCallables(top);

  return initializeRecursively(top);
}

void DeadCodeAnalysis::initializeSymbolCallables(Operation *top) {
  auto walkFn = [&](Operation *symTable, bool allUsesVisible) {
    Region &symbolTableRegion = symTable->getRegion(0);
    Block *symbolTableBlock = &symbolTableRegion.front();

    bool foundSymbolCallable = false;
    for (auto callable : symbolTableBlock->getOps<CallableOpInterface>()) {
      Region *callableRegion = callable.getCallableRegion();
      if (!callableRegion)
        continue;
      auto symbol = dyn_cast<SymbolOpInterface>(callable.getOperation());
      if (!symbol)
        continue;

      // Public symbol callables or those for which we can't see all uses have
      // potentially unknown callsites.
      if (symbol.isPublic() || (!allUsesVisible && symbol.isNested())) {
        auto *state = getOrCreate<PredecessorState>(callable);
        propagateIfChanged(state, state->setHasUnknownPredecessors());
      }
      foundSymbolCallable = true;
    }

    // Exit early if no eligible symbol callables were found in the table.
    if (!foundSymbolCallable)
      return;

    // Walk the symbol table to check for non-call uses of symbols.
    Optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(&symbolTableRegion);
    if (!uses) {
      // If we couldn't gather the symbol uses, conservatively assume that
      // we can't track information for any nested symbols.
      return top->walk([&](CallableOpInterface callable) {
        auto *state = getOrCreate<PredecessorState>(callable);
        propagateIfChanged(state, state->setHasUnknownPredecessors());
      });
    }

    for (const SymbolTable::SymbolUse &use : *uses) {
      if (isa<CallOpInterface>(use.getUser()))
        continue;
      // If a callable symbol has a non-call use, then we can't be guaranteed to
      // know all callsites.
      Operation *symbol = symbolTable.lookupSymbolIn(top, use.getSymbolRef());
      auto *state = getOrCreate<PredecessorState>(symbol);
      propagateIfChanged(state, state->setHasUnknownPredecessors());
    }
  };
  SymbolTable::walkSymbolTables(top, /*allSymUsesVisible=*/!top->getBlock(),
                                walkFn);
}

/// Returns true if the operation terminates a block. It is insufficient to
/// check for `OpTrait::IsTerminator` because unregistered operations can be
/// terminators.
static bool isTerminator(Operation *op) {
  if (op->hasTrait<OpTrait::IsTerminator>())
    return true;
  return &op->getBlock()->back() == op;
}

LogicalResult DeadCodeAnalysis::initializeRecursively(Operation *op) {
  // Initialize the analysis by visiting every op with control-flow semantics.
  if (op->getNumRegions() || op->getNumSuccessors() || isTerminator(op) ||
      isa<CallOpInterface>(op)) {
    // When the liveness of the parent block changes, make sure to re-invoke the
    // analysis on the op.
    if (op->getBlock())
      getOrCreate<Executable>(op->getBlock())->blockContentSubscribe(this);
    // Visit the op.
    if (failed(visit(op)))
      return failure();
  }
  // Recurse on nested operations.
  for (Region &region : op->getRegions())
    for (Operation &op : region.getOps())
      if (failed(initializeRecursively(&op)))
        return failure();
  return success();
}

void DeadCodeAnalysis::markEdgeLive(Block *from, Block *to) {
  auto *state = getOrCreate<Executable>(to);
  propagateIfChanged(state, state->setToLive());
  auto *edgeState = getOrCreate<Executable>(getProgramPoint<CFGEdge>(from, to));
  propagateIfChanged(edgeState, edgeState->setToLive());
}

void DeadCodeAnalysis::markEntryBlocksLive(Operation *op) {
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    auto *state = getOrCreate<Executable>(&region.front());
    propagateIfChanged(state, state->setToLive());
  }
}

LogicalResult DeadCodeAnalysis::visit(ProgramPoint point) {
  if (point.is<Block *>())
    return success();
  auto *op = point.dyn_cast<Operation *>();
  if (!op)
    return emitError(point.getLoc(), "unknown program point kind");

  // If the parent block is not executable, there is nothing to do.
  if (!getOrCreate<Executable>(op->getBlock())->isLive())
    return success();

  // We have a live call op. Add this as a live predecessor of the callee.
  if (auto call = dyn_cast<CallOpInterface>(op))
    visitCallOperation(call);

  // Visit the regions.
  if (op->getNumRegions()) {
    // Check if we can reason about the region control-flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
      visitRegionBranchOperation(branch);

      // Check if this is a callable operation.
    } else if (auto callable = dyn_cast<CallableOpInterface>(op)) {
      const auto *callsites = getOrCreateFor<PredecessorState>(op, callable);

      // If the callsites could not be resolved or are known to be non-empty,
      // mark the callable as executable.
      if (!callsites->allPredecessorsKnown() ||
          !callsites->getKnownPredecessors().empty())
        markEntryBlocksLive(callable);

      // Otherwise, conservatively mark all entry blocks as executable.
    } else {
      markEntryBlocksLive(op);
    }
  }

  if (isTerminator(op) && !op->getNumSuccessors()) {
    if (auto branch = dyn_cast<RegionBranchOpInterface>(op->getParentOp())) {
      // Visit the exiting terminator of a region.
      visitRegionTerminator(op, branch);
    } else if (auto callable =
                   dyn_cast<CallableOpInterface>(op->getParentOp())) {
      // Visit the exiting terminator of a callable.
      visitCallableTerminator(op, callable);
    }
  }
  // Visit the successors.
  if (op->getNumSuccessors()) {
    // Check if we can reason about the control-flow.
    if (auto branch = dyn_cast<BranchOpInterface>(op)) {
      visitBranchOperation(branch);

      // Otherwise, conservatively mark all successors as exectuable.
    } else {
      for (Block *successor : op->getSuccessors())
        markEdgeLive(op->getBlock(), successor);
    }
  }

  return success();
}

void DeadCodeAnalysis::visitCallOperation(CallOpInterface call) {
  Operation *callableOp = nullptr;
  if (Value callableValue = call.getCallableForCallee().dyn_cast<Value>())
    callableOp = callableValue.getDefiningOp();
  else
    callableOp = call.resolveCallable(&symbolTable);

  // A call to a externally-defined callable has unknown predecessors.
  const auto isExternalCallable = [](Operation *op) {
    if (auto callable = dyn_cast<CallableOpInterface>(op))
      return !callable.getCallableRegion();
    return false;
  };

  // TODO: Add support for non-symbol callables when necessary. If the
  // callable has non-call uses we would mark as having reached pessimistic
  // fixpoint, otherwise allow for propagating the return values out.
  if (isa_and_nonnull<SymbolOpInterface>(callableOp) &&
      !isExternalCallable(callableOp)) {
    // Add the live callsite.
    auto *callsites = getOrCreate<PredecessorState>(callableOp);
    propagateIfChanged(callsites, callsites->join(call));
  } else {
    // Mark this call op's predecessors as overdefined.
    auto *predecessors = getOrCreate<PredecessorState>(call);
    propagateIfChanged(predecessors, predecessors->setHasUnknownPredecessors());
  }
}

/// Get the constant values of the operands of an operation. If any of the
/// constant value lattices are uninitialized, return none to indicate the
/// analysis should bail out.
static Optional<SmallVector<Attribute>> getOperandValuesImpl(
    Operation *op,
    function_ref<const Lattice<ConstantValue> *(Value)> getLattice) {
  SmallVector<Attribute> operands;
  operands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    const Lattice<ConstantValue> *cv = getLattice(operand);
    // If any of the operands' values are uninitialized, bail out.
    if (cv->isUninitialized())
      return {};
    operands.push_back(cv->getValue().getConstantValue());
  }
  return operands;
}

Optional<SmallVector<Attribute>>
DeadCodeAnalysis::getOperandValues(Operation *op) {
  return getOperandValuesImpl(op, [&](Value value) {
    auto *lattice = getOrCreate<Lattice<ConstantValue>>(value);
    lattice->useDefSubscribe(this);
    return lattice;
  });
}

void DeadCodeAnalysis::visitBranchOperation(BranchOpInterface branch) {
  // Try to deduce a single successor for the branch.
  Optional<SmallVector<Attribute>> operands = getOperandValues(branch);
  if (!operands)
    return;

  if (Block *successor = branch.getSuccessorForOperands(*operands)) {
    markEdgeLive(branch->getBlock(), successor);
  } else {
    // Otherwise, mark all successors as executable and outgoing edges.
    for (Block *successor : branch->getSuccessors())
      markEdgeLive(branch->getBlock(), successor);
  }
}

void DeadCodeAnalysis::visitRegionBranchOperation(
    RegionBranchOpInterface branch) {
  // Try to deduce which regions are executable.
  Optional<SmallVector<Attribute>> operands = getOperandValues(branch);
  if (!operands)
    return;

  SmallVector<RegionSuccessor> successors;
  branch.getSuccessorRegions(/*index=*/{}, *operands, successors);

  for (const RegionSuccessor &successor : successors) {
    // The successor can be either an entry block or the parent operation.
    ProgramPoint point = successor.getSuccessor()
                             ? &successor.getSuccessor()->front()
                             : ProgramPoint(branch);
    // Mark the entry block as executable.
    auto *state = getOrCreate<Executable>(point);
    propagateIfChanged(state, state->setToLive());
    // Add the parent op as a predecessor.
    auto *predecessors = getOrCreate<PredecessorState>(point);
    propagateIfChanged(
        predecessors,
        predecessors->join(branch, successor.getSuccessorInputs()));
  }
}

void DeadCodeAnalysis::visitRegionTerminator(Operation *op,
                                             RegionBranchOpInterface branch) {
  Optional<SmallVector<Attribute>> operands = getOperandValues(op);
  if (!operands)
    return;

  SmallVector<RegionSuccessor> successors;
  branch.getSuccessorRegions(op->getParentRegion()->getRegionNumber(),
                             *operands, successors);

  // Mark successor region entry blocks as executable and add this op to the
  // list of predecessors.
  for (const RegionSuccessor &successor : successors) {
    PredecessorState *predecessors;
    if (Region *region = successor.getSuccessor()) {
      auto *state = getOrCreate<Executable>(&region->front());
      propagateIfChanged(state, state->setToLive());
      predecessors = getOrCreate<PredecessorState>(&region->front());
    } else {
      // Add this terminator as a predecessor to the parent op.
      predecessors = getOrCreate<PredecessorState>(branch);
    }
    propagateIfChanged(predecessors,
                       predecessors->join(op, successor.getSuccessorInputs()));
  }
}

void DeadCodeAnalysis::visitCallableTerminator(Operation *op,
                                               CallableOpInterface callable) {
  // If there are no exiting values, we have nothing to do.
  if (op->getNumOperands() == 0)
    return;

  // Add as predecessors to all callsites this return op.
  auto *callsites = getOrCreateFor<PredecessorState>(op, callable);
  bool canResolve = op->hasTrait<OpTrait::ReturnLike>();
  for (Operation *predecessor : callsites->getKnownPredecessors()) {
    assert(isa<CallOpInterface>(predecessor));
    auto *predecessors = getOrCreate<PredecessorState>(predecessor);
    if (canResolve) {
      propagateIfChanged(predecessors, predecessors->join(op));
    } else {
      // If the terminator is not a return-like, then conservatively assume we
      // can't resolve the predecessor.
      propagateIfChanged(predecessors,
                         predecessors->setHasUnknownPredecessors());
    }
  }
}

//===----------------------------------------------------------------------===//
// AbstractSparseDataFlowAnalysis
//===----------------------------------------------------------------------===//

AbstractSparseDataFlowAnalysis::AbstractSparseDataFlowAnalysis(
    DataFlowSolver &solver)
    : DataFlowAnalysis(solver) {
  registerPointKind<CFGEdge>();
}

LogicalResult AbstractSparseDataFlowAnalysis::initialize(Operation *top) {
  // Mark the entry block arguments as having reached their pessimistic
  // fixpoints.
  for (Region &region : top->getRegions()) {
    if (region.empty())
      continue;
    for (Value argument : region.front().getArguments())
      markPessimisticFixpoint(getLatticeElement(argument));
  }

  return initializeRecursively(top);
}

LogicalResult
AbstractSparseDataFlowAnalysis::initializeRecursively(Operation *op) {
  // Initialize the analysis by visiting every owner of an SSA value (all
  // operations and blocks).
  visitOperation(op);
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      getOrCreate<Executable>(&block)->blockContentSubscribe(this);
      visitBlock(&block);
      for (Operation &op : block)
        if (failed(initializeRecursively(&op)))
          return failure();
    }
  }

  return success();
}

LogicalResult AbstractSparseDataFlowAnalysis::visit(ProgramPoint point) {
  if (Operation *op = point.dyn_cast<Operation *>())
    visitOperation(op);
  else if (Block *block = point.dyn_cast<Block *>())
    visitBlock(block);
  else
    return failure();
  return success();
}

void AbstractSparseDataFlowAnalysis::visitOperation(Operation *op) {
  // Exit early on operations with no results.
  if (op->getNumResults() == 0)
    return;

  // If the containing block is not executable, bail out.
  if (!getOrCreate<Executable>(op->getBlock())->isLive())
    return;

  // Get the result lattices.
  SmallVector<AbstractSparseLattice *> resultLattices;
  resultLattices.reserve(op->getNumResults());
  // Track whether all results have reached their fixpoint.
  bool allAtFixpoint = true;
  for (Value result : op->getResults()) {
    AbstractSparseLattice *resultLattice = getLatticeElement(result);
    allAtFixpoint &= resultLattice->isAtFixpoint();
    resultLattices.push_back(resultLattice);
  }
  // If all result lattices have reached a fixpoint, there is nothing to do.
  if (allAtFixpoint)
    return;

  // The results of a region branch operation are determined by control-flow.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    return visitRegionSuccessors({branch}, branch,
                                 /*successorIndex=*/llvm::None, resultLattices);
  }

  // The results of a call operation are determined by the callgraph.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    const auto *predecessors = getOrCreateFor<PredecessorState>(op, call);
    // If not all return sites are known, then conservatively assume we can't
    // reason about the data-flow.
    if (!predecessors->allPredecessorsKnown())
      return markAllPessimisticFixpoint(resultLattices);
    for (Operation *predecessor : predecessors->getKnownPredecessors())
      for (auto it : llvm::zip(predecessor->getOperands(), resultLattices))
        join(std::get<1>(it), *getLatticeElementFor(op, std::get<0>(it)));
    return;
  }

  // Grab the lattice elements of the operands.
  SmallVector<const AbstractSparseLattice *> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    AbstractSparseLattice *operandLattice = getLatticeElement(operand);
    operandLattice->useDefSubscribe(this);
    // If any of the operand states are not initialized, bail out.
    if (operandLattice->isUninitialized())
      return;
    operandLattices.push_back(operandLattice);
  }

  // Invoke the operation transfer function.
  visitOperationImpl(op, operandLattices, resultLattices);
}

void AbstractSparseDataFlowAnalysis::visitBlock(Block *block) {
  // Exit early on blocks with no arguments.
  if (block->getNumArguments() == 0)
    return;

  // If the block is not executable, bail out.
  if (!getOrCreate<Executable>(block)->isLive())
    return;

  // Get the argument lattices.
  SmallVector<AbstractSparseLattice *> argLattices;
  argLattices.reserve(block->getNumArguments());
  bool allAtFixpoint = true;
  for (BlockArgument argument : block->getArguments()) {
    AbstractSparseLattice *argLattice = getLatticeElement(argument);
    allAtFixpoint &= argLattice->isAtFixpoint();
    argLattices.push_back(argLattice);
  }
  // If all argument lattices have reached their fixpoints, then there is
  // nothing to do.
  if (allAtFixpoint)
    return;

  // The argument lattices of entry blocks are set by region control-flow or the
  // callgraph.
  if (block->isEntryBlock()) {
    // Check if this block is the entry block of a callable region.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      const auto *callsites = getOrCreateFor<PredecessorState>(block, callable);
      // If not all callsites are known, conservatively mark all lattices as
      // having reached their pessimistic fixpoints.
      if (!callsites->allPredecessorsKnown())
        return markAllPessimisticFixpoint(argLattices);
      for (Operation *callsite : callsites->getKnownPredecessors()) {
        auto call = cast<CallOpInterface>(callsite);
        for (auto it : llvm::zip(call.getArgOperands(), argLattices))
          join(std::get<1>(it), *getLatticeElementFor(block, std::get<0>(it)));
      }
      return;
    }

    // Check if the lattices can be determined from region control flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
      return visitRegionSuccessors(
          block, branch, block->getParent()->getRegionNumber(), argLattices);
    }

    // Otherwise, we can't reason about the data-flow.
    return visitNonControlFlowArgumentsImpl(block->getParentOp(),
                                            RegionSuccessor(block->getParent()),
                                            argLattices, /*firstIndex=*/0);
  }

  // Iterate over the predecessors of the non-entry block.
  for (Block::pred_iterator it = block->pred_begin(), e = block->pred_end();
       it != e; ++it) {
    Block *predecessor = *it;

    // If the edge from the predecessor block to the current block is not live,
    // bail out.
    auto *edgeExecutable =
        getOrCreate<Executable>(getProgramPoint<CFGEdge>(predecessor, block));
    edgeExecutable->blockContentSubscribe(this);
    if (!edgeExecutable->isLive())
      continue;

    // Check if we can reason about the data-flow from the predecessor.
    if (auto branch =
            dyn_cast<BranchOpInterface>(predecessor->getTerminator())) {
      SuccessorOperands operands =
          branch.getSuccessorOperands(it.getSuccessorIndex());
      for (auto &it : llvm::enumerate(argLattices)) {
        if (Value operand = operands[it.index()]) {
          join(it.value(), *getLatticeElementFor(block, operand));
        } else {
          // Conservatively mark internally produced arguments as having reached
          // their pessimistic fixpoint.
          markPessimisticFixpoint(it.value());
        }
      }
    } else {
      return markAllPessimisticFixpoint(argLattices);
    }
  }
}

void AbstractSparseDataFlowAnalysis::visitRegionSuccessors(
    ProgramPoint point, RegionBranchOpInterface branch,
    Optional<unsigned> successorIndex,
    ArrayRef<AbstractSparseLattice *> lattices) {
  const auto *predecessors = getOrCreateFor<PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");

  for (Operation *op : predecessors->getKnownPredecessors()) {
    // Get the incoming successor operands.
    Optional<OperandRange> operands;

    // Check if the predecessor is the parent op.
    if (op == branch) {
      operands = branch.getSuccessorEntryOperands(successorIndex);
      // Otherwise, try to deduce the operands from a region return-like op.
    } else {
      assert(isTerminator(op) && "expected a terminator");
      if (isRegionReturnLike(op))
        operands = getRegionBranchSuccessorOperands(op, successorIndex);
    }

    if (!operands) {
      // We can't reason about the data-flow.
      return markAllPessimisticFixpoint(lattices);
    }

    ValueRange inputs = predecessors->getSuccessorInputs(op);
    assert(inputs.size() == operands->size() &&
           "expected the same number of successor inputs as operands");

    unsigned firstIndex = 0;
    if (inputs.size() != lattices.size()) {
      if (auto *op = point.dyn_cast<Operation *>()) {
        if (!inputs.empty())
          firstIndex = inputs.front().cast<OpResult>().getResultNumber();
        visitNonControlFlowArgumentsImpl(
            branch,
            RegionSuccessor(
                branch->getResults().slice(firstIndex, inputs.size())),
            lattices, firstIndex);
      } else {
        if (!inputs.empty())
          firstIndex = inputs.front().cast<BlockArgument>().getArgNumber();
        Region *region = point.get<Block *>()->getParent();
        visitNonControlFlowArgumentsImpl(
            branch,
            RegionSuccessor(region, region->getArguments().slice(
                                        firstIndex, inputs.size())),
            lattices, firstIndex);
      }
    }

    for (auto it : llvm::zip(*operands, lattices.drop_front(firstIndex)))
      join(std::get<1>(it), *getLatticeElementFor(point, std::get<0>(it)));
  }
}

const AbstractSparseLattice *
AbstractSparseDataFlowAnalysis::getLatticeElementFor(ProgramPoint point,
                                                     Value value) {
  AbstractSparseLattice *state = getLatticeElement(value);
  addDependency(state, point);
  return state;
}

void AbstractSparseDataFlowAnalysis::markPessimisticFixpoint(
    AbstractSparseLattice *lattice) {
  propagateIfChanged(lattice, lattice->markPessimisticFixpoint());
}

void AbstractSparseDataFlowAnalysis::markAllPessimisticFixpoint(
    ArrayRef<AbstractSparseLattice *> lattices) {
  for (AbstractSparseLattice *lattice : lattices) {
    markPessimisticFixpoint(lattice);
  }
}

void AbstractSparseDataFlowAnalysis::join(AbstractSparseLattice *lhs,
                                          const AbstractSparseLattice &rhs) {
  propagateIfChanged(lhs, lhs->join(rhs));
}

//===----------------------------------------------------------------------===//
// SparseConstantPropagation
//===----------------------------------------------------------------------===//

void SparseConstantPropagation::visitOperation(
    Operation *op, ArrayRef<const Lattice<ConstantValue> *> operands,
    ArrayRef<Lattice<ConstantValue> *> results) {
  LLVM_DEBUG(llvm::dbgs() << "SCP: Visiting operation: " << *op << "\n");

  // Don't try to simulate the results of a region operation as we can't
  // guarantee that folding will be out-of-place. We don't allow in-place
  // folds as the desire here is for simulated execution, and not general
  // folding.
  if (op->getNumRegions())
    return markAllPessimisticFixpoint(results);

  SmallVector<Attribute, 8> constantOperands;
  constantOperands.reserve(op->getNumOperands());
  for (auto *operandLattice : operands)
    constantOperands.push_back(operandLattice->getValue().getConstantValue());

  // Save the original operands and attributes just in case the operation
  // folds in-place. The constant passed in may not correspond to the real
  // runtime value, so in-place updates are not allowed.
  SmallVector<Value, 8> originalOperands(op->getOperands());
  DictionaryAttr originalAttrs = op->getAttrDictionary();

  // Simulate the result of folding this operation to a constant. If folding
  // fails or was an in-place fold, mark the results as overdefined.
  SmallVector<OpFoldResult, 8> foldResults;
  foldResults.reserve(op->getNumResults());
  if (failed(op->fold(constantOperands, foldResults)))
    return markAllPessimisticFixpoint(results);

  // If the folding was in-place, mark the results as overdefined and reset
  // the operation. We don't allow in-place folds as the desire here is for
  // simulated execution, and not general folding.
  if (foldResults.empty()) {
    op->setOperands(originalOperands);
    op->setAttrs(originalAttrs);
    return markAllPessimisticFixpoint(results);
  }

  // Merge the fold results into the lattice for this operation.
  assert(foldResults.size() == op->getNumResults() && "invalid result size");
  for (const auto it : llvm::zip(results, foldResults)) {
    Lattice<ConstantValue> *lattice = std::get<0>(it);

    // Merge in the result of the fold, either a constant or a value.
    OpFoldResult foldResult = std::get<1>(it);
    if (Attribute attr = foldResult.dyn_cast<Attribute>()) {
      LLVM_DEBUG(llvm::dbgs() << "Folded to constant: " << attr << "\n");
      propagateIfChanged(lattice,
                         lattice->join(ConstantValue(attr, op->getDialect())));
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Folded to value: " << foldResult.get<Value>() << "\n");
      AbstractSparseDataFlowAnalysis::join(
          lattice, *getLatticeElement(foldResult.get<Value>()));
    }
  }
}
