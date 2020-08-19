//===- rocm-runtime-wrappers.cpp - MLIR ROCM runner wrapper library -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the ROCM library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <numeric>
#include <iostream>
#include <iomanip>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include "hip/hip_runtime.h"

#define HIP_REPORT_IF_ERROR(expr)                                              \
  [](hipError_t result) {                                                      \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = hipGetErrorName(result);                                \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    llvm::errs() << "'" << #expr << "' failed with '" << name << "'\n";        \
  }(expr)

namespace {
// Context object that buffers GPU modules, functions, temporary storage.
struct Runtime {
  // Load a module and cache it.
  void loadModule(hipModule_t *module, void *data) {
    // Load the module during the first execution.
    if(moduleList.count(data) == 0) {
      HIP_REPORT_IF_ERROR(hipModuleLoadData(module, data));
      moduleList[data] = *module;
    }
    *module = moduleList[data];
  }

  // Get a function an cache it.
  void getFunction(hipFunction_t *function, hipModule_t module, const char *name) {
    // Get the function during the first execution.
    if(functionList.count(name) == 0) {
      HIP_REPORT_IF_ERROR(hipModuleGetFunction(function, module, name));
      functionList[name] = *function;
    }
    *function = functionList[name];
  }

  // Get the default stream.
  void createStream(hipStream_t *stream) {
    if(streamList.empty()) {
      hipStream_t stream;
      HIP_REPORT_IF_ERROR(hipStreamCreate(&stream));
      streamList.push_back(stream);
    }
    *stream = streamList.back();
  }

  // Allocate GPU device memory.
  void allocMem(hipDeviceptr_t *ptr, size_t size) {
    // Allocate storage if free list contains no matching allocation.
    if(tempList.count(size) == 0 || tempList[size].empty()) {
      HIP_REPORT_IF_ERROR(hipMalloc(ptr, size));
      return;
    }
    // Return existing allocation.
    *ptr = tempList[size].back();
    tempList[size].pop_back();
  }

  // Free GPU device memory.
  void freeMem(hipDeviceptr_t ptr) {
    hipDeviceptr_t allocPtr;
    size_t allocSize = 0;
    // Get the size of the allocation.
    HIP_REPORT_IF_ERROR(hipMemGetAddressRange(&allocPtr, &allocSize, ptr));
    tempList[allocSize].push_back(ptr);
  }

  static Runtime &getInstance() {
    thread_local Runtime runtime;
    return runtime;
  }

private:
  std::vector<hipStream_t> streamList;
  llvm::DenseMap<void*, hipModule_t> moduleList;
  llvm::DenseMap<const char*, hipFunction_t> functionList;
  llvm::DenseMap<size_t, std::vector<hipDeviceptr_t>> tempList;
};
} // anonymous namespace

extern "C" hipModule_t mgpuModuleLoad(void *data) {
  hipModule_t module = nullptr;
  Runtime::getInstance().loadModule(&module, data);
  return module;
}

extern "C" hipFunction_t mgpuModuleGetFunction(hipModule_t module,
                                               const char *name) {
  hipFunction_t function = nullptr;
  Runtime::getInstance().getFunction(&function, module, name);
  return function;
}

// The wrapper uses intptr_t instead of ROCM's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" void mgpuLaunchKernel(hipFunction_t function, intptr_t gridX,
                                 intptr_t gridY, intptr_t gridZ,
                                 intptr_t blockX, intptr_t blockY,
                                 intptr_t blockZ, int32_t smem,
                                 hipStream_t stream, void **params,
                                 void **extra) {
  hipEvent_t start, stop;
  HIP_REPORT_IF_ERROR(hipEventCreate(&start));
  HIP_REPORT_IF_ERROR(hipEventCreate(&stop));
  
  HIP_REPORT_IF_ERROR(hipEventRecord(start, stream));
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ, smem,
                                            stream, params, extra));
  HIP_REPORT_IF_ERROR(hipEventRecord(stop, stream));
  HIP_REPORT_IF_ERROR(hipEventSynchronize(stop));
  float duration = 0.0;
  HIP_REPORT_IF_ERROR(hipEventElapsedTime(&duration, start, stop));
  std::cout << std::setprecision(5) << "-> kernel time [ms]: " << duration << "\n";
}

extern "C" void *mgpuStreamCreate() {
  hipStream_t stream = nullptr;
  Runtime::getInstance().createStream(&stream);
  return stream;
}

extern "C" void mgpuStreamSynchronize(hipStream_t stream) {
  return HIP_REPORT_IF_ERROR(hipStreamSynchronize(stream));
}

extern "C" void mgpuMemAlloc(hipDeviceptr_t *ptr, uint64_t size) {
  Runtime::getInstance().allocMem(ptr, size);
}

extern "C" void mgpuMemFree(hipDeviceptr_t ptr) {
  Runtime::getInstance().freeMem(ptr);
}

/// Helper functions for writing mlir example code

// Allows to register byte array with the ROCM runtime. Helpful until we have
// transfer functions implemented.
extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  HIP_REPORT_IF_ERROR(hipHostRegister(ptr, sizeBytes, /*flags=*/0));
}

// Allows to register a MemRef with the ROCM runtime. Initializes array with
// value. Helpful until we have transfer functions implemented.
template <typename T>
void mgpuMemHostRegisterMemRef(T *pointer, llvm::ArrayRef<int64_t> sizes,
                               llvm::ArrayRef<int64_t> strides, T value) {
  assert(sizes.size() == strides.size());
  llvm::SmallVector<int64_t, 4> denseStrides(strides.size());

  std::partial_sum(sizes.rbegin(), sizes.rend(), denseStrides.rbegin(),
                   std::multiplies<int64_t>());
  auto count = denseStrides.front();

  // Only densely packed tensors are currently supported.
  std::rotate(denseStrides.begin(), denseStrides.begin() + 1,
              denseStrides.end());
  denseStrides.back() = 1;
  assert(strides == llvm::makeArrayRef(denseStrides));

  std::fill_n(pointer, count, value);
  mgpuMemHostRegister(pointer, count * sizeof(T));
}

extern "C" void mgpuMemHostRegisterFloat(int64_t rank, void *ptr) {
  auto *desc = static_cast<StridedMemRefType<float, 1> *>(ptr);
  auto sizes = llvm::ArrayRef<int64_t>(desc->sizes, rank);
  auto strides = llvm::ArrayRef<int64_t>(desc->sizes + rank, rank);
  mgpuMemHostRegisterMemRef(desc->data + desc->offset, sizes, strides, 1.23f);
}

extern "C" void mgpuMemHostRegisterInt32(int64_t rank, void *ptr) {
  auto *desc = static_cast<StridedMemRefType<int32_t, 1> *>(ptr);
  auto sizes = llvm::ArrayRef<int64_t>(desc->sizes, rank);
  auto strides = llvm::ArrayRef<int64_t>(desc->sizes + rank, rank);
  mgpuMemHostRegisterMemRef(desc->data + desc->offset, sizes, strides, 123);
}

template <typename T>
void mgpuMemGetDevicePointer(T *hostPtr, T **devicePtr) {
  HIP_REPORT_IF_ERROR(hipSetDevice(0));
  HIP_REPORT_IF_ERROR(
      hipHostGetDevicePointer((void **)devicePtr, hostPtr, /*flags=*/0));
}

extern "C" StridedMemRefType<float, 1>
mgpuMemGetDeviceMemRef1dFloat(float *allocated, float *aligned, int64_t offset,
                              int64_t size, int64_t stride) {
  float *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" StridedMemRefType<int32_t, 1>
mgpuMemGetDeviceMemRef1dInt32(int32_t *allocated, int32_t *aligned,
                              int64_t offset, int64_t size, int64_t stride) {
  int32_t *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}
