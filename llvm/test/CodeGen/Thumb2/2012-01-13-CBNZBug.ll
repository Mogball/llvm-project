; RUN: llc < %s -mtriple=thumbv7-apple-ios -relocation-model=pic -frame-pointer=all -mcpu=cortex-a8 | FileCheck %s
; rdar://10676853

%struct.Dict_node_struct = type { ptr, ptr, ptr, ptr, ptr }
%struct.Word_file_struct = type { [60 x i8], i32, ptr }
%struct.Exp_struct = type { i8, i8, i8, i8, %union.anon }
%union.anon = type { ptr }
%struct.E_list_struct = type { ptr, ptr }

@lookup_list = external hidden unnamed_addr global ptr, align 4

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind

define hidden fastcc void @rdictionary_lookup(ptr %dn, ptr nocapture %s) nounwind ssp {
; CHECK-LABEL: rdictionary_lookup:
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %if.then10, %entry
  %dn.tr = phi ptr [ %dn, %entry ], [ %9, %if.then10 ]
  %cmp = icmp eq ptr %dn.tr, null
  br i1 %cmp, label %if.end11, label %if.end

if.end:                                           ; preds = %tailrecurse
  %string = getelementptr inbounds %struct.Dict_node_struct, ptr %dn.tr, i32 0, i32 0
  %0 = load ptr, ptr %string, align 4
  br label %while.cond.i

while.cond.i:                                     ; preds = %while.body.i, %if.end
  %1 = phi ptr [ %s, %if.end ], [ %incdec.ptr.i, %while.body.i ]
  %storemerge.i = phi ptr [ %0, %if.end ], [ %incdec.ptr6.i, %while.body.i ]
  %2 = load i8, ptr %1, align 1
  %cmp.i = icmp eq i8 %2, 0
  %.pre.i = load i8, ptr %storemerge.i, align 1
  br i1 %cmp.i, label %lor.lhs.false.i, label %land.end.i

land.end.i:                                       ; preds = %while.cond.i
  %cmp4.i = icmp eq i8 %2, %.pre.i
  br i1 %cmp4.i, label %while.body.i, label %while.end.i

while.body.i:                                     ; preds = %land.end.i
  %incdec.ptr.i = getelementptr inbounds i8, ptr %1, i32 1
  %incdec.ptr6.i = getelementptr inbounds i8, ptr %storemerge.i, i32 1
  br label %while.cond.i

while.end.i:                                      ; preds = %land.end.i
  %cmp8.i = icmp eq i8 %2, 42
  br i1 %cmp8.i, label %if.end3, label %lor.lhs.false.i

lor.lhs.false.i:                                  ; preds = %while.end.i, %while.cond.i
  %3 = phi i8 [ %2, %while.end.i ], [ 0, %while.cond.i ]
  %cmp11.i = icmp eq i8 %.pre.i, 42
  br i1 %cmp11.i, label %if.end3, label %dict_match.exit

dict_match.exit:                                  ; preds = %lor.lhs.false.i
  %cmp14.i = icmp eq i8 %3, 46
  %conv16.i = sext i8 %3 to i32
  %.conv16.i = select i1 %cmp14.i, i32 0, i32 %conv16.i
  %cmp18.i = icmp eq i8 %.pre.i, 46
  %conv22.i = sext i8 %.pre.i to i32
  %cond24.i = select i1 %cmp18.i, i32 0, i32 %conv22.i
  %sub.i = sub nsw i32 %.conv16.i, %cond24.i
  %cmp1 = icmp sgt i32 %sub.i, -1
  br i1 %cmp1, label %if.end3, label %if.then10

if.end3:                                          ; preds = %dict_match.exit, %lor.lhs.false.i, %while.end.i
; CHECK: %if.end3
; CHECK: cmp
; CHECK-NOT: cbnz
  %storemerge1.i3 = phi i32 [ %sub.i, %dict_match.exit ], [ 0, %lor.lhs.false.i ], [ 0, %while.end.i ]
  %right = getelementptr inbounds %struct.Dict_node_struct, ptr %dn.tr, i32 0, i32 4
  %4 = load ptr, ptr %right, align 4
  tail call fastcc void @rdictionary_lookup(ptr %4, ptr %s)
  %cmp4 = icmp eq i32 %storemerge1.i3, 0
  br i1 %cmp4, label %if.then5, label %if.end8

if.then5:                                         ; preds = %if.end3
  %call6 = tail call fastcc ptr @xalloc(i32 20)
  %5 = bitcast ptr %call6 to ptr
  %6 = bitcast ptr %dn.tr to ptr
  tail call void @llvm.memcpy.p0.p0.i32(ptr align 4 %call6, ptr align 4 %6, i32 16, i1 false)
  %7 = load ptr, ptr @lookup_list, align 4
  %right7 = getelementptr inbounds i8, ptr %call6, i32 16
  %8 = bitcast ptr %right7 to ptr
  store ptr %7, ptr %8, align 4
  store ptr %5, ptr @lookup_list, align 4
  br label %if.then10

if.end8:                                          ; preds = %if.end3
  %cmp9 = icmp slt i32 %storemerge1.i3, 1
  br i1 %cmp9, label %if.then10, label %if.end11

if.then10:                                        ; preds = %if.end8, %if.then5, %dict_match.exit
  %left = getelementptr inbounds %struct.Dict_node_struct, ptr %dn.tr, i32 0, i32 3
  %9 = load ptr, ptr %left, align 4
  br label %tailrecurse

if.end11:                                         ; preds = %if.end8, %tailrecurse
  ret void
}

; Materializable
declare hidden fastcc ptr @xalloc(i32) nounwind ssp
