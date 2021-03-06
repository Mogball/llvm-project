//===-- tsd_shared.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_TSD_SHARED_H_
#define SCUDO_TSD_SHARED_H_

#include "linux.h" // for getAndroidTlsPtr()
#include "tsd.h"

namespace scudo {

template <class Allocator, u32 TSDsArraySize, u32 DefaultTSDCount>
struct TSDRegistrySharedT {
  void initLinkerInitialized(Allocator *Instance) {
    Instance->initLinkerInitialized();
    CHECK_EQ(pthread_key_create(&PThreadKey, nullptr), 0); // For non-TLS
    for (u32 I = 0; I < TSDsArraySize; I++)
      TSDs[I].initLinkerInitialized(Instance);
    const u32 NumberOfCPUs = getNumberOfCPUs();
    setNumberOfTSDs((NumberOfCPUs == 0) ? DefaultTSDCount
                                        : Min(NumberOfCPUs, DefaultTSDCount));
    Initialized = true;
  }
  void init(Allocator *Instance) {
    memset(this, 0, sizeof(*this));
    initLinkerInitialized(Instance);
  }

  void unmapTestOnly() {
    setCurrentTSD(nullptr);
    pthread_key_delete(PThreadKey);
  }

  ALWAYS_INLINE void initThreadMaybe(Allocator *Instance,
                                     UNUSED bool MinimalInit) {
    if (LIKELY(getCurrentTSD()))
      return;
    initThread(Instance);
  }

  ALWAYS_INLINE TSD<Allocator> *getTSDAndLock(bool *UnlockRequired) {
    TSD<Allocator> *TSD = getCurrentTSD();
    DCHECK(TSD);
    *UnlockRequired = true;
    // Try to lock the currently associated context.
    if (TSD->tryLock())
      return TSD;
    // If that fails, go down the slow path.
    if (TSDsArraySize == 1U) {
      // Only 1 TSD, not need to go any further.
      // The compiler will optimize this one way or the other.
      TSD->lock();
      return TSD;
    }
    return getTSDAndLockSlow(TSD);
  }

  void disable() {
    Mutex.lock();
    for (u32 I = 0; I < TSDsArraySize; I++)
      TSDs[I].lock();
  }

  void enable() {
    for (s32 I = static_cast<s32>(TSDsArraySize - 1); I >= 0; I--)
      TSDs[I].unlock();
    Mutex.unlock();
  }

  bool setOption(Option O, sptr Value) {
    if (O == Option::MaxTSDsCount)
      return setNumberOfTSDs(static_cast<u32>(Value));
    // Not supported by the TSD Registry, but not an error either.
    return true;
  }

private:
  ALWAYS_INLINE void setCurrentTSD(TSD<Allocator> *CurrentTSD) {
#if _BIONIC
    *getAndroidTlsPtr() = reinterpret_cast<uptr>(CurrentTSD);
#elif SCUDO_LINUX
    ThreadTSD = CurrentTSD;
#else
    CHECK_EQ(
        pthread_setspecific(PThreadKey, reinterpret_cast<void *>(CurrentTSD)),
        0);
#endif
  }

  ALWAYS_INLINE TSD<Allocator> *getCurrentTSD() {
#if _BIONIC
    return reinterpret_cast<TSD<Allocator> *>(*getAndroidTlsPtr());
#elif SCUDO_LINUX
    return ThreadTSD;
#else
    return reinterpret_cast<TSD<Allocator> *>(pthread_getspecific(PThreadKey));
#endif
  }

  bool setNumberOfTSDs(u32 N) {
    ScopedLock L(MutexTSDs);
    if (N < NumberOfTSDs)
      return false;
    if (N > TSDsArraySize)
      N = TSDsArraySize;
    NumberOfTSDs = N;
    NumberOfCoPrimes = 0;
    // Compute all the coprimes of NumberOfTSDs. This will be used to walk the
    // array of TSDs in a random order. For details, see:
    // https://lemire.me/blog/2017/09/18/visiting-all-values-in-an-array-exactly-once-in-random-order/
    for (u32 I = 0; I < N; I++) {
      u32 A = I + 1;
      u32 B = N;
      // Find the GCD between I + 1 and N. If 1, they are coprimes.
      while (B != 0) {
        const u32 T = A;
        A = B;
        B = T % B;
      }
      if (A == 1)
        CoPrimes[NumberOfCoPrimes++] = I + 1;
    }
    return true;
  }

  void initOnceMaybe(Allocator *Instance) {
    ScopedLock L(Mutex);
    if (LIKELY(Initialized))
      return;
    initLinkerInitialized(Instance); // Sets Initialized.
  }

  NOINLINE void initThread(Allocator *Instance) {
    initOnceMaybe(Instance);
    // Initial context assignment is done in a plain round-robin fashion.
    const u32 Index = atomic_fetch_add(&CurrentIndex, 1U, memory_order_relaxed);
    setCurrentTSD(&TSDs[Index % NumberOfTSDs]);
    Instance->callPostInitCallback();
  }

  NOINLINE TSD<Allocator> *getTSDAndLockSlow(TSD<Allocator> *CurrentTSD) {
    // Use the Precedence of the current TSD as our random seed. Since we are
    // in the slow path, it means that tryLock failed, and as a result it's
    // very likely that said Precedence is non-zero.
    const u32 R = static_cast<u32>(CurrentTSD->getPrecedence());
    u32 N, Inc;
    {
      ScopedLock L(MutexTSDs);
      N = NumberOfTSDs;
      DCHECK_NE(NumberOfCoPrimes, 0U);
      Inc = CoPrimes[R % NumberOfCoPrimes];
    }
    if (N > 1U) {
      u32 Index = R % N;
      uptr LowestPrecedence = UINTPTR_MAX;
      TSD<Allocator> *CandidateTSD = nullptr;
      // Go randomly through at most 4 contexts and find a candidate.
      for (u32 I = 0; I < Min(4U, N); I++) {
        if (TSDs[Index].tryLock()) {
          setCurrentTSD(&TSDs[Index]);
          return &TSDs[Index];
        }
        const uptr Precedence = TSDs[Index].getPrecedence();
        // A 0 precedence here means another thread just locked this TSD.
        if (Precedence && Precedence < LowestPrecedence) {
          CandidateTSD = &TSDs[Index];
          LowestPrecedence = Precedence;
        }
        Index += Inc;
        if (Index >= N)
          Index -= N;
      }
      if (CandidateTSD) {
        CandidateTSD->lock();
        setCurrentTSD(CandidateTSD);
        return CandidateTSD;
      }
    }
    // Last resort, stick with the current one.
    CurrentTSD->lock();
    return CurrentTSD;
  }

  pthread_key_t PThreadKey;
  atomic_u32 CurrentIndex;
  u32 NumberOfTSDs;
  u32 NumberOfCoPrimes;
  u32 CoPrimes[TSDsArraySize];
  bool Initialized;
  HybridMutex Mutex;
  HybridMutex MutexTSDs;
  TSD<Allocator> TSDs[TSDsArraySize];
#if SCUDO_LINUX && !_BIONIC
  static THREADLOCAL TSD<Allocator> *ThreadTSD;
#endif
};

#if SCUDO_LINUX && !_BIONIC
template <class Allocator, u32 TSDsArraySize, u32 DefaultTSDCount>
THREADLOCAL TSD<Allocator>
    *TSDRegistrySharedT<Allocator, TSDsArraySize, DefaultTSDCount>::ThreadTSD;
#endif

} // namespace scudo

#endif // SCUDO_TSD_SHARED_H_
