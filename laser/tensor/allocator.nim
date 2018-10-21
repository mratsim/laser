# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./datatypes, ../compiler_optim_hints

# Storage backend allocation primitives

withCompilerOptimHints()

proc finalizer[T](storage: CpuStorage[T]) {.hot.}=
  if storage.memowner and not storage.memalloc.isNil:
    storage.memalloc.deallocShared()

func align_raw_data(T: typedesc, p: pointer): ptr UncheckedArray[T] {.hot, aligned_ptr_result, malloc.} =
  let address = cast[ByteAddress](p)
  let aligned_ptr{.restrict.} = block: # We cannot directly apply restrict to the default "result"
    if (address and (LASER_MEM_ALIGN - 1)) == 0:
      assume_aligned cast[ptr UncheckedArray[T]](address)
    else:
      let offset = LASER_MEM_ALIGN - (address and (LASER_MEM_ALIGN - 1))
      assume_aligned cast[ptr UncheckedArray[T]](address +% offset)
  return aligned_ptr

proc allocCpuStorage*(T: typedesc, size: int): CpuStorage[T] {.hot.}=
  new(result, finalizer[T])
  result.memalloc = allocShared0(sizeof(T) * size + LASER_MEM_ALIGN - 1)
  result.memowner = true
  result.raw_data = align_raw_data(T, result.memalloc)
