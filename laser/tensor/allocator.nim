# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./datatypes, ../compiler_optim_hints, typetraits

# Storage backend allocation primitives

proc finalizer[T](storage: CpuStorage[T]) =
  static: assert T.supportsCopyMem, "Tensors of seq, strings, ref types and types with non-trivial destructors cannot be finalized by this proc"

  if storage.memowner and not storage.memalloc.isNil:
    storage.memalloc.deallocShared()

withCompilerOptimHints()

func align_raw_data(T: typedesc, p: pointer): ptr UncheckedArray[T] =
  static: assert T.supportsCopyMem, "Tensors of seq, strings, ref types and types with non-trivial destructors cannot be aligned"

  let address = cast[ByteAddress](p)
  let aligned_ptr{.restrict.} = block: # We cannot directly apply restrict to the default "result"
    if (address and (LASER_MEM_ALIGN - 1)) == 0:
      assume_aligned cast[ptr UncheckedArray[T]](address)
    else:
      let offset = LASER_MEM_ALIGN - (address and (LASER_MEM_ALIGN - 1))
      assume_aligned cast[ptr UncheckedArray[T]](address +% offset)
  return aligned_ptr

proc allocCpuStorage*[T](storage: var CpuStorage[T], size: int) =
  ## Allocate aligned memory to hold `size` elements of type T.
  ## If T does not supports copyMem, it is also zero-initialized.
  ## I.e. Tensors of seq, strings, ref types or types with non-trivial destructors
  ## are always zero-initialized. This prevents potential GC issues.
  when T.supportsCopyMem:
    new(storage, finalizer[T])
    storage.memalloc = allocShared0(sizeof(T) * size + LASER_MEM_ALIGN - 1)
    storage.memowner = true
    storage.raw_data = align_raw_data(T, storage.memalloc)
  else: # Always 0-initialize Tensors of seq, strings, ref types and types with non-trivial destructors
    new(storage)
    newSeq[T](storage.raw_data, size)
