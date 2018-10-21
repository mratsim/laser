# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../openmp/[omp_parallel, omp_tuning],
  ../compiler_optim_hints,
  ../strided_iteration/map_forEach,
  ./datatypes, ./allocator,
  typetraits

## Initialization and copy routines

func toMetadata(s: varargs[int]): Metadata =
  result.len = s.len
  for i in 0..<s.len:
    result.data[i] = s[i]

template toMetadata(m: Metadata): Metadata = m

template initTensorMetadataImpl(result: var Tensor, size: var int, shape: openarray[int]|Metadata) =
  ## We don't use a proc directly due to https://github.com/nim-lang/Nim/issues/6529
  result.shape = shape.toMetadata
  result.strides.len = result.rank

  size = 1
  for i in countdown(shape.len - 1, 0):
    result.strides[i] = size
    size *= shape[i]

func initTensorMetadata*(result: var Tensor, size: var int, shape: openarray[int]) =
  initTensorMetadataImpl(result, size, shape)

func initTensorMetadata*(result: var Tensor, size: var int, shape: Metadata) =
  initTensorMetadataImpl(result, size, shape)

proc deepCopy*[T](dst: var Tensor[T], src: Tensor[T]) =
  ## Performs a deep copy of y and copies it into x.
  ## Deepcopy is recursive including for ref types and custom types
  ## that implement deepCopy.
  ##
  ## Note that if x was already initialized with a ``storage``,
  ## the storage will be detached from x. This does not write
  ## into existing storage.

  var size: int
  initTensorMetadata(dst, size, src.shape)
  dst.storage = allocCpuStorage(T, size)

  when T.supportsCopyMem:
    # We use memcpy, due to SIMD optimizations in memcpy,
    # we require higher parallelization thresholds
    if src.is_C_contiguous:
      omp_parallel_chunks(
            size, chunk_offset, chunk_size,
            OMP_MEMORY_BOUND_THRESHOLD * 4, OMP_MEMORY_BOUND_GRAIN_SIZE * 4,
            use_simd = false):
        copyMem(
          dst.storage.raw_data[chunk_offset],
          src.storage.raw_data[chunk_offset],
          chunk_size
          )
    else:
      forEachStrided d in dst, s in src:
        d = s
  else:
    # If the type doesn't supports memcpy,
    # we assume we can't use OpenMP and we need
    # recursive deepCopy
    forEachSerial d in dst, s in src:
      deepCopy(d, s) # recursive deepcopy

proc copyFrom*[T](dst: var Tensor[T], src: Tensor[T]) =
  ## Copy the source tensor into the destination tensor.
  ## Both should have the same shape. If destination tensor is a view
  ## only the data exposed by the view is modified.
  ##
  ## This is useful to update subslices of an existing tensor.
  ##
  ## ⚠ Warning:
  ##    The data exposed by the destination tensor will be overwritten.
  ##    If destination tensor is a view, all views of that data will be changed.
  ##    They however conserve their shape and strides.
  ##
  ## Note: The copy is not recursive.
  when T.supportsCopyMem:
    # We use memcpy, due to SIMD optimizations in memcpy,
    # we require higher parallelization thresholds
    if src.is_C_contiguous:
      omp_parallel_chunks(
            src.size, chunk_offset, chunk_size,
            OMP_MEMORY_BOUND_THRESHOLD * 4, OMP_MEMORY_BOUND_GRAIN_SIZE * 4,
            use_simd = false):
        copyMem(
          dst.storage.raw_data[chunk_offset].addr,
          src.storage.raw_data[chunk_offset].unsafeAddr,
          chunk_size
          )
    else:
      forEachStrided d in dst, s in src:
        d = s
  else:
    # If the type doesn't supports memcpy,
    # we assume we can't use OpenMP
    forEachSerial d in dst, s in src:
      d = s # non-recursive copy

proc setZero*[T](t: var Tensor[T], check_contiguous: static bool = true) =
  ## Reset/initialize the tensor data to binary zero.
  ## The tensor metadata is not touched.
  ## Input tensor must be contiguous.
  ##
  ## ⚠ Warning:
  ##    The data of the input tensor will be overwritten.
  ##    If destination tensor is a view, all views of that data will be changed.
  ##    They however conserve their shape and strides.
  when check_contiguous:
    if unlikely(not t.is_C_contiguous):
      # TODO: error model - https://github.com/numforge/laser/issues/2
      # + If using exceptions, display the tensor ident with astToStr
      raise newException(ValueError, "Input tensor is not contiguous.")

  when not T.supportsCopyMem:
    t.storage.reset()
  else:
    omp_parallel_chunks(
          t.size, chunk_offset, chunk_size,
          OMP_MEMORY_BOUND_THRESHOLD * 4, OMP_MEMORY_BOUND_GRAIN_SIZE * 4,
          use_simd = false):
      zeroMem(
        t.storage.raw_data[chunk_offset].addr,
        chunk_size
        )

