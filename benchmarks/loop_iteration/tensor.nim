# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

import
  sequtils, random,
  ./metadata, ./utils

type
  Tensor*[T] = object
    ## Tensor data structure stored on Cpu
    ##   - ``shape``: Dimensions of the tensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the tensor. Note: offset can be negative, in particular for slices.
    ##   - ``storage``: A data storage for the tensor
    ##
    ## Warning ⚠:
    ##   Assignment ```var a = b``` does not copy the data. Data modification on one tensor will be reflected on the other.
    ##   However modification on metadata (shape, strides or offset) will not affect the other tensor.
    shape*: Metadata
    strides*: Metadata
    offset*: int
    storage*: CpuStorage[T]

  CpuStorage*{.shallow.}[T] = object
    ## Data storage for the tensor, copies are shallow by default
    data*: seq[T]

func size*(t: Tensor): int {.inline.} =
  t.shape.product

func rank*(t: Tensor): range[0 .. MAXRANK] {.inline.} =
  t.shape.len

template tensor(result: var Tensor, shape: openarray|Metadata) =
  result.shape = shape.toMetadata
  result.strides.len = result.rank

  var accum = 1
  for i in countdown(shape.len - 1, 0):
    result.strides[i] = accum
    accum *= shape[i]

func newTensor*[T](shape: varargs[int]): Tensor[T] =
  tensor(result, shape)
  result.storage.data = newSeq[T](shape.product)

func newTensor*[T](shape: Metadata): Tensor[T] =
  tensor(result, shape)
  result.storage.data = newSeq[T](shape.product)

proc randomTensor*[T](shape: openarray[int], max: T): Tensor[T] =
  tensor(result, shape)
  result.storage.data = newSeqWith(shape.product, T(rand(max)))

func getIndex[T](t: Tensor[T], idx: varargs[int]): int {.inline.} =
  ## Convert [i, j, k, l ...] to the memory location referred by the index
  result = t.offset
  for i in 0 ..< t.shape.len:
    {.unroll.} # I'm sad this doesn't work yet
    result += t.strides[i] * idx[i]

func `[]`*[T](t: Tensor[T], idx: varargs[int]): T {.inline.}=
  ## Index tensor
  t.storage.data[t.getIndex(idx)]

func `[]`*[T](t: var Tensor[T], idx: varargs[int]): var T {.inline.}=
  ## Index tensor
  t.storage.data[t.getIndex(idx)]

func `[]=`*[T](t: var Tensor[T], idx: varargs[int], val: T) {.inline.}=
  ## Index tensor
  t.storage.data[t.getIndex(idx)] = val

func `[]`*[T: SomeNumber](x: T, idx: varargs[int]): T {.inline.}=
  ## "Index" scalars
  x

func shape*(x: SomeNumber): array[1, int] {.inline.} =
  [1]

func dataPtr*[T](t: Tensor[T]): ptr UncheckedArray[T] {.inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the offset start of the data.
  ##       Return value supports array indexing.
  cast[ptr UncheckedArray[T]](t.storage.data[t.offset].unsafeAddr)

func is_C_contiguous*(t: Tensor): bool {.inline.}=
  ## Check if the tensor follows C convention / is row major
  var z = 1
  for i in countdown(t.shape.high,0):
    # 1. We should ignore strides on dimensions of size 1
    # 2. Strides always must have the size equal to the product of the next dimensions
    if t.shape[i] != 1 and t.strides[i] != z:
        return false
    z *= t.shape[i]
  return true

func transpose*(t: Tensor): Tensor {.inline.} =
  t.shape.reversed(result.shape)
  t.strides.reversed(result.strides)
  result.offset = t.offset
  result.storage = t.storage
