# Laser & Arraymancer
# Copyright (c) 2017-2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Types and low level primitives for tensors

import
  ../dynamic_stack_array, ../compiler_optim_hints,
  sugar

type
  RawImmutableView*[T] = distinct ptr UncheckedArray[T]
  RawMutableView*[T] = distinct ptr UncheckedArray[T]

  Metadata* = DynamicStackArray[int]

  Tensor*[T] = object                  # Total stack: 128 bytes = 2 cache-lines
    shape*: Metadata                   # 56 bytes
    strides*: Metadata                 # 56 bytes
    offset*: int                       # 8 bytes
    storage*: CpuStorage[T]            # 8 bytes

  CpuStorage*[T] = ref object          # Total heap: 25 bytes = 1 cache-line
    raw_data*: ptr UncheckedArray[T]   # 8 bytes
    memalloc*: pointer                 # 8 bytes
    memowner*: bool                    # 1 byte

func rank*(t: Tensor): Natural {.inline.} =
  t.shape.len

func size*(t: Tensor): Natural =
  t.shape.product

func is_C_contiguous*(t: Tensor): bool {.inline.} =
  ## Check if the tensor follows C convention / is row major
  var z = 1
  for i in countdown(t.shape.rank - 1,0):
    # 1. We should ignore strides on dimensions of size 1
    # 2. Strides always must have the size equal to the product of the next dimensions
    if t.shape[i] != 1 and t.strides[i] != z:
        return false
    z *= t.shape[i]
  return true

# ##################
# Raw pointer access
# ##################

# RawImmutableView and RawMutableView make sure that a non-mutable tensor
# is not mutated through it's raw pointer.
#
# Unfortunately there is no way to also prevent those from escaping their scope
# and outliving their source tensor (via `lent` destructors)
# and keeping the `restrict` and `alignment`
# optimization hints https://github.com/nim-lang/Nim/issues/7776

withCompilerOptimHints()

func unsafe_raw_data*[T](t: Tensor[T]): RawImmutableView[T] {.inline.} =
  ## Unsafe: the pointer can outlive the input tensor
  let raw_pointer{.restrict.} = assume_aligned t.storage.raw_data
  result = cast[type result](raw_pointer[t.offset].addr)

func unsafe_raw_data*[T](t: var Tensor[T]): RawMutableView[T] {.inline.} =
  ## Unsafe: the pointer can outlive the input tensor
  let raw_pointer{.restrict.} = assume_aligned t.storage.raw_data
  result = cast[type result](raw_pointer[t.offset].addr)

func `[]`*[T](v: RawImmutableView[T], idx: int): T {.inline.}=
  distinctBase(type v)(v)[idx]

func `[]`*[T](v: RawMutableView[T], idx: int): var T {.inline.}=
  distinctBase(type v)(v)[idx]
