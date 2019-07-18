# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./frontend/lux_frontend,
  ./dsl/primitives

from ./core/lux_types import Iter, Invariant, Fn

# ###########################
#
#         Tests
#
# ###########################

when isMainModule:
  import
    sequtils,
    ../tensor/[datatypes, allocator, initialization],

    # TODO: How to bindSym to "[]" / nkBracketExpr
    ../dynamic_stack_arrays

  proc `$`[T](t: Tensor[T]): string =
    var tmp = newSeq[T](t.size)
    copyMem(tmp[0].addr, cast[ptr T](t.unsafe_raw_data), t.size * sizeof(T))
    result = $tmp

  proc `[]`[T](t: Tensor[T], idx: varargs[int]): T =
    # Hack for this example
    assert t.rank == 2
    assert idx.len == 2
    t.storage.raw_buffer[idx[0] * t.strides[0] + idx[1] * t.strides[1]]

  proc `[]=`[T](t: Tensor[T], idx: varargs[int], val: T) =
    # Hack for this example
    assert t.rank == 2
    assert idx.len == 2
    t.storage.raw_buffer[idx[0] * t.strides[0] + idx[1] * t.strides[1]] = val

  proc foobar(a, b, c: Fn): Fn =

    # Iteration Domain
    var i, j: Iter

    # Avoid in-place update of implicit result ref address
    # https://github.com/nim-lang/Nim/issues/11637
    var bar: Fn

    bar[i, j] = a[i, j] + b[i, j] + c[i, j]

    # Update result
    result = bar

  generate foobar:
    proc foobar(a: Tensor[float32], b, c: Tensor[float32]): Tensor[float32]

  let
    u = [[float32 1, 1, 1],
         [float32 1, 1, 1],
         [float32 1, 1, 1]].toTensor()
    v = [[float32 2, 2, 2],
         [float32 2, 2, 2],
         [float32 2, 2, 2]].toTensor()
    w = [[float32 3, 3, 3],
         [float32 3, 3, 3],
         [float32 3, 3, 3]].toTensor()

  let r = foobar(u, v, w)
  echo r
