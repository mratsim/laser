# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./ast/ast_compiler,
  ./ast/ast_definition

# ###########################
#
#         Tests
#
# ###########################

when isMainModule:
  import
    sequtils,
    ../tensor/[datatypes, allocator, initialization]

  proc toTensor[T](s: seq[T]): Tensor[T] =
    var size: int
    initTensorMetadata(result, size, [s.len])
    allocCpuStorage(result.storage, size)
    result.copyFromRaw(s[0].unsafeAddr, s.len)

  proc `$`[T](t: Tensor[T]): string =
    var tmp = newSeq[T](t.size)
    copyMem(tmp[0].addr, cast[ptr T](t.unsafe_raw_data), t.size * sizeof(T))
    result = $tmp

  # We intentionally have a tricky non-canonical function signature
  proc foobar(a: LuxNode, b, c: LuxNode): tuple[bar: LuxNode, baz, buzz: LuxNode] =
    let foo = a + b + c

    # Don't use in-place updates
    # https://github.com/nim-lang/Nim/issues/11637
    let bar = foo * 2

    var baz = foo * 3
    var buzz = baz

    buzz += a * 10000
    baz += b
    buzz += b

    result.bar = bar
    result.baz = baz
    result.buzz = buzz

  proc foobar(a: int, b, c: int): tuple[bar, baz, buzz: int] =
    echo "Overloaded proc to test bindings"
    discard

  generate foobar:
    proc foobar(a: Tensor[float32], b, c: Tensor[float32]): tuple[bar: Tensor[float32], baz, buzz: Tensor[float32]]

  generate foobar:
    proc foobar(a: Tensor[float64], b, c: Tensor[float64]): tuple[bar: Tensor[float64], baz, buzz: Tensor[float64]]

  block: # float32
    let
      len = 10
      u = newSeqWith(len, 1'f32).toTensor()
      v = newSeqWith(len, 2'f32).toTensor()
      w = newSeqWith(len, 3'f32).toTensor()

    let (pim, pam, poum) = foobar(u, v, w)

    echo pim  # 12
    echo pam  # 20
    echo poum # 10020

  block: # float64
    let
      len = 10
      u = newSeqWith(len, 1'f64).toTensor()
      v = newSeqWith(len, 2'f64).toTensor()
      w = newSeqWith(len, 3'f64).toTensor()

    let (pim, pam, poum) = foobar(u, v, w)

    echo pim  # 12
    echo pam  # 20
    echo poum # 10020
