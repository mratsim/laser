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

  proc foobar(a: LuxNode, b, c: LuxNode): tuple[bar: LuxNode, baz, buzz: LuxNode] =

    let foo = a + b + c

    # Don't use in-place updates
    # https://github.com/nim-lang/Nim/issues/11637
    let bar = foo * 2

    var baz = foo * 3
    var buzz = baz

    buzz += a * 1000
    baz += b
    buzz += b

    result.bar = bar
    result.baz = baz
    result.buzz = buzz

  proc foobar(a: int, b, c: int): tuple[bar, baz, buzz: int] =
    echo "Overloaded proc to test bindings"
    discard

  generate foobar:
    proc foobar(a: seq[float32], b, c: seq[float32]): tuple[bar: seq[float32], baz, buzz: seq[float32]]

  # Note to use aligned store, SSE requires 16-byte alignment and AVX 32-byte alignment
  # Unfortunately there is no way with normal seq to specify that (pending destructors)
  # As a hack, we use the unaligned load and store simd, and a required alignment of 4,
  # in practice we define our own tensor type
  # with aligned allocator

  import sequtils

  let
    len = 10
    u = newSeqWith(len, 1'f32)
    v = newSeqWith(len, 2'f32)
    w = newSeqWith(len, 3'f32)

  let (pim, pam, poum) = foobar(u, v, w)

  echo pim  # 12
  echo pam  # 20
  echo poum # 10020
