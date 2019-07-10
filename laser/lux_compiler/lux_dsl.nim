# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./frontend/lux_frontend,
  ./dsl/primitives

from ./core/lux_types import LuxNode

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

    # Domain
    var i, j: LuxNode
    newLuxIterDomain(i, 0, a.shape(0))
    newLuxIterDomain(j, 0, a.shape(1))

    # Avoid in-place update of implicit result ref address
    # https://github.com/nim-lang/Nim/issues/11637
    var bar: LuxNode
    newLuxMutTensor(bar)

    bar[i, j] = a[i, j] + b[i, j] + c[i, j]

    # Update result
    result.bar = bar

  generate foobar:
    proc foobar(a: Tensor[float32], b, c: Tensor[float32]): tuple[bar: Tensor[float32], baz, buzz: Tensor[float32]]
