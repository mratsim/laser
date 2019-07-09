# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros

# ###########################################
#
#    Stub SIMD for non-vectorized codegen
#
# ###########################################

type
  SimdPrimitives* = enum
    simdSetZero
    simdBroadcast
    simdLoadA
    simdLoadU
    simdStoreA
    simdStoreU
    simdAdd
    simdMul
    simdFma
    simdType

template noop(scalar: untyped): untyped =
  scalar

template unreachable(): untyped =
  {.error: "Unreachable".}

proc genericPrimitives*: array[SimdPrimitives, NimNode] =
  # Use a proc instead of a const
  # to workaround https://github.com/nim-lang/Nim/issues/11668
  [
    simdSetZero:   bindSym"unreachable",
    simdBroadcast: bindSym"noop",
    simdLoadA:     bindSym"unreachable",
    simdLoadU:     bindSym"unreachable",
    simdStoreA:    bindSym"unreachable",
    simdStoreU:    bindSym"unreachable",
    simdAdd:       bindSym"+",
    simdMul:       bindSym"*",
    simdFma:       bindSym"unreachable",
    simdType:      bindSym"unreachable"
  ]
