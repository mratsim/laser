# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros,
  # Internal
  ./platforms/platform_common

when defined(i386) or defined(x86_64):
  import ./platforms/platform_x86

export SimdPrimitives
export SimdArch, SimdAlignment, SimdMap

func elemsPerVector*(arch: SimdArch, T: typedesc): int {.compileTime.}=
  SimdWidth[arch] div sizeof(T)

macro parseSIMDs(s: static string): set[SimdArch] =
  result = parseExpr(s)

const
  LASER_SIMDS {.strdefine.} = "{ArchGeneric, x86_SSE, x86_AVX}"
  LaserTargetSimds* = parseSIMDs(LASER_SIMDS)
