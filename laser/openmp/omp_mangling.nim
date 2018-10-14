# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Compile-time name mangling for OpenMP thresholds
# Workaround https://github.com/nim-lang/Nim/issues/9365
# and https://github.com/nim-lang/Nim/issues/9366
import random
from strutils import toHex

var mangling_rng {.compileTime.} = initRand(0x1337DEADBEEF)
var current_suffix {.compileTime.} = ""

proc omp_suffix*(genNew: static bool = false): static string =
  ## genNew:
  ##   if false, return the last suffix
  ##   else return a fresh one

  if genNew:
    current_suffix = mangling_rng.rand(high(uint32)).toHex
  result = current_suffix
