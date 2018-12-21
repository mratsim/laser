# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

packageName   = "Laser"
version       = "0.0.1"
author        = "Mamy André-Ratsimbazafy"
description   = "High Performance Computing and Image Toolbox: SIMD, JIT Assembler, OpenMP, runtime CPU feature detection, optimised machine learning primitives"
license       = "Apache License 2.0"

### Dependencies
requires "nim >= 0.19.2"

### Helper functions
proc test(name: string, defaultLang = "c") =
  if not dirExists "build":
    mkDir "build"
  --run
  switch("out", ("./build/" & name))
  setCommand defaultLang, "tests/" & name & ".nim"

### tasks
task test, "Run all tests":
  test "all_tests"
