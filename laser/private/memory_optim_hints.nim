# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

template withMemoryOptimHints*() =
  when not defined(js):
    {.pragma: align64, codegenDecl: "$# $# __attribute__((aligned(64)))".}
    {.pragma: restrict, codegenDecl: "$# __restrict__ $#".}
  else:
    {.pragma: align64.}
    {.pragma: restrict.}

const withBuiltins = defined(gcc) or defined(clang)

when withBuiltins:
  proc builtin_assume_aligned[T](data: ptr T, n: csize): ptr T {.importc: "__builtin_assume_aligned",noDecl.}

when defined(cpp):
  proc static_cast[T](input: T): T
    {.importcpp: "static_cast<'0>(@)".}

template assume_aligned*[T](data: ptr T, n: csize): ptr T =
  when defined(cpp) and withBuiltins: # builtin_assume_aligned returns void pointers, this does not compile in C++, they must all be typed
    static_cast builtin_assume_aligned(data, n)
  elif withBuiltins:
    builtin_assume_aligned(data, n)
  else:
    data
