# Laser & Arraymancer
# Copyright (c) 2017-2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

const LASER_MEM_ALIGN*{.intdefine.} = 64
static:
  assert LASER_MEM_ALIGN != 0, "Alignment " & $LASER_MEM_ALIGN & "must be a power of 2"
  assert (LASER_MEM_ALIGN and (LASER_MEM_ALIGN - 1)) == 0, "Alignment " & $LASER_MEM_ALIGN & "must be a power of 2"

template withCompilerOptimHints*() =
  {.pragma: align_array, codegenDecl: "$# $# __attribute__((aligned(" & $LASER_MEM_ALIGN & ")))".}
  when not defined(vcc):
    {.pragma: restrict, codegenDecl: "$# __restrict__ $#".}
  else:
    {.pragma: restrict, codegenDecl: "$# __restrict $#".}
  {.pragma: align_function, codegenDecl: "__attribute__((assume_aligned(" & $LASER_MEM_ALIGN & "))) $# $#$#".}

const withBuiltins = defined(gcc) or defined(clang) or defined(icc)

when withBuiltins:
  proc builtin_assume_aligned[T](data: ptr T, n: csize): ptr T {.importc: "__builtin_assume_aligned", noDecl.}

when defined(cpp):
  proc static_cast[T](input: T): T
    {.importcpp: "static_cast<'0>(@)".}

template assume_aligned*[T](data: ptr T): ptr T =
  when defined(cpp) and withBuiltins: # builtin_assume_aligned returns void pointers, this does not compile in C++, they must all be typed
    static_cast builtin_assume_aligned(data, LASER_MEM_ALIGN)
  elif withBuiltins:
    builtin_assume_aligned(data, LASER_MEM_ALIGN)
  else:
    data

template pragma_ivdep() =
  ## Tell the compiler to ignore unproven loop dependencies
  ## such as "a[i] = a[i + k] * c;" if k is unknown, as it introduces a loop
  ## dependency if it's negative
  ## https://software.intel.com/en-us/node/524501
  ##
  ## Placeholder
  # We don't expose that as it only works on C for loop. Nim only generates while loop
  # except when using OpenMP. But the OpenMP "simd" already achieves the same as ivdep.
  when defined(gcc):
    {.emit: "#pragma GCC ivdep".}
  else: # Supported on ICC and Cray
    {.emit: "pragma ivdep".}
