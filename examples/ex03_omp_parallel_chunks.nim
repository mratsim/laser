# ################################################################
#
#          Example of using OpenMP parallel routine
#  This one returns a chunk offset and length and can be combine
#     with threadsafe routine that accept a ptr + len pair
#
# ################################################################

import ../laser/openmp
import math, random, sequtils

proc doOp(s: var seq[float32]) {.noInline.}=
  omp_parallel_chunks_default(s.len, chunk_offset, chunk_size):
    # Create a block range for each thread
    # Each thread can work on it's own range
    for idx in chunk_offset ..< chunk_offset + chunk_size:
      s[idx] = sin(s[idx]) + cos(s[idx])

proc main() =
  randomize(42) # Reproducibility
  var x = newSeqWith(300_000_000, float32 rand(1.0))

  x.doOp()

  echo x[0 ..< 10]
  echo x[^10 .. ^1]

main()

# Compile with
# `nim c -d:release -r -d:openmp -o:build/omp examples/ex02_omp_parallel_for.nim`
# If needed pass the custom compiler flags.
# For example on MacOS, we need GCC from Homebrew as the default Clang does not support OpenMP:
# `--cc:gcc --gcc.exe:"/usr/local/bin/gcc-7" --gcc.linkerexe:"/usr/local/bin/gcc-7"`
