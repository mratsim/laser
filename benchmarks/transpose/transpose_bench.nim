# ##########################################
# Benchmarking tools
import random, times, stats, strformat, math, sequtils

proc warmup() =
  # Warmup - make sure cpu is on max perf
  let start = epochTime()
  var foo = 123
  for i in 0 ..< 300_000_000:
    foo += i*i mod 456
    foo = foo mod 789

  # Compiler shouldn't optimize away the results as cpuTime rely on sideeffects
  let stop = epochTime()
  echo &"Warmup: {stop - start:>4.4f} s, result {foo} (displayed to avoid compiler optimizing warmup away)"

template printStats(name: string, output: openarray) {.dirty.} =
  echo "\n" & name
  echo &"Collected {stats.n} samples in {global_stop - global_start:>4.3f} seconds"
  echo &"Average time: {stats.mean * 1000 :>4.3f} ms"
  echo &"Stddev  time: {stats.standardDeviationS * 1000 :>4.3f} ms"
  echo &"Min     time: {stats.min * 1000 :>4.3f} ms"
  echo &"Max     time: {stats.max * 1000 :>4.3f} ms"
  echo &"Perf:         {req_ops.float / stats.mean / float(10^9):>4.3f} GFLOP/s"
  echo "\nDisplay output[0] to make sure it's not optimized away"
  echo output[0] # Prevents compiler from optimizing stuff away

template bench(name: string, initialisation, body: untyped) {.dirty.}=
  block: # Actual bench
    var stats: RunningStat
    let global_start = epochTime()
    for _ in 0 ..< nb_samples:
      initialisation
      let start = epochTime()
      body
      let stop = epochTime()
      stats.push stop - start
    let global_stop = epochTime()
    printStats(name, output)

# #############################################
# Params
import ./transpose_common, ../blas

const
  M     =  2000
  N     =  1000

const
  ashape: MatrixShape = (M, N)

let req_ops = M*N
let req_bytes = sizeof(float32) * M*N

let out_shape: MatrixShape = (N, M)
let out_size = out_shape.M * out_shape.N

# #############################################

# TODO: could not import: cblas_somatcopy
# proc benchBLAS(a: seq[float32], nb_samples: int) =
#   var output = newSeq[float32](out_size)
#   bench("BLAS omatcopy benchmark"):
#     # Initialisation, not measured apart for the "Collected n samples in ... seconds"
#     zeroMem(output[0].addr, out_size) # We zero memory between computation
#   do:
#     # Main work
#     omatcopy(
#       rowMajor, noTranspose,
#       M, N, 1,
#       a[0].unsafeaddr, N,
#       output[0].unsafeAddr, N,
#     )

# ###########################################

when defined(fast_math):
  {.passC:"-ffast-math".}

when defined(march_native):
  {.passC:"-march=native".}

when isMainModule:
  randomize(42) # For reproducibility
  warmup()
  echo ""
  echo "A matrix shape: " & $ashape
  echo "Output shape: " & $out_shape
  echo &"Required number of operations: {req_ops.float / float(10^6):>9.3f} millions"
  echo &"Required bytes:                {req_bytes.float / float(10^6):>9.3f} MB"
  echo &"Arithmetic intensity:          {req_ops.float / req_bytes.float:>9.3f} FLOP/byte"
  block:
    let a = newSeqWith(M*N, float32 rand(1.0))

    # benchBLAS(a, nb_samples = 1000)
