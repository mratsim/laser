# Apache v2 License
# Mamy Ratsimbazafy

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
import ./gemm_common, ../blas

const
  M     =  224
  K     =  16*3*20*3*3 # to make required ops similar to conv
  N     =  224

const
  ashape: MatrixShape = (M, K)
  bshape: MatrixShape = (K, N)

let req_ops = gemm_required_ops(ashape, bshape)
let req_bytes = sizeof(float32) * gemm_required_data(ashape, bshape)

let out_shape: MatrixShape = gemm_out_shape(ashape, bshape)
let out_size = out_shape.M * out_shape.N

# #############################################

proc benchBLAS(a, b: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  bench("BLAS benchmark"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(output[0].addr, out_size) # We zero memory between computation
  do:
    # Main work
    gemm(
      rowMajor, noTranspose, noTranspose,
      M, N, K,
      1, a[0].unsafeaddr, K,
      b[0].unsafeAddr, N,
      0, output[0].addr, N
    )

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
  echo "B matrix shape: " & $bshape
  echo "Output shape: " & $out_shape
  echo &"Required number of operations: {req_ops.float / float(10^6):>9.3f} millions"
  echo &"Required bytes:                {req_bytes.float / float(10^6):>9.3f} MB"
  echo &"Arithmetic intensity:          {req_ops.float / req_bytes.float:>9.3f} FLOP/byte"
  block:
    let a  = newSeqWith(M*K, float32 rand(1.0))
    let b = newSeqWith(K*N, float32 rand(1.0))

    benchBLAS(a, b, nb_samples = 20)

# Warmup: 1.2052 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 224, N: 8640)
# B matrix shape: (M: 8640, N: 224)
# Output shape: (M: 224, N: 224)
# Required number of operations:   867.041 millions
# Required bytes:                   15.483 MB
# Arithmetic intensity:             56.000 FLOP/byte

# BLAS benchmark
# Collected 20 samples in 0.206 seconds
# Average time: 10.284 ms
# Stddev  time: 1.709 ms
# Min     time: 7.978 ms
# Max     time: 13.609 ms
# Perf:         84.307 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 2158.1591796875
