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
import ./conv2d_common, ./conv2d_direct_convolution

const
  N     =  16
  C_in  =   3
  C_out =  20
  H     = 224
  W     = 224
  kH    =   3
  kW    =   3
  padding: Padding = (0, 0)
  strides: Strides = (1, 1)

const
  ishape: TensorShape = (N, C_in, H, W)
  kshape: KernelShape = (C_out, C_in, kH, kW)
  in_size = N * C_in * H * W

let req_ops = conv2d_required_ops(
  ishape, kshape, padding, strides
)
let req_bytes = sizeof(float32) * conv2d_required_data(
  ishape, kshape, padding, strides
)

let out_shape = conv2d_out_shape(
  ishape, kshape, padding, strides
)
let out_size = out_shape.n * out_shape.c * out_shape.h * out_shape.w

# #############################################
let input  = newSeqWith(N*C_in*H*W, float32 rand(1.0))
let kernel = newSeqWith(C_out*C_in*kH*kW, float32 rand(1.0))

proc benchDirect(input, filter: Tensor[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  bench("Direct convolution"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(output[0].addr, out_size) # We zero memory between computation
  do:
    # Main work
    conv2d_direct(output, input, ishape, kernel, kshape, padding, strides)

# ###########################################

when defined(fast_math):
  {.passC:"-ffast-math".}

when defined(march_native):
  {.passC:"-march=native".}

when isMainModule:
  randomize(42) # For reproducibility
  warmup()
  echo ""
  echo "Input shape: " & $ishape
  echo "Kernel shape: " & $kshape
  echo "Padding and strides: ", padding, strides
  echo "Output shape: " & $out_shape
  echo &"Required number of operations: {req_ops.float / float(10^6):>9.3f} millions"
  echo &"Required bytes:                {req_bytes.float / float(10^6):>9.3f} MB"
  echo &"Arithmetic intensity:          {req_ops.float / req_bytes.float:>9.3f} FLOP/byte"
  block:
    benchDirect(input, kernel, nb_samples = 20)

# CPU: i5-5257U https://ark.intel.com/products/84985/Intel-Core-i5-5257U-Processor-3M-Cache-up-to-3-10-GHz-
# Frequency: 2.7GHz, Turbo 3.1
# Cores: 2
# flop per cycle: 32 (AVX2 - 2*(8*add + 8*mul))
#
# Peak theoretical GFLOPS        = 2.7 * 2 * 32 = 172.8
# Peak theoretical GFLOPS, turbo = 3.1 * 2 * 32 = 198.4

# Warmup: 1.2015 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Input shape: (n: 16, c: 3, h: 224, w: 224)
# Kernel shape: (c_out: 20, c_in: 3, kH: 3, kW: 3)
# Padding and strides: (h: 0, w: 0)(h: 1, w: 1)
# Output shape: (n: 16, c: 20, h: 222, w: 222)
# Required number of operations:   851.628 millions
# Required bytes:                   72.719 MB
# Arithmetic intensity:             11.711 FLOP/byte

# Direct convolution
# Collected 20 samples in 6.146 seconds
# Average time: 306.128 ms
# Stddev  time: 37.475 ms
# Min     time: 274.167 ms
# Max     time: 408.455 ms
# Perf:         2.782 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 6.69718599319458
