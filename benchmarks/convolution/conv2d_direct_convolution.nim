# Apache v2 License
# Mamy Ratsimbazafy

import
  ./conv2d_common,
  ../../laser/openmp

proc conv2d_direct*[T](
    oim: var Tensor[T],     # Output images
    iim: Tensor[T],         # Input images
    ishape: TensorShape,    # Shape of a single image
    kernel: Tensor[T],      # filter kernel
    kshape: KernelShape,    # kernel shape (should be const)
    padding: Padding,       # Padding (should be const)
    strides: Strides        # Strides (should be const)
  ) =
  ## oim must be zero-initialized
  # Reminder: convolution deep learning == cross-correlation signal processing

  assert ishape.c == kshape.c_in
  let out_shape = conv2d_out_shape(ishape, kshape, padding, strides)
  assert oim.len == out_shape.n * out_shape.c * out_shape.h * out_shape.w

  let
    N = ishape.n
    H = ishape.h
    W = ishape.w
    outH = out_shape.h
    outW = out_shape.w
    kdata = cast[ptr UncheckedArray[T]](kernel[0].unsafeaddr)

  let # Should be const but padding.h causes problem and padding[0] indexing
      # doesn't work in generic proc
    C_out = kshape.c_out
    C_in = kshape.c_in
    kH = kshape.kH
    kW = kshape.kW
    pH = padding.h
    pW = padding.w
    sH = strides.h
    sW = strides.w

  let odata = cast[ptr UncheckedArray[T]](oim[0].addr)
  let idata = cast[ptr UncheckedArray[T]](iim[0].unsafeaddr)

  const arithmetic_intensity = 12 # number of FLOP per byte for the convolution
  const flop_per_elem = arithmetic_intensity div sizeof(T)
  let parallelize = OMP_MEMORY_BOUND_GRAIN_SIZE div flop_per_elem < outH * outW

  for n in 0 ..< N:
    for co in 0 ..< C_out:
      for ci in 0 ..< C_in:
        # We parallelize over the image height to deal with cases
        # where we have a single image or a low number of channels

        # We assume no false sharing with a proper grain size
        omp_parallel_if(parallelize):
          omp_for(oh, outH, use_simd = true, nowait=true):                  # output height
            let ih = sH * oh                                                # input height
            for ow in 0 ..< outW:                                           # output width
              let iw = sH * ow                                              # input width
              # The following should be loop hoisted by the compiler
              let oidx = ow + outW * (oh + outH * (co + C_out * n))         # odata[n][co][oh][ow]
              # Entering conv kernel region
              for krow in 0 ..< kH:
                let row = ih + krow - pH
                if row <% H:     # Unsigned '<' does 0 < row < H.
                  for kcol in 0 ..< kW:
                    let col = iw + kcol - pW
                    if col <% W: # Unsigned '<' does 0 < row < H.
                      let iidx = col + W * (row + H * (ci + C_in * n))      # idata[n][ci][row][col]
                      let kidx = kcol + kW * (krow + kH * (ci + C_in * co)) # kdata[co][ci][krow][kcol]
                      odata[oidx] += idata[iidx] * kdata[kidx]

when isMainModule:
  conv_impl_check(output, oshape, input, ishape, kernel, kshape, padding, strides):
    conv2d_direct(
      output,
      input, ishape,
      kernel, kshape,
      padding,
      strides
    )
