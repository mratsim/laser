# Apache v2 License
# Mamy Ratsimbazafy

import
  ./conv2d_common,
  ../blas,
  ../../laser/openmp,
  ../../laser/compiler_optim_hints

# Memory efficient convolution based on
# MEC: Memory-efficient Convolution for Deep Neural Network
# Cho et al, 2017
# https://arxiv.org/abs/1706.06873v1

# Unfinished - MEC lowering to fix
# Due to the format being NHWC which is not the
# target format, improving NCHW convolution is probably better

withCompilerOptimHints()

func mec_workspace_size*(
      ishape: TensorShape,
      kshape: KernelShape,
      padding: Padding,
      strides: Strides
    ): int =

  let oshape = conv2d_out_shape(ishape, kshape, padding, strides)
  result = ishape.n * oshape.w * ishape.h * kshape.kW * ishape.c

template `+=`[T](p: var ptr (T or UncheckedArray[T]), offset: int) =
  ## Pointer arithmetic - in-place addition

  # Does: {.emit: "`p` += `offset`;".}
  const psym = p.astToStr()
  const osym = offset.astToStr()
  {.emit: psym & " += " & osym & ";".}

func mec_lowering[T](
      pworkspace: ptr T,   # Mutated
      oshape: TensorShape, # Note: shape of final output, not the im2col buffer
      pi_nhwc: ptr T,      # Input tensor in nhwc format
      ishape: TensorShape,
      kshape: KernelShape,
      padding: Padding,
      strides: Strides
    ) =

  let
    N = ishape.n
    H = ishape.h
    W = ishape.w
    C_in = ishape.c
    Kh = kshape.kH
    Kw = kshape.kW
    pH = padding.h
    pW = padding.w
    sH = strides.h
    sW = strides.w
    OutH = oshape.h
    OutW = oshape.w

  # Lowered MEC tensor of shape [N, outW, H, kW, C]
  # Input tensor of shape [N, H, W, C]

  template wrk_idx(n, ow, h, kW, c: Natural): Natural =
    (((n*OutW + ow)*H + h)*Kw + kW)*C_in + c

  template input_idx(n, h, w, c: Natural): Natural =
    ((n*H + h)*W + w)*C_in + c

  let copy_size = Kw * C_in * sizeof(T)

  for n in 0 ..< N:
    for w in 0 ..< OutW:
      for h in 0 ..< H:
        ## TODO: assumes no padding
        var wstart{.restrict.} = pworkspace
        wstart += wrk_idx(n, w, h, 0, 0)

        var istart{.restrict.} = pi_nhwc
        istart += input_idx(n, h, sW*w, 0)

        # We copy a range of size Kw * Ic according to the equation
        # L[n, w, h, 0:kw, 0:ic] = I[n, h, sw*w:sw*w+kw, 0:ic]
        # L is the lowered mec tensor
        # I the input tensor
        copyMem(wstart, istart, copy_size)

proc conv2d_mec*(
    output: var Tensor[float32], # Output tensor
    oshape: TensorShape,         # Shape of output
    in_nhwc: Tensor[float32],    # Input tensor
    ishape: TensorShape,         # Shape of input
    ker_hwcc: Tensor[float32],   # Convolution filter
    kshape: KernelShape,         # kernel shape (should be const)
    padding: Padding,            # Padding (should be const)
    strides: Strides,            # Strides (should be const)
    pworkspace: ptr float32      # Workspace buffer, can be reused between batches
  ) =

  mec_lowering(
      pworkspace,
      oshape,
      in_nhwc[0].unsafeAddr,
      ishape,
      kshape,
      padding,
      strides
    )

  let
    B = ishape.n  # Batch size, N is used for BLAS MNK notation
    H = ishape.h
    W = ishape.w
    C_in = ishape.c
    C_out = kshape.c_out
    Kh = kshape.kH
    Kw = kshape.kW
    sH = strides.h
    sW = strides.w
    OutH = oshape.h
    OutW = oshape.w

  doAssert C_out == oshape.c
  doAssert B == oshape.n
  doAssert C_in == kshape.c_in

  # Warning: the kernel is in format [kH,kW,C_in,C_out]
  # instead of the usual [C_out,C_in,kH,kW]
  #
  # Workspace will have shape [N, outW, H, kW, C]

  # A naive MEC would have the input as format "HNWC"
  # from collapsed [outH * [N*outW, kC]]
  # Computed as batched matrix multiplication of lowered and kernel matrices:
  # for h in 0 ..< outH:
  #   output[h] = lowered[N*outW, (h:h+kH)*kW*C_in] * kernel[kH*kW*C_in, kC]
  #
  # So either we shuffle the dimensions after computation (and need a workspace buffer
  # for the permutation)
  # or
  # we can iterate on n and h and do smaller matrix multiplications
  # We only choose the latter scheme if those small matrix multiplications
  # are bigger than a threshold.
  const threshold = 100
  let osize = B * OutH * OutW * C_out
  let wsize = mec_workspace_size(ishape, kshape, padding, strides)

  # if OutW < 100 and (osize < wsize):
  when true:
    let
      M = B*OutW
      K = Kh*Kw*C_in
      N = C_out

    for h in 0 ..< OutH:
      var pwrk{.restrict.} = pworkspace
      pwrk += sH*Kw*C_in*h
      gemm(
        rowMajor, noTranspose, noTranspose,
        M, N, K,
        1'f32,
        pwrk, K,
        ker_hwcc[0].unsafeAddr, N,
        0, output[0].addr, N
      )
    # Now we need to convert output[H, N, W, C] to [N, H, W, C]
    copyMem(pworkspace, output[0].addr, osize*sizeof(float32))
    template wrk_idxA(oh, n, owco: Natural): Natural =
      (oh*N + n)*OutW*C_out + owco
    template out_idxA(n, oh, owco: Natural): Natural =
      (n*OutH + oh)*OutW*C_out + owco

    let pwrk{.restrict.} = cast[ptr UncheckedArray[float32]](pworkspace)

    # TODO: Tiled loop
    for n in 0 ..< N:
      for h in 0 ..< OutH:
        for wc in 0 ..< OutW*C_out:
          output[out_idxA(n, h, wc)] = pwrk[wrk_idxA(h, n, wc)]

func kernel_to_hwcc*(
        out_kernel: ptr float32,
        in_kernel: ptr float32,
        kshape: KernelShape
      ) =
  let
    C_out = kshape.c_out
    C_in = kshape.c_in
    Kh = kshape.kh
    Kw = kshape.kw

    po{.restrict.} = cast[ptr UncheckedArray[float32]](out_kernel)
    pi{.restrict.} = cast[ptr UncheckedArray[float32]](in_kernel)

  template oidx(kh, kw, ci, co: Natural): Natural =
    ((kH*Kw + kw)*C_in + ci)*C_out + co
  template iidx(co, ci, kh, kw: Natural): Natural =
    ((co*C_in + ci)*Kh + kh)*Kw + kw

  for kh in 0 ..< Kh:
    for kw in 0 ..< Kw:
      for ci in 0 ..< C_in:
        for co in 0 ..< C_out:
          po[oidx(kh, kw, ci, co)] = pi[iidx(co, ci, kh, kw)]

when isMainModule:
  import ../../laser/primitives/swapaxes

  conv_impl_check(output, oshape, input, ishape, kernel, kshape, padding, strides):

    let buffer_size = mec_workspace_size(ishape, kshape, padding, strides)
    var workspace = newSeq[float32](buffer_size)
    let pworkspace = workspace[0].addr

    var ker_hwcc = newSeq[float32](kernel.len)
    kernel_to_hwcc(ker_hwcc[0].addr, kernel[0].unsafeAddr, kshape)

    var input_nhwc = newSeq[float32](input.len)
    nchw2nhwc(input[0].unsafeAddr, input_nhwc[0].addr, ishape[0], ishape[1], ishape[2], ishape[3])

    conv2d_mec(
      output, oshape,
      input, ishape,
      kernel, kshape,
      padding,
      strides,
      pworkspace
    )

    echo workspace
