# Apache v2 License
# Mamy Ratsimbazafy

import
  ./conv2d_common,
  ../blas,
  ../../laser/openmp,
  ../../laser/compiler_optim_hints

func im2col_workspace_size*(
      ishape: TensorShape,
      kshape: KernelShape,
      padding: Padding,
      strides: Strides
    ): int =

  let out_shape = conv2d_out_shape(ishape, kshape, padding, strides)

  result = ishape.c * kshape.kH * kshape.kW *
              out_shape.h * out_shape.w

# Note on pointer arithmetic for `+=`:
#   - Emitting with interpolation in a template requires
#     symbols to be defined locally
#   - Using a func instead would be tricky as well as
#     var param are passed by hidden pointer and `+=`
#     will increment the hidden pointer instead of what we want

func `+`[T](p: ptr (T or UncheckedArray[T]), offset: int): type(p) {.inline.}=
  # Pointer arithmetic
  {.emit: "`result` = `p` + `offset`;".}

template `+=`[T](p: var ptr (T or UncheckedArray[T]), offset: int) =
  # Pointer arithmetic
  const psym = p.astToStr()
  const osym = offset.astToStr()
  # {.emit: "`p` += `offset`;".}
  {.emit: psym & " += " & osym & ";".}

withCompilerOptimHints()

func im2col*[T](
      pworkspace: ptr T,            # Mutated
      oshape: TensorShape,          # Note: shape of final output, not the im2col buffer
      pinput: ptr UncheckedArray[T];
      ishape: TensorShape,
      kshape: KernelShape,
      padding: Padding,
      strides: Strides
    ) =
  var # local pointer, emit doesn't properly work with shadowing
    lpworkspace{.restrict.} = pworkspace
    lpinput{.restrict.} = pinput
  let
    C = ishape.c
    H = ishape.h
    W = ishape.w
    kH = kshape.kH
    kW = kshape.kW
    pH = padding.h
    pW = padding.w
    sH = strides.h
    sW = strides.w
    outH = oshape.h
    outW = oshape.w

  # We reorganize data into [C_in * kH * kW, outH * outW] matrix
  let channel_size = H * W
  for _ in 0 ..< C:
    for krow in 0 ..< kH:
      for kcol in 0 ..< kW:
        var row = -pH + krow # * dilation
        for _ in 0 ..< outH:
          if not(row <% H):  # Unsigned '<' does 0 < row < H.
            for _ in 0 ..< outW:
              lpworkspace[] = 0.T
              lpworkspace += 1
          else:
            var col = -pW + kcol # * dilation
            for _ in 0 ..< outW:
              if col <% W:
                lpworkspace[] = lpinput[row * W + col]
              else:
                lpworkspace[] = 0
              lpworkspace += 1
              col += sW
          row += sH
    lpinput += channel_size

proc conv2d_im2col*(
    output: var Tensor[float32], # Output tensor
    oshape: TensorShape,         # Shape of output
    input: Tensor[float32],      # Input tensor
    ishape: TensorShape,         # Shape of input
    kernel: Tensor[float32],     # Convolution filter
    kshape: KernelShape,         # kernel shape (should be const)
    padding: Padding,            # Padding (should be const)
    strides: Strides,            # Strides (should be const)
    pworkspace: ptr float32      # Workspace buffer, can be reused between batches
  ) =
  ## output must be zero-initialized
  ## workspace is a buffer of minimal size that can hold
  ##   [C * kH * kW, outH * outW]
  const
    groups = 1    # unused, for group convolution
    alpha = 1'f32 # for GEMM C = αAB + βC
    beta = 0'f32

  assert oshape.c == kshape.c_out

  let
    B = ishape.n     # batch size, N is use for BLAS MNK notation
    C_in = ishape.c
    C_out = oshape.c
    H = ishape.h
    W = ishape.w
    C_in_per_group = C_in div groups
    C_out_per_group = C_out div groups
    kH = kshape.kH
    kW = kshape.kW
    outH = oshape.h
    outW = oshape.w

    is1x1 = kH*kW == 1

  for n in 0 ..< B:
    let ioffset = n * C_in * H * W
    let pinput{.restrict.} = cast[ptr UncheckedArray[float32]](input[ioffset].unsafeAddr)
    if not is1x1:
      im2col(
        pworkspace,
        oshape,
        pinput,
        ishape,
        kshape,
        padding,
        strides,
      )
    for g in 0 ..< groups:
      let koffset = g * kH * kW * C_in_per_group * C_out_per_group
      let woffset = g * (kH * kW * C_in_per_group) * (outH * outW)
      let ooffset = (n * C_out + g * C_out_per_group) * (outH * outW)

      let pkernel{.restrict.} = kernel[koffset].unsafeAddr
      let lpworkspace{.restrict.} = block:
        if is1x1:
          pinput[0].addr
        else:
          pworkspace + woffset
      let poutput{.restrict.} = output[ooffset].addr

      let M = C_out_per_group
      let K = C_in_per_group * kH * kW
      let N = outH * outW

      # O = αFW + βO
      # O: output [M, N],
      # F: Filter kernel [M, K],
      # W: workspace [K, N]

      gemm(rowMajor, noTranspose, noTranspose,
        M, N, K,
        alpha, pkernel, K,
        lpworkspace, N,
        beta, poutput, N
      )

when isMainModule:
  conv_impl_check(output, oshape, input, ishape, kernel, kshape, padding, strides):

    let buffer_size = im2col_workspace_size(ishape, kshape, padding, strides)
    var workspace = newSeq[float32](buffer_size)
    let pworkspace = workspace[0].addr

    conv2d_im2col(
      output, oshape,
      input, ishape,
      kernel, kshape,
      padding,
      strides,
      pworkspace
    )
