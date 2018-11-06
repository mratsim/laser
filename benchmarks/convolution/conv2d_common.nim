# Apache v2 License
# Mamy Ratsimbazafy

type
  ImageShape* = tuple[c, h, w: int]              # Channel/Color, Height, Width
  KernelShape* = tuple[c_out, c_in, kH, kW: int] # Channel out, Channel in, kernel height, kernel out
  PaddingShape* = tuple[h, w: int]
  StrideShape* = tuple[h, w: int]
  ## We don't suport dilation for benchmarking

func conv2d_out_shape*(
      input: ImageShape,
      kernel: KernelShape,
      padding: PaddingShape,
      strides: StrideShape
    ): ImageShape =

  ## Dilation, unsupported
  const dH = 1
  const dW = 1

  result.c = kernel.c_out
  result.h = 1 + (input.h + 2*padding.h - (((kernel.kH-1) * dH) + 1) div strides.h)
  result.w = 1 + (input.w + 2*padding.w - (((kernel.kW-1) * dW) + 1) div strides.w)
