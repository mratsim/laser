# Apache v2 License
# Mamy Ratsimbazafy

import sequtils

type
  TensorShape* = tuple[n, c, h, w: int]          # BatchSize, Channel/Color, Height, Width
  KernelShape* = tuple[c_out, c_in, kH, kW: int] # Channel out, Channel in, kernel height, kernel out
  Padding* = tuple[h, w: int]
  Strides* = tuple[h, w: int]
  ## We don't support dilation for benchmarking

  Tensor*[T] = seq[T]

func conv2d_out_shape*(
      input: TensorShape,
      kernel: KernelShape,
      padding: Padding,
      strides: Strides
    ): TensorShape =

  let
    iH = input.h
    iW = input.w
    kH = kernel.kH
    kW = kernel.kW

  let # Should be const but padding.h causes problem and padding[0] indexing
      # doesn't work in generic proc
    pH = padding.h
    pW = padding.w
    sH = strides.h
    sW = strides.w

  doAssert 0 < sH and sH < iH
  doAssert 0 < sW and sW < iW

  ## Dilation, unsupported
  const dH = 1
  const dW = 1

  result.n = input.n
  result.c = kernel.c_out
  result.h = 1 + (iH + 2*pH - (((kH-1) * dH) + 1)) div sH
  result.w = 1 + (iW + 2*pW - (((kW-1) * dW) + 1)) div sW

func conv2d_required_ops*(
      input: TensorShape,
      kernel: KernelShape,
      padding: Padding,
      strides: Strides
    ): int =
  # - A non-padded convolution of strides 1 requires `kH * kW * C_in` operation per sliding window
  # - It slides over the **result** (and not the input) and so requires `kH * kW * C_in * outH * outW`
  # - The output is multichannel: `kH * kW * C_in * outH*outW * C_out`
  # - And on multiple images: so `kH * kW * C_in * outH*outW * C_out * N`
  #
  # TODO: check the numbers:
  #   - A base operation is output[n][outC][outH][outW] += input[n][inC][row][col] * kernel[outC][inC][kh][kw]
  #     It's composed of 1 addition, 1 multiplication
  #   - we consider that padding doesn't impact op count.
  #   - striding will divide the work done on a dimension by that much

  let
    out_shape = conv2d_out_shape(input, kernel, padding, strides)
    N = input.n
    C_in = input.c
    oH = out_shape.h
    oW = out_shape.w
    C_out = kernel.c_out
    kH = kernel.kH
    kW = kernel.kW

  doAssert C_in == kernel.c_in

  result = N * C_out * kH * kW * C_in *
            oH *
            oW *
            2 # 1 add + 1 mul

func conv2d_required_data*(
      input: TensorShape,
      kernel: KernelShape,
      padding: Padding,
      strides: Strides
    ): int =
  let
    out_shape = conv2d_out_shape(input, kernel, padding, strides)
    N = input.n
    C_in = input.c
    iH = input.h
    iW = input.w
    oH = out_shape.h
    oW = out_shape.w
    C_out = kernel.c_out
    kH = kernel.kH
    kW = kernel.kW

  result = C_out * C_in * kH * kW + # kernel size
            N * C_in * iH * iW +    # input size
            N * C_out * oH * oW     # output size

iterator flatIter*[T](s: openarray[T]): auto {.noSideEffect.}=
  for item in s:
    when item is array|seq:
      for subitem in flatIter(item):
        yield subitem
    else:
      yield item

func toTensor*[T](oa: openarray[T]): auto =
  ## Convert to a flattened tensor/image
  toSeq(flatiter(oa))

proc toString*(im: Tensor, shape: TensorShape): string =
  for n in 0 ..< shape.n:
    for c in 0 ..< shape.c:
      for h in 0 ..< shape.h:
        for w in 0 ..< shape.w:
          let idx = w + shape.w * (h + shape.h * (c + shape.c * n))
          result.add $im[idx] & '\t'
        result.add '\n'
      result.add '\n'
    result.add '\n'

# #######################################

when isMainModule:
  block:
    let img =  [[1, 2, 0, 0],
                [5, 3, 0, 4],
                [0, 0, 0, 7],
                [9, 3, 0, 0]].toTensor()

    echo img.toString((1, 1, 3, 4))

# #######################################

template conv_impl_check*(
    output, oshape,
    input, ishape,
    kernel, kshape,
    padding,
    strides,
    conv_call_body: untyped) =
  block:
    let input{.inject.} = [[float32 1, 2, 0, 0],
                          [float32 5, 3, 0, 4],
                          [float32 0, 0, 0, 7],
                          [float32 9, 3, 0, 0]].toTensor()
    let ishape{.inject.}: TensorShape = (1, 1, 4, 4)

    let kernel{.inject.} = [[float32 1, 1, 1],
                            [float32 1, 1, 0],
                            [float32 1, 0, 0]].toTensor()
    const kshape{.inject.}: KernelShape = (1, 1, 3, 3)

    let target = [[float32 1,  8,  5,  0],
                  [float32 8, 11,  5,  4],
                  [float32 8, 17, 10, 11],
                  [float32 9, 12, 10,  7]].toTensor()

    let
      padding{.inject.} = (1, 1)
      strides{.inject.} = (1, 1)

    let oshape{.inject.} = conv2d_out_shape(
                      ishape,
                      kshape,
                      padding,
                      strides
                    )
    var output{.inject.} = newSeq[float32](oshape.n * oshape.c * oshape.h * oshape.w)
    echo "Output shape: " & $oshape

    conv_call_body
    # conv2d_direct(
    #   output,
    #   input, ishape,
    #   kernel, kshape,
    #   padding,
    #   strides
    # )

    echo output.toString(oshape)
    doAssert target == output

  block:
    let input{.inject.} = [
      [
        [
          [float32 2, 2, 0, 2, 1],
          [float32 0, 1, 1, 0, 2],
          [float32 1, 2, 1, 2, 1],
          [float32 2, 2, 0, 0, 2],
          [float32 2, 1, 1, 1, 2]
        ], [
          [float32 2, 0, 1, 1, 1],
          [float32 2, 2, 0, 0, 2],
          [float32 2, 2, 1, 0, 0],
          [float32 1, 1, 2, 2, 0],
          [float32 2, 1, 1, 1, 0]
        ], [
          [float32 0, 1, 2, 2, 0],
          [float32 1, 1, 1, 1, 0],
          [float32 2, 1, 2, 2, 0],
          [float32 0, 2, 2, 2, 1],
          [float32 0, 0, 2, 2, 1]
        ]
      ]].toTensor()
    let ishape{.inject.} = (1, 3, 5, 5)

    let kernel{.inject.} =
      [
        [
          [
            [float32 -1, -1, -1],
            [float32  1,  0,  1],
            [float32  0, -1,  0]
          ], [
            [float32  1,  0, -1],
            [float32  1, -1,  1],
            [float32  0,  1,  0]
          ], [
            [float32  0,  0,  1],
            [float32 -1, -1, -1],
            [float32 -1,  0,  0]
          ]
        ], [
          [
            [float32  0,  1,  0],
            [float32  1, -1, -1],
            [float32  1,  1, -1]
          ], [
            [float32 -1,  0,  1],
            [float32 -1, -1,  1],
            [float32  1,  1,  0]
          ], [
            [float32  0,  1,  1],
            [float32 -1,  1, -1],
            [float32 -1, -1,  0]
          ]
        ]
      ].toTensor()
    let kshape{.inject.} = (2, 3, 3, 3)

    let target =
      [
        [
          [float32  1, -3, -1],
          [float32 -4,  1, -6],
          [float32 -3, -2, -1]
        ],[
          [float32 -7,  1,  0],
          [float32  3, -3,  2],
          [float32  1,  3, -2]
        ]
      ].toTensor()

    let
      padding{.inject.} = (1, 1)
      strides{.inject.} = (2, 2)

    let oshape{.inject.} = conv2d_out_shape(
                      ishape,
                      kshape,
                      padding,
                      strides
                    )
    var output{.inject.} = newSeq[float32](oshape.n * oshape.c * oshape.h * oshape.w)
    echo "Output shape: " & $oshape

    conv_call_body
    # conv2d_direct(
    #   output,
    #   input, ishape,
    #   kernel, kshape,
    #   padding,
    #   strides
    # )

    echo output.toString(oshape)
    doAssert target == output
