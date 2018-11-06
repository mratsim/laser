# Apache v2 License
# Mamy Ratsimbazafy

import sequtils

type
  ImageShape* = tuple[c, h, w: int]              # Channel/Color, Height, Width
  KernelShape* = tuple[c_out, c_in, kH, kW: int] # Channel out, Channel in, kernel height, kernel out
  Padding* = tuple[h, w: int]
  Strides* = tuple[h, w: int]
  ## We don't support dilation for benchmarking

  Image*[T] = seq[T]

func conv2d_out_shape*(
      input: ImageShape,
      kernel: KernelShape,
      padding: Padding,
      strides: Strides
    ): ImageShape =

  let
    iH = input.h
    iW = input.w
    kH = kernel.kH
    kW = kernel.kW

  let # Should be const but padding.h causes problem and padding[0] indexing
      # doesn't work in generic proc
    pH = padding[0] # padding.h causes problem
    pW = padding[1]
    sH = strides[0]
    sW = strides[1]

  ## Dilation, unsupported
  const dH = 1
  const dW = 1

  result.c = kernel.c_out
  result.h = 1 + (iH + 2*pH - (((kH-1) * dH) + 1) div sH)
  result.w = 1 + (iW + 2*pW - (((kW-1) * dW) + 1) div sW)

iterator flatIter*[T](s: openarray[T]): auto {.noSideEffect.}=
  for item in s:
    when item is array|seq:
      for subitem in flatIter(item):
        yield subitem
    else:
      yield item

func toflatSeq*[T](oa: openarray[T]): auto =
  toSeq(flatiter(oa))

proc toString*(im: Image, shape: ImageShape): string =
  for c in 0 ..< shape.c:
    for h in 0 ..< shape.h:
      for w in 0 ..< shape.w:
        let idx = w + shape.w * (h + shape.h * c)
        result.add $im[idx] & '\t'
      result.add '\n'
    result.add '\n'

when isMainModule:
  block:
    let img =  [[1, 2, 0, 0],
                [5, 3, 0, 4],
                [0, 0, 0, 7],
                [9, 3, 0, 0]].toflatSeq()

    echo img.toString((1, 3, 4))
