# Apache v2 License
# Mamy Ratsimbazafy

import
  ./conv2d_common,
  ../../laser/openmp

proc conv2d_direct[T](
    oim: var seq[Image[T]],     # Output images
    iim: seq[Image[T]],         # Input images
    ishape: ImageShape,         # Shape of a single image
    kernel: seq[T],             # filter kernel
    kshape: KernelShape,        # kernel shape (should be const)
    padding: Padding,           # Padding (should be const)
    strides: Strides            # Strides (should be const)
  ) =
  ## oim must be zero-initialized
  # Reminder: convolution deep learning == cross-correlation signal processing

  assert oim.len == iim.len
  assert ishape.c == kshape.c_in
  let out_shape = conv2d_out_shape(ishape, kshape, padding, strides)
  when compileOption("boundChecks"):
    let expected_osize = out_shape.c * out_shape.h * out_shape.w
    let expected_isize = ishape.c    * ishape.h    * ishape.w
    for i in 0 ..< oim.len:
      assert oim[i].len == expected_osize
      assert iim[i].len == expected_isize

  let
    N = iim.len
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

  for n in 0||(N - 1):
    let odata = cast[ptr UncheckedArray[T]](oim[n][0].addr)
    let idata = cast[ptr UncheckedArray[T]](iim[n][0].unsafeaddr)
    for co in `||`(0,C_out-1, "simd"):
      for ci in 0 ..< C_in:
        for oh in 0 ..< outH:              # output height
          let ih = sH * oh                 # input height
          for ow in 0 ..< outW:            # output width
            let iw = sH * ow               # input width
            # The following should be loop hoisted by the compiler
            let oidx = ow + outW * (oh + outH * (co + C_out * n)) # odata[n][co][oh][ow]
            # Entering conv kernel region
            for hK in 0 ..< kH:
              let row = ih + hK - pH
              if 0 <= row and row < H:     # Check if within row boundaries
                for wK in 0 ..< kW:
                  let col = iw + wK - pW
                  if 0 <= col and col < W: # Check if within column boundaries
                    let iidx = col + W * (row + H * (ci + n * C_in)) #idata[n][ci][row][col]
                    let kidx = wK + kW * (hK + kH * (ci + C_in * co))
                    odata[oidx] += idata[iidx] * kdata[kidx]

when isMainModule:
  block:
    let img =  [[1, 2, 0, 0],
                [5, 3, 0, 4],
                [0, 0, 0, 7],
                [9, 3, 0, 0]].toflatSeq()
    let ishape: ImageShape = (1, 4, 4)
    let imgs = @[img]

    let kernel = [[1, 1, 1],
                  [1, 1, 0],
                  [1, 0, 0]].toflatSeq()
    const kshape: KernelShape = (1, 1, 3, 3)
    let target = [[1,  8,  5,  0],
                  [8, 11,  5,  4],
                  [8, 17, 10, 11],
                  [9, 12, 10,  7]].toflatSeq()

    let
      padding = (1, 1)
      strides = (1, 1)

    let out_shape = conv2d_out_shape(
                      ishape,
                      kshape,
                      padding,
                      strides
                    )
    var output = @[newSeq[int](out_shape.c * out_shape.h * out_shape.w)]
    echo "Output shape: " & $out_shape

    conv2d_direct(
      output,
      imgs, ishape,
      kernel, kshape,
      padding,
      strides
    )

    echo output[0].toString(out_shape)
    doAssert target == output[0]
