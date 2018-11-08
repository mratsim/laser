# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

## This file implements a highly efficient transpose kernel.
## It is about 3xfaster than naive transposition

import
  ../compiler_optim_hints,
  ./private/ptr_arithmetic

when defined(openmp):
  {.passC: "-fopenmp".}
  {.passL: "-fopenmp".}

func transpose2D_copy*[T](
        dst, src: ptr (T or UncheckedArray[T]),
        NR, NC: Natural) =
  ## Efficient physical transposition of a contiguous 2D matrix
  ## Output:
  ##   - dst: a pointer to an allocated buffer of size NC * NR
  ##     dst does not need to be initialized and will be overwritten
  ## Input:
  ##   - src: a pointer to the source matrix of shape [NR, NC]
  ##   - NR, NC: the number of rows and columns respectively in the source matrix.

  # Implementation
  #
  # If using OpenMP
  # We construct blocks of 32x32 patches that we distribute
  # on all cores. I.e. The minimal grain size per core is 32*32 = 1024
  #
  # Benchmarking shows that it is faster to:
  #   - write into the dst tensor contiguously
  #     (and access src in a strided manner)
  #   - rather than read the src tensor contiguously
  #     and writing in a strided manner
  # i.e. scatters are cheaper than gathers

  withCompilerOptimHints()
  const blck = 32
  let pd{.restrict.} = dst
  let ps{.restrict.} = src

  {.emit: """
    #pragma omp parallel for simd collapse(2)
    for (int j = 0; j < `NC`; j+=`blck`)
      for (int i = 0; i < `NR`; i+=`blck`)
        for (int jj = j; jj<j+`blck` && jj<`NC`; jj++)
          for (int ii = i; ii<i+`blck` && ii<`NR`; ii++)
            `pd`[ii+jj*`NR`] = `ps`[jj+ii*`NC`];
  """.}

func transpose2D_batched*[T](
        dst, src: ptr (T or UncheckedArray[T]),
        N, NR, NC: Natural) =
  ## Efficient physical transposition of a batch of contiguous 2D matrices
  ## Output:
  ##   - dst: a pointer to an allocated buffer of size [N, NC * NR]
  ##     dst does not need to be initialized and will be overwritten
  ## Input:
  ##   - src: a pointer to the source matrices [N, [NR, NC]]
  ##     The source matrices must be contiguous
  ##   - N: The number of matrices in the batch
  ##   - NR, NC: the number of rows and columns respectively in the source matrix.

  withCompilerOptimHints()
  const blck = 32
  var pd{.restrict.} = dst
  var ps{.restrict.} = src

  {.emit: """
    #pragma omp parallel for simd collapse(3)
    for (int n = 0; n < `N`; n++)
      for (int j = 0; j < `NC`; j+=`blck`)
        for (int i = 0; i < `NR`; i+=`blck`)
          for (int jj = j; jj<j+`blck` && jj<`NC`; jj++)
            for (int ii = i; ii<i+`blck` && ii<`NR`; ii++)
              `pd`[ii+jj*(`NR` + n*`NC`)] = `ps`[jj+ii*(`NC` + n*`NR`)];
  """.}

func nchw2nhwc*[T](
        dst_hwnc, src_nchw: ptr (T or UncheckedArray[T]),
        N, C, H, W: Natural
      ){.inline.} =
  ## Convert from NCHW format to NHWC format.
  ## N stands for the batch_size or number of images
  ## C stands for the colors, feature channels or feature maps
  ## H stands for height
  ## W stands for width
  ##
  ## NCHW is the default format on PyTorch, CuDNN, mxnet, Chainer
  ## NHWC is the default format on Tensorflow
  transpose2D_batched(dst_hwnc, src_nchw, C, H*W)

func nhwc2nchw*[T](
        dst_nchw, src_nhwc: ptr (T or UncheckedArray[T]),
        N, C, H, W: Natural
      ){.inline.} =
  ## Convert from NHWC format to NCHW format.
  ## N stands for the batch_size or number of images
  ## C stands for the colors, feature channels or feature maps
  ## H stands for height
  ## W stands for width
  ##
  ## NCHW is the default format on PyTorch, CuDNN, mxnet, Chainer
  ## NHWC is the default format on Tensorflow
  transpose2D_batched(dst_nchw, src_nhwc, H*W, C)
