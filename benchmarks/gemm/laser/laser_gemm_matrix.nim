# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Minimal matrix view type
# for GEMM implementation

type
  MatrixView*[T] = object
    buffer*: ptr UncheckedArray[T]
    rowStride*, colStride*: int

func toMatrixView*[T](data: ptr T, rowStride, colStride: int): MatrixView[T] {.inline.} =
  result.buffer = cast[ptr UncheckedArray[T]](data)
  result.rowStride = rowStride
  result.colStride = colStride

template `[]`*[T](view: MatrixView[T], row, col: Natural): T =
  ## Access like a 2D matrix
  view.buffer[row * view.rowStride + col * view.colStride]

template `[]=`*[T](view: MatrixView[T], row, col: Natural, value: T) =
  ## Access like a 2D matrix
  view.buffer[row * view.rowStride + col * view.colStride] = value

func stride*[T](view: MatrixView[T], row, col: Natural): MatrixView[T]{.inline.}=
  ## Returns a new view offset by the row and column stride
  result.buffer = cast[ptr UncheckedArray[T]](
    addr view.buffer[row*view.rowStride + col*view.colStride]
  )
  result.rowStride = view.rowStride
  result.colStride = view.colStride
