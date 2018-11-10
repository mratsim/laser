# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Minimal matrix view type
# for GEMM implementation

type
  MatrixView*[T] = object
    buffer*: ptr UncheckedArray[T]
    nrows*, ncols*: int
    rowStride*, colStride*: int

func toMatrixView*[T](data: ptr T, nrows, ncols, incRow, incCol: int): MatrixView[T] {.inline.} =
  result.buffer = cast[ptr UncheckedArray[T]](data)
  result.nrows = nrows
  result.ncols = ncols
  result.rowStride = incRow
  result.colStride = incCol

func incRow*[T](view: var MatrixView[T], offset = 1) {.inline.} =
  {.emit: "`view.buffer` += `view.rowStride` * `offset`;".}
  dec view.nrows, offset

func incCol*[T](view: var MatrixView[T], offset = 1) {.inline.} =
  {.emit: "`view.buffer` += `view.colStride` * `offset`;".}
  dec view.ncols, offset

func sliceRows*[T](view: MatrixView[T], size: int): MatrixView[T]{.inline.} =
  ## Returns a slice: view[0 ..< size, _]
  result = view
  assert nrows >= size
  result.nrows = size

func sliceCols*[T](view: MatrixView[T], size: int): MatrixView[T]{.inline.} =
  ## Returns a slice: view[_, 0 ..< size]
  result = view
  assert ncols >= size
  result.ncols = size

template at*[T](view: MatrixView[T], offset: int): T =
  ## Access a specific offset (like a linear memory range)
  view.buffer[idx]

template `[]`*[T](view: MatrixView[T], row, col: int): T =
  ## Access like a 2D matrix
  view.buffer[row * view.rowStride + col * view.colStride]
