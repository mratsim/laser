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

func toMatrixView*[T](data: ptr T, nrows, ncols: Positive, rowStride, colStride: int): MatrixView[T] {.inline.} =
  result.buffer = cast[ptr UncheckedArray[T]](data)
  result.nrows = nrows
  result.ncols = ncols
  result.rowStride = rowStride
  result.colStride = colStride

func incRow*[T](view: var MatrixView[T], offset: Natural = 1) {.inline.} =
  # Need to dereference the hidden pointer for var
  {.emit: "`(*view).buffer` += `(*view).rowStride` * `offset`;".}
  dec view.nrows, offset

func incCol*[T](view: var MatrixView[T], offset: Natural = 1) {.inline.} =
    # Need to dereference the hidden pointer for var
  {.emit: "`(*view).buffer` += `(*view).colStride` * `offset`;".}
  dec view.ncols, offset

func sliceRows*[T](view: MatrixView[T], size: Natural): MatrixView[T]{.inline.} =
  ## Returns a **exclusive** slice: view[0 ..< size, _]
  result = view
  assert size <= result.nrows
  result.nrows = size

func sliceCols*[T](view: MatrixView[T], size: Natural): MatrixView[T]{.inline.} =
  ## Returns an **exclusive** slice: view[_, 0 ..< size]
  result = view
  assert size <= result.ncols
  result.ncols = size

template at*[T](view: MatrixView[T], offset: Natural): T =
  ## Access a specific offset (like a linear memory range)
  assert offset < (view.nrows * view.rowStride) + (view.ncols * view.colStride)
  view.buffer[idx]

template `[]`*[T](view: MatrixView[T], row, col: Natural): T =
  ## Access like a 2D matrix
  # assert row < view.nrows
  # assert col < view.ncols
  view.buffer[row * view.rowStride + col * view.colStride]

template `[]=`*[T](view: MatrixView[T], row, col: Natural, value: T) =
  ## Access like a 2D matrix
  # assert row < view.nrows
  # assert col < view.ncols
  view.buffer[row * view.rowStride + col * view.colStride] = value
