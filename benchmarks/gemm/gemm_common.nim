# Apache v2 License
# Mamy Ratsimbazafy

import sequtils

type
  MatrixShape* = tuple[M, N: int]
  Matrix*[T] = seq[T]

func gemm_out_shape*(
      a: MatrixShape,
      b: MatrixShape,
    ): MatrixShape =

  doAssert a.N == b.M

  result.M = a.M
  result.N = b.N

func gemm_required_ops*(
      a: MatrixShape,
      b: MatrixShape
    ): int =
  doAssert a.N == b.M
  result = a.M * a.N * b.N * 2 # (1 add, 1 mul)

func gemm_required_data*(
      a: MatrixShape,
      b: MatrixShape
    ): int =
  doAssert a.N == b.M
  result = a.M * a.N + b.M * b.N

iterator flatIter*[T](s: openarray[T]): auto {.noSideEffect.}=
  for item in s:
    when item is array|seq:
      for subitem in flatIter(item):
        yield subitem
    else:
      yield item

func toMatrix*[T](oa: openarray[T]): auto =
  ## Convert to a flattened tensor/image
  toSeq(flatiter(oa))

proc toString*(mat: Matrix, shape: MatrixShape): string =
  for i in 0 ..< shape.M:
    for j in 0 ..< shape.N:
      let idx = j + shape.N * i
      result.add $mat[idx] & '\t'
    result.add '\n'

