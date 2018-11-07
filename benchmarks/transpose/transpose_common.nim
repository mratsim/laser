# Apache v2 License
# Mamy Ratsimbazafy

# ##########################################
import sequtils

type
  MatrixShape* = tuple[M, N: int]
  Matrix*[T] = seq[T]

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
