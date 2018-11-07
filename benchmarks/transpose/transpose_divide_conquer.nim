# Apache v2 License
# Mamy Ratsimbazafy

# ##########################################

# Parallel Recursive Divide-and-Conquer Cache Oblivious Transposition™

func `+`[T](p: ptr (T or UncheckedArray[T]), offset: int): type(p) {.inline.}=
  # Pointer arithmetic
  {.emit: "`result` = `p` + `offset`;".}

func transpose_conquer[T](dst, src: ptr UncheckedArray[T], NR, NC: Natural) =
  for j in 0 ..< NR:
    for i in 0 ..< NC:
      dst[i + j*NR] = src[j + i*NC]

func transpose_divide[T](dst, src: ptr UncheckedArray[T], nr, nc, NR, NC: Natural) =
  ## dst of shape NC, NR -> leading dim MR
  ## src of shape NR, NC -> leading dim NC
  if nr < 2: # TODO higher threshold
    transpose_conquer(dst, src, NR, NC)
  else:
    # We divide the longer dim
    if nr > nc:
      let nr2 = nr shr 1
      #     |D1|
      # D = |  |    D1 has nr2 rows and D2 has nr - nr2 rows
      #     |D2|
      #
      # S = |S1 S2| S1 has nr2 columns and S2 has nr - nr2 columns

      let d2row = dst + nr2
      let s2col = src + nr2*NC # (leading dim NC = stride)

      transpose_divide(dst,   src,   nr2,    nc, NR, NC) # Write transpose of S1 into D1
      transpose_divide(d2row, s2col, nr-nr2, nc, NR, NC) # Write transpose of S2 into D2
    else:
      let nc2 = nc shr 1
      # D = |D1 D2| D1 has nc2 columns and D2 has nc - nc2 columns
      #
      #     |S1|
      # S = |  |    S1 has nc2 rows and S2 has nc - nc2 rows
      #     |S2|

      let d2col = dst + nc2*NR # (leading dim NR = stride)
      let s2row = src + nc2

      transpose_divide(dst,   src,   nr, nc2,    NR, NC) # Write transpose of S1 into D1
      transpose_divide(d2col, s2row, nr, nc-nc2, NR, NC) # Write transpose of S2 into D2

# #####################################################################################

func transpose_recursive[T](dst, src: ptr UncheckedArray[T], nr, nc, NR, NC: Natural) =
  ## dst of shape NC, NR -> leading dim MR
  ## src of shape NR, NC -> leading dim NC
  if nr == 1 and nc == 1:
    dst[0] = src[0]
  else:
    # We divide the longer dim
    if nr > nc:
      let nr2 = nr shr 1
      #     |D1|
      # D = |  |    D1 has nr2 rows and D2 has nr - nr2 rows
      #     |D2|
      #
      # S = |S1 S2| S1 has nr2 columns and S2 has nr - nr2 columns

      let d2row = dst + nr2
      let s2col = src + nr2*NC # (leading dim NC = stride)

      transpose_recursive(dst,   src,   nr2,    nc, NR, NC) # Write transpose of S1 into D1
      transpose_recursive(d2row, s2col, nr-nr2, nc, NR, NC) # Write transpose of S2 into D2
    else:
      let nc2 = nc shr 1
      # D = |D1 D2| D1 has nc2 columns and D2 has nc - nc2 columns
      #
      #     |S1|
      # S = |  |    S1 has nc2 rows and S2 has nc - nc2 rows
      #     |S2|

      let d2col = dst + nc2*NR # (leading dim NR = stride)
      let s2row = src + nc2

      transpose_recursive(dst,   src,   nr, nc2,    NR, NC) # Write transpose of S1 into D1
      transpose_recursive(d2col, s2row, nr, nc-nc2, NR, NC) # Write transpose of S2 into D2

func transpose_cache_oblivious*[T](dst, src: ptr UncheckedArray[T], M, N: Natural) =
  transpose_recursive(dst, src, M, N, M, N)

when isMainModule:
  import ./transpose_common, sequtils, random

  const
    M = 5
    N = 5

  let src = newSeqWith(M*N, rand(20))
  var dst = newSeq[int](N*M)

  transpose_cache_oblivious(
    cast[ptr UncheckedArray[int]](dst[0].addr),
    cast[ptr UncheckedArray[int]](src[0].unsafeAddr),
    M, N
    )

  echo src.toString((M, N))
  echo dst.toString((N, M))
