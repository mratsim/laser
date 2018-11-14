# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#                    Pointer arithmetics
#
# ############################################################

# Warning for pointer arithmetics be careful of not passing a `var ptr`
# to a function as `var` are passed by hidden pointers in Nim and the wrong
# pointer will be modified. Templates are fine.

func `+`*(p: ptr, offset: int): type(p) {.inline.}=
  ## Pointer increment
  {.emit: "`result` = `p` + `offset`;".}

# ############################################################
#
#                    Matrix View
#
# ############################################################

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

# ############################################################
#
#                    SIMD intrinsics
#
# ############################################################

# Vector helpers - crashes the compiler :/

# template load*[T](vecsize: static int, packedB: ptr UncheckedArray[T]): m256 =
#   when vecsize == 32:
#     when T is float32: mm256_load_ps(packedB[0].addr)
#     # else: mm256_load_pd(packedB[0].addr)
#   elif vecsize == 16:
#     when T is float32: mm_load_ps(packedB[0].addr)
#     # else: mm_load_pd(packedB[0].addr)

# template setzero*(vecsize: static int, T: typedesc): m256 =
#   when vecsize == 32:
#     when T is float32: mm256_setzero_ps()
#     # else: mm256_setzero_pd()
#   elif vecsize == 16:
#     when T is float32: mm_setzero_ps()
#     # else: mm_setzero_pd()

# template set1*[T](vecsize: static int, value: T): m256 =
#   when vecsize == 32:
#     when T is float32: mm256_set1_ps(value)
#     # else: mm256_setzero_pd(value)
#   elif vecsize == 16:
#     when T is float32: mm_set1_ps(value)
#     # else: mm_set_pd(value)

# func fma*[T](simd: CPUFeatureX86, a, b, c: T): T {.inline.}=
#   when T is m256:
#     when simd in {x86_AVX2 or x86_AVX512}: mm256_fmadd_ps(a, b, c)
#     else: mm256_add_ps(mm256_mul_ps(a, b), c)
#   elif T is m256d:
#     when simd in {x86_AVX2 or x86_AVX512}: mm256_fmadd_pd(a, b, c)
#     # else: mm256_add_ps(mm256_mul_pd(a, b), c)
#   when T is m128:
#     when simd in {x86_AVX2 or x86_AVX512}: mm_fmadd_ps(a, b, c)
#     else: mm_add_ps(mm_mul_ps(a, b), c)
#   elif T is m128d:
#     when simd in {x86_AVX2 or x86_AVX512}: mm_fmadd_pd(a, b, c)
#     # else: mm_add_pd(mm_mul_pd(a, b), c)
