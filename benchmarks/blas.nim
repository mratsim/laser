when defined(osx):
  const blas = "libopenblas.dylib"
  {.passC: "-I'/usr/local/opt/openblas/include' -L'/usr/local/opt/openblas/lib'".}
elif defined(linux):
  const blas = "libcblas.so"
else:
  {.fatal: "OpenBLAS not configured for this platform".}
  # When adding new platform, you also need to update nim.cfg
# Don't forget to have Openblas in your path
# for example "export LD_LIBRARY_PATH=/usr/local/opt/openblas/lib"

type
  TransposeType* {.size: sizeof(cint).} = enum
    noTranspose = 111, transpose = 112, conjTranspose = 113
  OrderType* {.size: sizeof(cint).} = enum
    rowMajor = 101, colMajor = 102

proc gemm*(ORDER: OrderType, TRANSA, TRANSB: TransposeType, M, N, K: int, ALPHA: float32,
  A: ptr float32, LDA: int, B: ptr float32, LDB: int, BETA: float32, C: ptr float32, LDC: int)
  {. dynlib: blas, importc: "cblas_sgemm" .}
proc gemm*(ORDER: OrderType, TRANSA, TRANSB: TransposeType, M, N, K: int, ALPHA: float64,
  A: ptr float64, LDA: int, B: ptr float64, LDB: int, BETA: float64, C: ptr float64, LDC: int)
  {. dynlib: blas, importc: "cblas_dgemm" .}

# proc omatcopy*(ORDER: OrderType, TRANS: TransposeType,
#               rows: int, cols: int, scale_factor: float32,
#               a: ptr float32, lda: int,
#               b: ptr float32, ldb: int){. dynlib: blas, importc: "cblas_somatcopy" .}
