import os, strutils
const cSourcesPath = currentSourcePath.rsplit(DirSep, 1)[0] & '/'

{.passC:"-I" & cSourcesPath & "mkl-dnn/include".}
{.passC:"-I" & cSourcesPath & "mkl-dnn/src/common".}
{.passC:"-I" & cSourcesPath & "mkl-dnn/src/cpu".}
{.passC:"-I" & cSourcesPath & "mkl-dnn/src/cpu/gemm/f32".}
# {.passC:"-std=c++11".}

{.compile: cSourcesPath & "mkl-dnn/src/common/utils.cpp".}
{.compile: cSourcesPath & "mkl-dnn/src/cpu/gemm/f32/ref_gemm_f32.cpp".}
{.compile: cSourcesPath & "mkl-dnn/src/cpu/gemm/f32/gemm_utils_f32.cpp".}


type MkldnnStatus {.importc: "mkldnn_status_t".} = enum
    # The operation was successful
    MkldnnSuccess = 0,
    # The operation failed due to an out-of-memory condition
    MkldnnOutOfMemory = 1,
    # The operation failed and should be retried
    MkldnnTryAgain = 2,
    # The operation failed because of incorrect function arguments
    MkldnnInvalidArguments = 3,
    # The operation failed because a primitive was not ready for execution
    MkldnnNotReady = 4,
    # The operation failed because requested functionality is not implemented
    MkldnnUnimplemented = 5,
    # Primitive iterator passed over last primitive descriptor
    MkldnnIteratorEnds = 6,
    # Primitive or engine failed on execution
    MkldnnRuntimeError = 7,
    # Queried element is not required for given primitive
    MkldnnNotRequired = 8

proc mkldnn_ref_gemm*[T](
  transa: ptr char, transb: ptr char,
  M, N, K: ptr int32,
  alpha, A: ptr T, lda: ptr int32,
         B: ptr T, ldb: ptr int32,
  beta,  C: ptr T, ldc: ptr int32,
      bias: ptr T 
): MkldnnStatus {.
  importcpp:"mkldnn::impl::cpu::ref_gemm<'*6>(@)",
  header: cSourcesPath & "mkl-dnn/src/cpu/gemm/f32/ref_gemm_f32.hpp"
.}