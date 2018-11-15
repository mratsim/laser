# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Specialized microkernels for matrix multiplication

import
  ../../cpuinfo, ../../compiler_optim_hints,
  ../../simd,
  macros,
  ./gemm_tiling, ./gemm_utils,
  ./gemm_ukernel_generic

# ############################################################
#
#               AVX, AVX + FMA (AVX2) Microkernels
#
# ############################################################

proc gebb_ukernel_epilogue_f32_avx[MR, NbVecs: static int, T](
      alpha: T, AB: array[MR, array[NbVecs, m256]],
      beta: T, vC: MatrixView[float32]
    ) {.inline.} =

  let C{.restrict.} = cast[ptr UncheckedArray[m256]](vC.buffer[0].addr)

  if beta == 0.T:
    for i in 0 ..< MR:
      for j in 0 ..< NbVecs:
        C[i*vC.rowStride + j] = mm256_setzero_ps()
  elif beta != 1.T:
    let beta_vec = mm256_set1_ps(beta)
    for i in 0 ..< MR:
      for j in 0 ..< NbVecs:
        C[i*vC.rowStride + j] = mm256_mul_ps(C[i*vC.rowStride + j], beta_vec)

  if alpha == 1.T:
    for i in 0 ..< MR:
      for j in 0 ..< NbVecs:
        C[i*vC.rowStride + j] = mm256_add_ps(C[i*vC.rowStride + j], AB[i][j])
  else:
    let alpha_vec = mm256_set1_ps(alpha)
    # TODO - non FMA
    for i in 0 ..< MR:
      for j in 0 ..< NbVecs:
        C[i*vC.rowStride + j] = mm256_fmadd_ps(C[i*vC.rowStride + j], AB[i][j], alpha_vec)

macro ukernel_impl(simd: static CPUFeatureX86, A, B: untyped, NbVecs, NBElems, MR, NR: static int, kc: int): untyped =

  result = newStmtList()
  let k = genSym(nskForVar)

  ## Registers
  var rA: seq[NimNode]           # array[MR div 2, m256] - TODO, support NbVecs != 2
  var rB: seq[NimNode]           # array[NR div vecsize, m256]
  for i in 0 ..< NbVecs:
    rA.add genSym(nskVar, "A" & $i)
    rB.add genSym(nskVar, "B" & $i)
  var rAB = nnkBracket.newTree() # array[MR, array[NbVecs, m256]]
  for i in 0 ..< MR:
    var rABi = nnkBracket.newTree()
    for j in 0 ..< NbVecs:
      rABi.add genSym(nskVar, "AB" & $i & "__" & $j)
    rAB.add rABi

  ## Declare
  var declBody = newStmtList()
  for a in rA:
    declBody.add quote do:
      var `a`{.noinit.}: m256
  for b in rB:
    declBody.add quote do:
      var `b`{.noinit.}: m256
  for i in 0 ..< MR:
    for j in 0 ..< NbVecs:
      let ab = rAB[i][j]
      declBody.add quote do:
        var `ab` = mm256_setzero_ps()

  ## Prefetch
  var prefetchBody = newStmtList()
  for jj in 0 ..< NbVecs:
    prefetchBody.add quote do:
      prefetch(`B`[(`k`+1)*NR+`jj`*`NBElems`].addr, Read, LowTemporalLocality)

  ## Load
  var loadBody = newStmtList()
  for jj in 0 ..< NbVecs:
    let b = rB[jj]
    loadBody.add quote do:
      `b` = mm256_load_ps(`B`[`k`*NR+`jj`*`NBElems`].addr)

  ## Interleaved broadcast and FMA
  var bcast_fma = newStmtList()
  block:
    let a0 = rA[0]
    bcast_fma.add quote do:
      `a0` = mm256_set1_ps(`A`[`k`*MR])

  for i in countup(0, MR-1, 2): #  to MR inclusive
    for ii in 0 ..< NbVecs:
      if i != MR:
        # broadcast next iteration
        let a_next = rA[(ii+1) mod NbVecs]
        bcast_fma.add quote do:
          `a_next` = mm256_set1_ps(`A`[`k`*MR+(`i`+1)+`ii`*`NBElems`])

      # Do FMA on the current one
      let a = rA[ii]
      for jj in 0 ..< NbVecs:
        let b = rB[jj]
        let AB = rAB[min(MR-1, i + ii)][jj]
        if simd == x86_AVX:
          bcast_fma.add quote do:
            `AB` = mm256_add_ps(mm256_mul_ps(`a`, `b`), `AB`)
        else:
          bcast_fma.add quote do:
            `AB` = mm256_fmadd_ps(`a`, `b`, `AB`)

  ## Assemble:
  result = quote do:
    `declBody`
    for `k` in 0 ..< `kc`:
      `loadBody`
      `prefetchBody`
      `bcast_fma`
    ## Write registers to a MR/NR array
    `rAB`

proc gebb_ukernel_f32_avx*[ukernel: static MicroKernel](
      kc: int,
      alpha: float32, packedA, packedB: ptr UncheckedArray[float32],
      beta: float32, vC: MatrixView[float32]
    ) =
  const
    MR = ukernel.extract_mr()
    NR = ukernel.extract_nr()
    vec_size = ukernel.extract_vecsize
    simd = ukernel.extract_cpu_simd
    NbElems = 8
    NbVecs = NR div NbElems

  static:
    assert vecsize == 32
    assert simd in {x86_AVX, x86_AVX2, x86_AVX512}
    assert NR div 8 == 0 # Unrolling checks
    assert MR div 2 == 0

  var  A {.restrict.} = assume_aligned packedA # [kc, mc] by chunks of mr
  var  B {.restrict.} = assume_aligned packedB # [kc, nc] by chunks of nr

  let AB{.align_variable.} = simd.ukernel_impl(A, B, NbVecs, NBElems, MR, NR, kc)

  const is_c_unit_stride = ukernel.extract_c_unit_stride
  when is_c_unit_stride:
    gebb_ukernel_epilogue_f32_avx(
      alpha, AB,
      beta, vC
    )
  else:
    gebb_ukernel_epilogue(
      alpha, to_ptr(AB, MR, NR, float32),
      beta, vC, is_c_unit_stride)

# #####################################################
  # Reference loop -  AB: array[MR, array[NR, float32]]
  # for k in 0 ..< kc:
  #   for i in 0 ..< MR:
  #     for j in 0 ..< NR:
  #       AB[i][j] += A[k*MR+i] * B[k*NR+j]
