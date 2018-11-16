# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../compiler_optim_hints,
  ../../simd,
  macros, typetraits,
  ./gemm_tiling, ./gemm_utils,
  ./gemm_ukernel_generic

# ############################################################
#
#             SIMD implementation generator
#
# ############################################################

# Macro should be invoked in different files so that specific
# flags like "-mavx -mfma" are isolated.
# Add the corresponding compilation flags to "nim.cfg"

macro ukernel_impl*(simd: static CPUFeatureX86, A, B: untyped, NbVecs, NBElems, MR, NR: static int, kc: int): untyped =

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
      prefetch(`B`[(`k`+1)*`NR`+`jj`*`NBElems`].addr, Read, LowTemporalLocality)

  ## Load
  var loadBody = newStmtList()
  for jj in 0 ..< NbVecs:
    let b = rB[jj]
    loadBody.add quote do:
      `b` = mm256_load_ps(`B`[`k`*`NR`+`jj`*`NBElems`].addr)

  ## Interleaved broadcast and FMA
  var bcast_fma = newStmtList()
  block:
    let a0 = rA[0]
    bcast_fma.add quote do:
      `a0` = mm256_set1_ps(`A`[`k`*`MR`])

  for i in countup(0, MR-1, 2): #  to MR inclusive
    for ii in 0 ..< NbVecs:
      if i != MR:
        # broadcast next iteration
        let a_next = rA[(ii+1) mod NbVecs]
        bcast_fma.add quote do:
          `a_next` = mm256_set1_ps(`A`[`k`*`MR`+(`i`+1)+`ii`*`NBElems`])

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

macro ukernel_generator*(
      typedescs: varargs[typedesc],
      simd: static CPUFeatureX86
    ): untyped =

  let T = newIdentNode("float32")
  let V = newIdentNode("m256")
  let epilogue_name = newIdentNode("gebb_ukernel_epilogue_" & $T & "_" & "x86_AVX2")
  result = newStmtList()

  # 1. Generate the epilogue function
  result.add quote do:
    proc `epilogue_name`[MR, NbVecs: static int](
          alpha: `T`, AB: array[MR, array[NbVecs, `V`]],
          beta: `T`, vC: MatrixView[`T`]
        ) =

      const VecSize = 8
      template C(i,j: int): untyped {.dirty.} = vC.buffer[i*vC.rowStride + j*VecSize]

      if beta == 0.`T`:
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            mm256_storeu_ps(C(i,j).addr, mm256_setzero_ps())
      elif beta != 1.`T`:
        let beta_vec = mm256_set1_ps(beta)
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            mm256_storeu_ps(C(i,j).addr, mm256_mul_ps(beta_vec, C(i,j).addr.mm256_loadu_ps))

      if alpha == 1.`T`:
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            mm256_storeu_ps(C(i,j).addr, mm256_add_ps(AB[i][j], C(i,j).addr.mm256_loadu_ps))
      else:
        let alpha_vec = mm256_set1_ps(alpha)
        # TODO - non FMA
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            mm256_storeu_ps(C(i,j).addr, mm256_fmadd_ps(alpha_vec, AB[i][j], C(i,j).addr.mm256_loadu_ps))

  # 2. Generate the microkernel
  let ukernel_name = newIdentNode("gebb_ukernel_" & $T & "_" & "x86_AVX2")
  result.add quote do:
    proc `ukernel_name`*[ukernel: static MicroKernel](
          kc: int,
          alpha: `T`, packedA, packedB: ptr UncheckedArray[`T`],
          beta: `T`, vC: MatrixView[`T`]
        ) =
      const
        MR = ukernel.extract_mr()
        NR = ukernel.extract_nr()
        simd = ukernel.extract_cpu_simd
        NbElems = 8
        NbVecs = NR div NbElems

      var  A {.restrict.} = assume_aligned packedA # [kc, mc] by chunks of mr
      var  B {.restrict.} = assume_aligned packedB # [kc, nc] by chunks of nr

      let AB{.align_variable.} = simd.ukernel_impl(A, B, NbVecs, NBElems, MR, NR, kc)

      const is_c_unit_stride = ukernel.extract_c_unit_stride
      when is_c_unit_stride:
        `epilogue_name`(
          alpha, AB,
          beta, vC)
      else:
        gebb_ukernel_epilogue(
          alpha, to_ptr(AB, MR, NR, `T`),
          beta, vC, is_c_unit_stride)
