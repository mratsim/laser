# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ./omp_tuning, ./omp_mangling

template omp_parallel_for*[T](
      index: untyped,
      length: Natural,
      omp_threshold: static Natural,
      use_simd: static bool = true,
      body: untyped
      ) =
  when omp_threshold == 0:
    when use_simd:
      for index in `||`(0, length - 1, "simd"):
        body
    else:
      for index in 0||(length-1):
        body
  else:
    const ompsize_Csymbol = "ompsize_" & omp_suffix(genNew = true)
    let ompsize {.exportc: "ompsize_" & omp_suffix(genNew = false).} = data.len
    when use_simd:
      const omp_annotation = "simd if(" & $ompthreshold & " < " & ompsize_Csymbol & ")"
    else:
      const omp_annotation = "if(" & $ompthreshold & " < " & ompsize_Csymbol & ")"
    for index in `||`(0, length - 1, omp_annotation):
      body
