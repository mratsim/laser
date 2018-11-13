# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Specialized microkernels for matrix multiplication

import
  ./laser_gemm_tiling, ./laser_gemm_matrix, ./laser_gemm_utils,
  ../../../laser/[cpuinfo, compiler_optim_hints],
  macros
