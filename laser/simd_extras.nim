# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import simd

# ############################################################
#
#         Extras that really should be in official SIMDs
#
# ############################################################

when defined(i386) or defined(amd64):
  template laser_mm512_cmpgt_epi32*(a, b: m512i): m512i =
    ## Compare a greater than b
    mm512_movm_epi32(mm512_cmpgt_epi32_mask(a, b))
  
  template laser_mm512_movemask_epi8*(a: m512i): uint64 =
    uint64 a.mm512_movepi8_mask # .cvtmask64_u64 -- cvtmask is not implemented in LLVM and automatically generated

