# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import photon_jit/photon_types
export JitFunction, Assembler, call, hash, Label, initLabel, label

import photon_jit/x86_64/x86_64_base
export
  X86_64, RegX86_64, RegX86_32, RegX86_16, # TODO: RegX86_8, RegX86_8_REX
  gen_x86_64

import photon_jit/x86_64/[x86_64_ops, x86_64_ops_call_stack]
export x86_64_ops, x86_64_ops_call_stack
