# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ./x86_64_op_generator

op_generator:
  op MOV: # MOV(dst, src) load/copy src into destination
    ## Copy 64-bit register content to another register
    [dst64, src64]: [rex(w=1), 0x89, modrm(Direct, reg = src, rm = dst)]
    ## Copy 32-bit register content to another register
    [dst32, src32]: [          0x89, modrm(Direct, reg = src, rm = dst)]
    ## Copy 16-bit register content to another register
    [dst16, src16]: [    0x66, 0x89, modrm(Direct, reg = src, rm = dst)]

    ## Copy  8-bit register content to another register
    [dst8,  src8]:  [          0x88, modrm(Direct, reg = src, rm = dst)]

    ## Copy 64-bit immediate value into register
    [dst64, imm64]: [rex(w=1), 0xB8 + dst, imm64]
    ## Copy 32-bit immediate value into register
    [dst64, imm32]: [          0xB8 + dst, imm32]
    ## Copy 16-bit immediate value into register
    [dst64, imm16]: [    0x66, 0xB8 + dst, imm16]

    ## Copy 32-bit immediate value into register
    [dst32, imm32]: [          0xB8 + dst, imm32]
    ## Copy 16-bit immediate value into register
    [dst32, imm16]: [    0x66, 0xB8 + dst, imm16]

    ## Copy 16-bit immediate value into register
    [dst16, imm16]: [    0x66, 0xB8 + dst, imm16]
    ## Copy  8-bit immediate value into register
    [dst8,  imm8]:  [          0xB0 + dst, imm8]

  op LEA:
    ## Load effective address of the target label into a register
    [dst64, label]: [rex(w=1), 0x8D, modrm(Direct, reg = src, rm = dst)]

  op CMP:
    ## Compare 8-bit immediate with byte at memory location stored in adr register
    [adr, imm8]: [0x80, modrm(Indirect, opcode_ext = 7, rm = adr[0]), imm8]

  op JZ:
    ## Jump to label if zero flag is set
    [label]: [0x0F, 0x84]
  op JNZ:
    ## Jump to label if zero flag is not set
    [label]: [0x0F, 0x85]

  op INC:
    ## Increment register by 1. Carry flag is never updated.
    [reg]: [0xFF, modrm(Direct, opcode_ext = 0, rm = reg)]
    ## Increment data at the address by 1. Data type must be specified.
    [adr, type(16, 32, 64)]:
      [rex(w=1), 0xFF, modrm(Indirect, opcode_ext = 0, rm = adr[0])]
    ## Increment data at the address by 1. Data type must be specified.
    [adr, type(8)]: [0xFE, modrm(Indirect, opcode_ext = 0, rm = adr[0])]

  op DEC:
    ## Increment register by 1. Carry flag is never updated.
    [reg]: [0xFF, modrm(Direct, opcode_ext = 1, rm = reg)]
    ## Increment data at the address by 1. Data type must be specified.
    [adr, type(16, 32, 64)]:
      [rex(w=1), 0xFF, modrm(Indirect, opcode_ext = 1, rm = adr[0])]
    ## Increment data at the address by 1. Data type must be specified.
    [adr, type(8)]: [0xFE, modrm(Indirect, opcode_ext = 1, rm = adr[0])]

