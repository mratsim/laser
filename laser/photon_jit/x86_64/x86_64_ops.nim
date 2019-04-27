# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../photon_types,
  ./x86_64_op_generator, ./x86_64_base

# ################################################################
#
#   Mnemonic and Opcodes table for JIT Assembler proc generation
#
# ################################################################

# Notes:
#   - The imm64 version will generate a proc for uint64 and int64
#     and another one for pointers immediate
#   - The dst64, imm32 version will generate a proc for uint32 and int32
#     and a proc for int literals (known at compile-time)
#     that will call proc(reg, imm32) if the int is small enough.
#     ---> (dst64, imm64) should be defined before (dst64, imm32)

op_generator:
  op MOV: # MOV(dst, src) load/copy src into destination
    ## Copy 64-bit register content to another register
    [dst64, src64]: [rex(w=1), 0x89, modrm(Direct, reg = src64, rm = dst64)]
    ## Copy 32-bit register content to another register
    [dst32, src32]: [          0x89, modrm(Direct, reg = src32, rm = dst32)]
    ## Copy 16-bit register content to another register
    [dst16, src16]: [    0x66, 0x89, modrm(Direct, reg = src16, rm = dst16)]

    ## Copy  8-bit register content to another register
    [dst8,  src8]:  [          0x88, modrm(Direct, reg = src8, rm = dst8)]

    ## Copy 64-bit immediate value into register
    [dst64, imm64]: [rex(w=1), 0xB8 + dst64] & imm64
    ## Copy 32-bit immediate value into register
    [dst64, imm32]: [          0xB8 + dst64] & imm32
    ## Copy 16-bit immediate value into register
    [dst64, imm16]: [    0x66, 0xB8 + dst64] & imm16

    ## Copy 32-bit immediate value into register
    [dst32, imm32]: [          0xB8 + dst32] & imm32
    ## Copy 16-bit immediate value into register
    [dst32, imm16]: [    0x66, 0xB8 + dst32] & imm16

    ## Copy 16-bit immediate value into register
    [dst16, imm16]: [    0x66, 0xB8 + dst16] & imm16
    ## Copy  8-bit immediate value into register
    [dst8,  imm8]:  [          0xB0 + dst8, imm8]

  op LEA:
    ## Load effective address of the target label into a register
    [dst64, label]: [rex(w=1), 0x8D, modrm(Direct, reg = dst64, rm = rbp)]

  op CMP:
    ## Compare 32-bit immediate with 32-bit int at memory location stored in adr register
    [adr, imm64]: [ rex(w=1), 0x81, modrm(Indirect, opcode_ext = 7, rm = adr[0])] & imm64
    ## Compare 32-bit immediate with 32-bit int at memory location stored in adr register
    [adr, imm32]: [           0x81, modrm(Indirect, opcode_ext = 7, rm = adr[0])] & imm32
    ## Compare 16-bit immediate with 16-bit int at memory location stored in adr register
    [adr, imm16]: [     0x66, 0x81, modrm(Indirect, opcode_ext = 7, rm = adr[0])] & imm16
    ## Compare 8-bit immediate with byte at memory location stored in adr register
    [adr, imm8]:  [           0x80, modrm(Indirect, opcode_ext = 7, rm = adr[0]), imm8]

  op JZ:
    ## Jump to label if zero flag is set
    [label]: [0x0F, 0x84]
  op JNZ:
    ## Jump to label if zero flag is not set
    [label]: [0x0F, 0x85]

  op INC:
    ## Increment register by 1. Carry flag is never updated.
    [dst64]: [rex(w=1), 0xFF, modrm(Direct, opcode_ext = 0, rm = dst64)]
    [dst32]: [          0xFF, modrm(Direct, opcode_ext = 0, rm = dst32)]
    [dst16]: [    0x66, 0xFF, modrm(Direct, opcode_ext = 0, rm = dst16)]
    [dst8]:  [          0xFE, modrm(Direct, opcode_ext = 0, rm = dst8)]
    ## Increment data at the address by 1. Data type must be specified.
    [adr, type(64)]: [rex(w=1), 0xFF, modrm(Indirect, opcode_ext = 0, rm = adr[0])]
    [adr, type(32)]: [          0xFF, modrm(Indirect, opcode_ext = 0, rm = adr[0])]
    [adr, type(16)]: [    0x66, 0xFF, modrm(Indirect, opcode_ext = 0, rm = adr[0])]
    [adr, type(8)]:  [0xFE, modrm(Indirect, opcode_ext = 0, rm = adr[0])]

  op DEC:
    ## Increment register by 1. Carry flag is never updated.
    [dst64]: [rex(w=1), 0xFF, modrm(Direct, opcode_ext = 1, rm = dst64)]
    [dst32]: [          0xFF, modrm(Direct, opcode_ext = 1, rm = dst32)]
    [dst16]: [    0x66, 0xFF, modrm(Direct, opcode_ext = 1, rm = dst16)]
    [dst8]:  [          0xFE, modrm(Direct, opcode_ext = 1, rm = dst8)]
    ## Increment data at the address by 1. Data type must be specified.
    [adr, type(64)]: [rex(w=1), 0xFF, modrm(Indirect, opcode_ext = 1, rm = adr[0])]
    [adr, type(32)]: [          0xFF, modrm(Indirect, opcode_ext = 1, rm = adr[0])]
    [adr, type(16)]: [    0x66, 0xFF, modrm(Indirect, opcode_ext = 1, rm = adr[0])]
    [adr, type(8)]:  [0xFE, modrm(Indirect, opcode_ext = 1, rm = adr[0])]

