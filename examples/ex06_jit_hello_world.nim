# ################################################################
#
#                Hello world using Photon JIT
#
# ################################################################

import sequtils, ../laser/photon_jit

proc main() =
  let HelloWorld = "Hello World!"

  let fn = gen_x86_64(assembler = a, clean_registers = true):
    # "write" syscall
    # rax = write syscall (0x01 on Linux, 0x02000004 on OSX)
    # rdi = stdout (stdout file descriptor = 0x01)
    # rsi = ptr to HelloWorld
    # rdx = HelloWorld.len
    # os.write(rdi, rsi, rdx) // os.write(file_descriptor, str_pointer, str_length)
    when defined(linux):
      a.mov rax, 0x01
    elif defined(osx):
      a.mov rax, 0x02000004
    else:
      {.error: "Unsupported OS".}

    a.mov rdi, 0x01
    a.mov rsi, HelloWorld[0].unsafeAddr
    a.mov rdx, uint32 HelloWorld.len # [1]
    a.syscall()
    a.ret()

  fn.call()
  echo '\n'

main()

# [1]: For runtime values we don't know if it's safe to use a short register/opcode
#      so you need to enforce the type of int and uint. (No need for uint32, uint64, int32, int64)
