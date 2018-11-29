# ################################################################
#
#          Advanced Photon JIT use with a Brainfuck VM
#
# ################################################################

# Compile and call the executable with one of the *.bf files
# in the ex07_jit_brainfuck_src folder.
# Or compile directly .nim file that embeds the BrainFuck Hello World.
#
# Note that the "factor.bf" expects you to input an integer in the
# command-line. The program will then return the factors.

import streams, ../laser/photon_jit

type
  BrainfuckVM = object
    mem: seq[uint8]      # Memory
    program: JitFunction # Dynamic recompilation of the brainfuck program into x86 Assembly

const MemSize = 30000

proc initBrainFuckVM(stream: Stream): BrainfuckVM =

  result.mem = newSeq[uint8](MemSize)
  result.program = gen_x86_64(assembler = a, clean_registers = true):

    # Load a pointer to allocated mem
    a.mov(rbx, result.mem[0].addr)

    template os_write(a: Assembler[X86_64]) =
      # os.write(rdi, rsi, rdx)
      # os.write(file_descriptor, str_pointer, str_length)
      when defined(linux):
        a.mov rax, 0x01
      elif defined(osx):
        a.mov rax, 0x02000004
      else:
        {.error: "Unsupported OS".}
      a.mov rdi, 0x01   # stdout file descriptor = 0x01
      a.mov rsi, rbx    # rbx holds the current memory pointer
      a.mov rdx, 1      # write one char at a time (note that this is unset after the syscall, hence we need to load it again and again ...)
      a.syscall

    template os_read(a: Assembler[X86_64]) =
      # os.read(rdi, rsi, rdx)
      # os.read(file_descriptor, str_pointer, str_length)
      when defined(linux):
        a.mov rax, 0x00
      elif defined(osx):
        a.mov rax, 0x02000003
      else:
        {.error: "Unsupported OS".}
      a.mov rdi, 0x00   # stdin file descriptor = 0x00
      a.mov rsi, rbx    # rbx holds the current memory pointer
      a.mov rdx, 1      # read one char at a time (note that this is unset after the syscall, hence we need to load it again and again ...)
      a.syscall

    ## We keep track of the jump targets with a stack
    var stack: seq[tuple[loop_start, loop_end: Label]]

    while not stream.atEnd():
      case stream.readChar()
      of '>': a.inc rbx          # Pointer increment
      of '<': a.dec rbx          # Pointer decrement
      of '+': a.inc [rbx], uint8 # Memory increment
      of '-': a.dec [rbx], uint8 # Memory decrement
      of '.': a.os_write()       # Print
      of ',': a.os_read()        # Read from stdin
      of '[':                    # If mem == 0, Skip block to corresponding ']'
        let
          loop_start = initLabel()
          loop_end   = initLabel()
        a.cmp [rbx], uint8 0
        a.jz loop_end
        a.label loop_start
        stack.add (loop_start, loop_end)
      of ']':
        let (loop_start, loop_end) = stack.pop()
        a.cmp [rbx], uint8 0
        a.jnz loop_start
        a.label loop_end
      else:
        discard
    a.ret

proc run(vm: BrainfuckVM) {.inline.} =
  vm.program.call()
  echo '\n'

proc execBFfile*(file: string) =
  let s = openFileStream(file)
  defer: s.close()
  var vm = s.initBrainfuckVM()
  vm.run()

proc execBFstring*(prog: string) =
  let s = newStringStream(prog)
  defer: s.close()
  var vm = s.initBrainfuckVM()
  vm.run()

when isMainModule:
  import os
  let filePath = paramStr(1).string
  execBFfile(filePath)
