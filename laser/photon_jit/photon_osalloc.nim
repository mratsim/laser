# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# For now only working on POSIX
# TODO: for windows use VirtualAlloc, VirtualProtect, VirtualFree

# See - https://github.com/nim-lang/Nim/blob/devel/lib/system/osalloc.nim
const PageSize* = 4096

when not defined(posix):
  {.fatal: "Only POSIX systems are supported".}

type MemProt* {.size: cint.sizeof.}= enum
  ProtNone  = 0 # Page cannot be accessed
  ProtRead  = 1 # Page can be read
  ProtWrite = 2 # Page can be written
  ProtExec  = 4 # Page can be executed

when defined(osx) or defined(ios) or defined(bsd):
  # Note: MacOS and Iphone uses MAP_ANON instead of MAP_ANONYMOUS
  # They also define MAP_JIT= 0x0800
  type MemMap* {.size: cint.sizeof.}= enum
    MapPrivate = 0x02       # Changes are private
    MapAnonymous = 0x1000   # Don't use a file
elif defined(solaris):
  type MemMap* {.size: cint.sizeof.}= enum
    MapPrivate = 0x02
    MapAnonymous = 0x100
elif defined(haiku):
  type MemMap* {.size: cint.sizeof.}= enum
    MapPrivate = 0x02
    MapAnonymous = 0x08
else: # ASM-Generic
  # Note, Nim splits linux x86-64 and the rest
  # This is at least valid on Android ARM
  # unsure about MIPS and co
  type MemMap* {.size: cint.sizeof.}= enum
    MapPrivate = 0x02
    MapAnonymous = 0x20

type Flag*[E: enum] = distinct cint

func flag*[E: enum](e: varargs[E]): Flag[E] {.inline.} =
  ## Enum should only have power of 2 fields
  # Unfortunately iterating on low(E)..high(E)
  # will also iterate on the holes
  # static:
  #   for val in low(E)..high(E):
  #     assert (ord(val) and (ord(val) - 1)) == 0, "Enum values should all be power of 2, found " &
  #                                                 $val & " with value " & $ord(val) & "."
  var flags = 0
  for val in e:
    flags = flags or ord(val)
  result = Flag[E](flags)

when not defined(release):
  block:
    var DebugMapPrivate {.importc: "MAP_PRIVATE", header: "<sys/mman.h>".}: cint
    var DebugMapAnonymous {.importc: "MAP_ANONYMOUS", header: "<sys/mman.h>".}: cint

    assert ord(MapPrivate) == DebugMapPrivate, "Your CPU+OS platform is misconfigured"
    assert ord(MapAnonymous) == DebugMapAnonymous, "Your CPU+OS platform is misconfigured"

proc mmap*(
    adr: pointer, len: int,
    prot: Flag[MemProt], flags: Flag[MemMap],
    file_descriptor: cint, # -1 for anonymous memory
    offset: cint           # Offset in the file descriptor, PageSize aligned. Return Offset
  ): pointer {.header: "<sys/mman.h>", sideeffect.}
  ## The only portable address adr is "nil" to let OS decide
  ## where to alloc
  ## Returns -1 if error

proc mprotect*(adr: pointer, len: int, prot: Flag[MemProt]) {.header: "<sys/mman.h>", sideeffect.}
  ## len should be a multiple of PageSize
  ## replace previously existing protection with a set of new ones
  ## If an access is disallowed, program will segfault

proc munmap*(adr: pointer, len: int) {.header: "<sys/mman.h>", sideeffect.}
