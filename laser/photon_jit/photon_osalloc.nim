# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# See - https://github.com/nim-lang/Nim/blob/devel/lib/system/osalloc.nim
const PageSize* = 4096

when defined(windows):
  type MemProt* {.size: cint.sizeof.} = enum
    ProtReadWrite = 0x04 # Page can be read or written to
    ProtReadExec  = 0x02 # Page can be read or executed from
else:
  type MemProt* {.size: cint.sizeof.} = enum
    # is *not* a flag on Win32!
    ProtRead  = 1 # Page can be read
    ProtWrite = 2 # Page can be written
    ProtExec  = 4 # Page can be executed

when defined(windows):
  type
    MemAlloc* {.size: cint.sizeof.} = enum
      # is a flag on Win32!
      MemCommit = 0x1000 # Allocate immediately
    MemFree* {.size: cint.sizeof.} = enum
      # is *not* a flag on Win32!
      MemRelease = 0x1000 # Free immediately
elif defined(osx) or defined(ios) or defined(bsd):
  # Note: MacOS and Iphone uses MAP_ANON instead of MAP_ANONYMOUS
  # They also define MAP_JIT= 0x0800
  type MemMap* {.size: cint.sizeof.} = enum
    MapPrivate = 0x02       # Changes are private
    MapAnonymous = 0x1000   # Don't use a file
elif defined(solaris):
  type MemMap* {.size: cint.sizeof.} = enum
    MapPrivate = 0x02
    MapAnonymous = 0x100
elif defined(haiku):
  type MemMap* {.size: cint.sizeof.} = enum
    MapPrivate = 0x02
    MapAnonymous = 0x08
else: # ASM-Generic
  # Note, Nim splits linux x86-64 and the rest
  # This is at least valid on Android ARM
  # unsure about MIPS and co
  type MemMap* {.size: cint.sizeof.} = enum
    MapPrivate = 0x02
    MapAnonymous = 0x20

type Flag*[E: enum] = distinct cint

func flag*[E: enum](e: varargs[E]): Flag[E] {.inline.} =
  ## Enum should only have power of 2 fields
  # static:
  #   for val in E:
  #     assert (ord(val) and (ord(val) - 1)) == 0, "Enum values should all be power of 2, found " &
  #                                                 $val & " with value " & $ord(val) & "."
  var flags = 0
  for val in e:
    flags = flags or ord(val)
  result = Flag[E](flags)

when not defined(release):
  when defined(windows):
      var DebugMemCommit {.importc: "MEM_COMMIT", header: "<windows.h>".}: cint
      var DebugMemRelease {.importc: "MEM_RELEASE", header: "<windows.h>".}: cint

      assert ord(MemCommit) == DebugMemCommit, "Your CPU+OS platform is misconfigured"
      assert ord(MemRelease) == DebugMemRelease, "Your CPU+OS platform is misconfigured"
  else:
    block:
      var DebugMapPrivate {.importc: "MAP_PRIVATE", header: "<sys/mman.h>".}: cint
      var DebugMapAnonymous {.importc: "MAP_ANONYMOUS", header: "<sys/mman.h>".}: cint

      assert ord(MapPrivate) == DebugMapPrivate, "Your CPU+OS platform is misconfigured"
      assert ord(MapAnonymous) == DebugMapAnonymous, "Your CPU+OS platform is misconfigured"

when defined(windows):
  proc VirtualAlloc*(
      lpAddress: pointer,
      dwSize: int;
      flAllocationType: Flag[MemAlloc],
      flProtect: MemProt
    ): pointer {.importc, stdcall, header: "windows.h", sideeffect.}

  proc VirtualProtect*(
      lpAddress: pointer,
      dwSize: int;
      flNewProtect :MemProt,
      lpflOldProtect: ptr MemProt
    ): cint {.importc, stdcall, header: "windows.h", sideeffect.}

  proc VirtualFree*(
      lpAddress: pointer,
      dwSize: int;
      dwFreeType: MemFree
    ): cint {.importc, stdcall, header: "Kernel32", sideeffect.}

else:
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
