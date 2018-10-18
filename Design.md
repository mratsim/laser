# Design document

## Allocation

Besides the allocating procs, Laser does not allocate.
Creating or reusing buffers is left at the discretion of the high-level lib.
Note that the CPUStorage is GC-ed to keep track of references to
the same memory location.

A memory pool will probably be provided in the future

## Inlining

If there is no clear benefit, Laser does not specify if a proc is inline.
This limits code size explosion but allow the compiler to optimize if needed.
Code size is especially important as many time memory and cache access is
the limiting factor.
