# Runtime: Multithreading, Task scheduling, NUMA

Research on a multithreading runtime

Currently Laser target OpenMP but might
encounter roadblocks down the line.

Furthermore research shows that GCC libgomp has a potential bottleneck for tasking due to scheduling via a central list protected by a mutex.

## Constraints

- Work for JIT (see https://github.com/numforge/laser/issues/31)
  - OpenMP doesn't though support might come in the following year (http://lists.flang-compiler.org/pipermail/flang-dev_lists.flang-compiler.org/2019-May/000197.html)
- NUMA-aware memory allocation and threading (https://github.com/numforge/laser/issues/30)
- Support nested loops.
  - Biggest design flaw of OpenMP, if the OpenMP-parallel program is called from inside another parallel section, each thread will create N new threads unless OMP_NESTED=false.
  - Lux compiler will optimize across functions so a whole NN network might only declare a single parallel region at the top, but a task queue might be better when doing graph level scheduling, for example to schedule beam search or GANs

## Research

### Memory allocation

- NUMA aware OpenMP allocator:
  ![](./media/omp_numa_alloc.png)

- From Intel's TBB: https://techdecoded.intel.io/essentials/turbocharge-your-c-code-efficient-memory-allocation-for-increased-performance/#gs.p065tg

### Papers

- Memory and data aware scheduling, 2018, L. Marchal

  https://hal.inria.fr/tel-01934712/document

  http://perso.ens-lyon.fr/loris.marchal/hdr/slides.pdf

- Correct and Efficient Work-Stealing for Weak Memory Models

  https://www.di.ens.fr/~zappa/readings/ppopp13.pdf

- Work stealing on the critical path

  http://mao.imag.fr/reu03avr08/Cerin_paper.pdf

- Scheduling Parallel Programs by
Work Stealing with Private Deques

  https://www.chargueraud.org/research/2013/ppopp/full.pdf

- Dynamic Circular Work-Stealing Deque

  https://www.dre.vanderbilt.edu/~schmidt/PDF/work-stealing-dequeue.pdf

- Scheduling Multithreaded Computations by Work Stealing
  http://supertech.csail.mit.edu/papers/steal.pdf

### Implementation: Shared memory parallelism and tasking

- A Comparative Critical Analysis of Modern Task-Parallel Runtimes

  https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2012/1210594.pdf

- OMP proc_bind for NUMA-aware threading
  ![](./media/omp_proc_bind.png)

- Intel's Thread Building Blocks, see task scheduler details: https://software.intel.com/en-us/node/506294
  and architecture of the library from https://techdecoded.intel.io/big-picture/adventures-in-threading-how-tbb-is-advancing-parallelism/#gs.p062y9
  ![](./media/TBB Functions.png)

- Extending Intel OpenMP with libkomp + multicore-myltiGPU scheduling:
  https://calcul.math.cnrs.fr/attachments/spip/Documents/Journees/janv2017/kaapi_libkomp.pdf

- OpenMP runtime's: GNU's libGOMP, Intel's libOMP, Inria's libkomp, Inria's XKaapi

  https://hal.archives-ouvertes.fr/hal-01666343/file/wamca2017.pdf

- Phylanx: Distributed Array Processing on top of HPX

  https://github.com/STEllAR-GROUP/phylanx

  http://stellar.cct.lsu.edu/pubs/wei_riken_2019_phylanx_poster.pdf

- HPX: Higher-level Parallelization in C++ for Asynchronous
Task-Based Programming

  https://calcul.math.cnrs.fr/attachments/spip/Documents/Journees/janv2017/parallelism_in_cpp_sc15.pdf

  https://github.com/STEllAR-GROUP/hpx

  http://stellar.cct.lsu.edu/pubs/pgas14.pdf

  https://indico.cern.ch/event/513104/contributions/2035127/attachments/1275186/1891535/Diving_into_Parallelization_in_Modern_C_using_HPX_-_Unleashing_the_full_power.pdf

- OpenMP implemented on top of HPX

  https://arxiv.org/abs/1903.03023

  https://github.com/STEllAR-GROUP/hpxMP

- Adobe's Stlib, concurrency/parallelism for image editing primitives: https://github.com/stlab/libraries/tree/develop/stlab/concurrency
- Sean Parent (Adobe)'s presentation "Better Code: COncurrency" on how to build a task system: https://youtu.be/zULU6Hhp42w?t=1695

- Libdispatch / Grand Central Dispatch: https://github.com/apple/swift-corelibs-libdispatch

- Grand Central Dispatch Internals: http://newosxbook.com/articles/GCD.html

- Rust's Rayon
  - https://github.com/rayon-rs/rayon/blob/master/FAQ.md
  - https://www.youtube.com/watch?v=gof_OEv71Aw

- Halide runtime:
  - https://github.com/halide/Halide/blob/master/src/runtime/posix_threads.cpp
  - https://github.com/halide/Halide/blob/master/src/runtime/runtime_api.cpp
  - https://github.com/halide/Halide/blob/master/src/runtime/thread_pool_common.h

- Cpp Taskflow: Fast Task-based Parallel Programming using Modern C++. (with DNN examples vs OpenMP and TBB)

  https://github.com/cpp-taskflow/cpp-taskflow

  https://tsung-wei-huang.github.io/talk/ipdps19-presentation.pdf

  https://tsung-wei-huang.github.io/papers/ipdps19.pdf

  https://woset-workshop.github.io/PDFs/a3.pdf

- Transwarp: header-only task scheduling via a DAG of tasks, https://github.com/bloomen/transwarp

- Staccatto: Work-Stealing Task Scheduler, https://github.com/rkuchumov/staccato


### Heterogeneous arch scheduling

- Kokkos: Performance portability ecosystem

  with HBM memory, Cuda UVM and AMD ROCm support

  https://github.com/kokkos/kokkos

  https://cfwebprod.sandia.gov/cfdocs/CompResearch/docs/Trott-white-paper.pdf

  https://www.osti.gov/servlets/purl/1457941

- Programming modern HPC platforms

  Scalability study with 1152 cores + 288 GPUs

  https://calcul.math.cnrs.fr/attachments/spip/Documents/Journees/janv2017/tasks_programming_and_starpu.pdf

- StarPU: A Unified Runtime System for Heterogeneous Multicore Architectures, http://starpu.gforge.inria.fr

- High performance scheduling of mixed-mode DAGs on heterogeneous multicores, 2019

  https://arxiv.org/pdf/1901.05907.pdf

- Xkaapi OpenMP runtime: http://kaapi.gforge.inria.fr/#!index.md

- Kstar OpenMP compiler: http://kstar.gforge.inria.fr/#!index.md

- XKaapi: A Runtime System for Data-Flow Task
  Programming on Heterogeneous Architectures, 2013

  https://hal.inria.fr/hal-00799904/document
