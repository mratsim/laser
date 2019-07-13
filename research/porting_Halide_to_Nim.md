# Transcript of Arraymancer #347 "Porting Halide to Nim"

# @timotheecour - 2019-01-28

opening this issue to track ideas for how to port Halide to nim.
This could be either using halide in Nim via:
* A1: a wrapper (eg using nimterop) or
* A2: re-implementing some ideas in nim directly, which although a lot of work would result in a much simpler solution given nim's better DSL abilities

## proposal: approach 1
use a plugin at CT approach as done in nimterop (https://github.com/genotrance/nimterop/pull/66) to transparently compile (using halide) a c++/Halide DSL, save to a shared library and cache it; eg:

```nim
# this is in Nim

import halide_funs

cPluginHalide: """ // this is compiled on the fly and cached, using a hash of the string, eg saved to: /tmp/nimhalide/libplugin_12abf23.dylib
// this is C++ halide code
Func blur_3x3(Func input) { // example from https://en.wikipedia.org/wiki/Halide_(programming_language)
  Func blur_x, blur_y;
  Var x, y, xi, yi;

  // The algorithm - no storage or order
  blur_x(x, y) = (input(x-1, y) + input(x, y) + input(x+1, y))/3;
  blur_y(x, y) = (blur_x(x, y-1) + blur_x(x, y) + blur_x(x, y+1))/3;

  // The schedule - defines order, locality; implies storage
  blur_y.tile(x, y, xi, yi, 256, 32)
        .vectorize(xi, 8).parallel(y);
  blur_x.compute_at(blur_y, x).vectorize(x, 8);

  return blur_y;
}"""

proc main() =
  let image = loadImage("img.jpg")
  blur_3x3(image)
```

## [EDIT] proposal: approach 2
using https://github.com/fragcolor-xyz/nimline could be another possiblity, but would probalby need adaptation as halide compiler needs to be run on C++/halide program

## related
* halide: https://en.wikipedia.org/wiki/Halide_(programming_language)
* some discussion on gitter https://gitter.im/Arraymancer/Lobby?at=5c4f1b7d13a2814df6ded8f5 + a few hours before that

* https://github.com/mratsim/Arraymancer/issues/203 briefly mentions halide but wasn't the focus of that issue

* other language integrations:
  * python: https://stackoverflow.com/questions/34439657/halide-with-c-layout-numpy-arrays
  * rust: [Rust Image Processing DSL – dpzmick.com](http://dpzmick.com/2016/08/11/rust-jit-image-processing/)

* tutorial http://halide-lang.org/tutorials/tutorial_lesson_16_rgb_generate.html

* nlvm: https://github.com/arnetheduck/nlvm

# @mratsim - 2019-01-31

As discussed, this sounds like a very interesting idea and from a learning perspective I'd rather implement it from scratch, just like Arraymancer started as a way to learn NN from scratch instead of reusing Tensorflow or Torch.

## Overview of Halide

Halide is a C++ domain specific language for computational image processing. It started with the idea of separating the algorithm (the what) from its efficient hardware implementation (the how). This is similar to a (functional programming) source file vs machine code when passed through an optimizing compiler.

The "how" can be hinted to assist the compiler in its threading and vectorization by specifying cache blocking parameters, unrolling factor and threading hints.

Furthermore Halide supports multiple backends (x86, ARM, Vulkan, DX12, OpenGL, OpenCL, Cuda ...) including a JIT backend that will dispatch with the latest instruction set supported (for example AVX512).

The pipeline is: C++ DSL --> Halide Intermediate Representation (Computational Graph) --> LLVM (AOT or JIT) --> machine code.

## Advantages

### In general

- Short code, much faster to develop new efficient algorithms.
- No need to fiddle with platform specific optimisations like SIMD.
- Portability
- Computation can be fused across functions and more parallelism can be exposed.
- Much easier to test different parallel schedule

### For ML and DL

The main advantages are the portability with SIMD optimisations opportunities and computation fusion to prevent intermediate allocation and results.

Parallelism in ML/DL is either straightforward (it's a single loop) or very complex and would need manual involvement:

- For example Halide is memory-bound on matrix multiplication (0.5~1 TFLOP/s at most while I can reach [1.8TFLOP/s with OpenBLAS or MKLDNN](https://github.com/numforge/laser/blob/56df4643530ada0a01d29116feb626d93ac11379/benchmarks/gemm/gemm_bench_float32.nim#L379-L465))
- Halide is unsuited for trees ensembles like random forests

Some DL layers that would benefits from Halide-like functionality:
  - Convolution (apparently Halide has a schedule that reaches MKL-DNN performance)
  - Locally-Connected Layer/Unshared Convolution (#255). Though this might be faster with summed-area table/integral image and then a simple for loop.
  - Bilinear
  - Pooling layers (Max, Average, Region of Interest Pooling)

### Disadvantages

1. Halide is pretty much a static graph compiler like Tensorflow XLA which is not that great for research and debugging. It's a pain to use Tensorflow for NLP, RNNs and GANs for example.

    **The main issue is the lack of control flow like for loop and if/else branches.**

Note that OpenAI Dali (not to be confused with Nvidia DALI) apparently is a compiler that plugs in dynamic graphs framework.

2.  Another disadvantage is the LLVM dependency at runtime, but that only concerns JIT. LLVM is a huge library.

### Halide papers

- Jonathan Ragan-Kelley thesis: https://people.csail.mit.edu/jrk/jrkthesis.pdf
- Halide a language and compiler for optimising, parallelism, locality and recomputation
  in image processing pipelines: http://people.csail.mit.edu/jrk/halide-pldi13.pdf
- Compiling high performance recursive filters: http://www-sop.inria.fr/reves/Basilic/2015/CRPDD15a/recfilter.pdf
- CVPR slides - recursive filtering: https://halide-lang.org/assets/lectures/CVPR_Halide_RecursiveFiltering.pdf
- Parallel Associative Reduction in Halide: https://andrew.adams.pub/par.pdf
- Automatically Scheduling Halide Image Processing Pipelines: http://graphics.cs.cmu.edu/projects/halidesched/mullapudi16_halidesched.pdf


### Beyond Halide

Halide has been expanded with autodiff capabilities in [gradient-halide](https://people.csail.mit.edu/tzumao/gradient_halide/) which may greatly enhance autograd which is a recurrent source of slowness in Arraymancer due to the need of a DAG which doesn't play well with reference counting. However while gradient-halide results are promising, TVM (a fork of Halide for DL) [reported perf issues](https://discuss.tvm.ai/t/automatic-differentiation-on-the-level-of-tvm-ir/781)

### Related projects

- Tensorflow XLA: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/g3doc/overview.md
- Tensorflow MLIR: https://github.com/tensorflow/mlir
- PyTorch Glow: https://github.com/pytorch/glow
- Dask: https://dask.org
- Gradient Halide: https://people.csail.mit.edu/tzumao/gradient_halide/
- OpenAI Dali (can't find the repo): http://learningsys.org/nips18/assets/papers/100CameraReadySubmissionDali__Lazy_Compilation_of_Dynamic_Computation_Graphs.pdf
- DLMC TVM: https://github.com/dmlc/tvm
- Intel PlaidML: https://github.com/plaidml/plaidml
- Intel Nervana Ngraph: https://github.com/NervanaSystems/ngraph/tree/master/python
- The Tensor Algebra Compiler: https://github.com/tensor-compiler/taco
- Numba: http://numba.pydata.org (less related as it doesn't compile Expression Graphs)
- PyGPU (2006): An embedded DSL for high speed image processing
    - [Techniques for implementing embedded domain specific languages in dynamic languages](https://pdfs.semanticscholar.org/731a/19929d8b10c483eb46d6d8a9f27d5607f9d2.pdf)
    - [High-speed GPU Programming](https://pdfs.semanticscholar.org/731a/19929d8b10c483eb46d6d8a9f27d5607f9d2.pdf) (thesis)
- Stanford's Delite and OptiML (2011):
  - [Delite: A Compiler Architecture for Performance-Oriented Embedded Domain-Specific Languages](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.416&rep=rep1&type=pdf)
  - [An heterogeneous parellel framework for DSL](https://stanford-ppl.github.io/website/papers/pact11-brown.pdf)
  - [slides](http://on-demand.gputechconf.com/gtc/2012/presentations/S0365-Delite-A-Framework-for-Implementing-Heterogeneous-Parallel-DSLs.pdf)
  - [OptiML: An Implicitly Parallel Domain-Specific Language for
Machine Learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.220.1967&rep=rep1&type=pdf)
  - [Building Blocks for Performance oriented DSL](https://arxiv.org/pdf/1109.0778.pdf)
  - http://stanford-ppl.github.io/Delite/optiml/index.html
  - https://github.com/stanford-ppl/hyperdsl
- DLVM Deep Learning Virtual Machine (2017 - Authors migrated to Swift for Tensorflow)
  - https://arxiv.org/pdf/1711.03016.pdf
  - https://github.com/dlvm-team/dlvm-core
- TriNNity:
  - https://www.scss.tcd.ie/~andersan/projects/live/triNNity.html
  - https://bitbucket.org/STG-TCD/trinnity/src/master/
- Graphi, scheduling deep learning computation graphs on manycore CPU:
  - https://arxiv.org/abs/1807.09667
- CGT: Computational Graph Toolkit (computation graph and autodiff on multidimensional arrays algorithm, with C and Python API):
  - https://github.com/joschu/cgt/
  - http://rll.berkeley.edu/cgt/
- PolyMage: Automatic Optimization for image processing pipelines:
  - Note:
    - Polymage use a Python DSL that compiles to C++, including OpenMP and ivdep pragma
    - It uses polyhedral optimization with several tiling algorithms (notable parallelogram and overlapped)
    - Author later refactored Halide auto-scheduler
  - http://mcl.csa.iisc.ac.in/polymage/asplos235-mullapudi.pdf
  - http://mcl.csa.iisc.ernet.in/downloads/slides/PolyMage.pdf
  - http://mcl.csa.iisc.ac.in/polymage.html
  - https://bitbucket.org/udayb/polymage/
- SMIL / FREIA: A Dynamic to Static DSL Compiler for Image
Processing Applications
  - https://www.cri.ensmp.fr/classement/doc/E-398-slides.pdf
  - https://www.cri.mines-paristech.fr/classement/doc/A-640.pdf
  - https://www.cri.mines-paristech.fr/classement/doc/A-670.pdf
- DeepDSL (Scala): A Compilation-based Domain-Specific Language for Deep Learning
  - https://github.com/deepdsl/deepdsl
  - https://openreview.net/pdf?id=Bks8cPcxe
- Lift (Scala): A Functional Data-Parallel IR for
High-Performance GPU Code Generation
  - https://www.lift-project.org/publications/2017/steuwer17LiftIR.pdf

### Auto-scheduling - Polyhedral tiling

- The Promises of Hybrid
Hexagonal/Classical
Tiling for GPU: https://hal.inria.fr/hal-00848691/document
- Applying the Polyhedral Model to Tile Loops
in Devito: https://pdfs.semanticscholar.org/d0bd/7c5f51c80c3236e99009298734291745e2e7.pdf
- Jie Zhao:
  - A Combined Language and Polyhedral Approach for
Heterogeneous Parallelism: https://tel.archives-ouvertes.fr/tel-01988073/file/thesis.pdf
  - A General Purpose Automatic Overlapped Tiling
Technique in Polyhedral Frameworks: https://www.di.ens.fr/~zhaojie/cgo2018-poster-presentation
- Tiramisu: TIRAMISU: A Polyhedral Compiler for Expressing. Fast and Portable Code.
  - Note: supports RNN
  - https://arxiv.org/pdf/1804.10694.pdf
  - https://andreask.cs.illinois.edu/cs598apk-f18/talks/njchris2.pdf
  - http://tiramisu-compiler.org
  - https://github.com/Tiramisu-Compiler/tiramisu

### Approaches to porting

#### Wrapping
I'm not really fan of wrapping unless we can do away with CMake and completely configure everything via Nim. This is the approach I took for the Arcade Learning Environment for reinforcement learning here with a [Nim build script that basically replaces CMake](https://github.com/numforge/agent-smith/blob/a2d9251e289f92f6b5fb68e19a98d16b00f2694c/third_party/ale_build.nim) and the DLL built can be called from another Nim: https://github.com/numforge/agent-smith/blob/master/third_party/ale_wrap.nim. This preserves C++ name mangling, puts everything in nimcache and can cleanly [made into a nimble task](https://github.com/numforge/agent-smith/blob/a2d9251e289f92f6b5fb68e19a98d16b00f2694c/agent_smith.nimble#L34-L41).

The alternative is something like nimtorch which requires either a build script (conda in NimTorch case), potentially a docker or [building the library apart and then copying it](https://github.com/fragcolor-xyz/nimtorch#manual-way-without-requiring-conda).

Once wrapped we need a way for Nim to generate valide Halide DSL, this means removing all the fluff Nim adds to a C/C++ file. Interestingly this is a very similar problem to generating Cuda kernel like what is done in [Cudanim](https://github.com/jcosborn/cudanim), this would also apply to OpenCL code generator idea from [2014](https://github.com/nim-lang/Nim/wiki/GSoC-2014-Ideas#add-a-code-generator-for-opencl), [2015](https://github.com/nim-lang/Nim/wiki/GSoC-2015-Ideas#add-a-code-generator-for-opencl), [2016](https://github.com/nim-lang/Nim/wiki/GSoC-2016-Ideas#add-a-code-generator-for-opencl)

#### Reimplementing from scratch

As said in the beginning, from a learning perspective it's the most interesting though the time-to-market and the testing (+ fuzzing) would probably be years away.

Interestingly, it seems like Halide started as FImage and had it's own assembler. The IR looks already quite similar to what we have now after only 3 months: https://github.com/halide/Halide/tree/5a2f90b7bcf4d9298e4068698f52731401e07dc8.

There are 2 big parts of Halide from an engineering perspective:

- The frontend, which converts the DSL to Halide IR.
- A lowering step from Halide IR to LLVM IR.
- The backend which uses LLVM to generate code for the different targets.

Skipping the front-end since that has to be implemented to discuss the backend.

##### Intermediate representation

For the IR there are several choices: Halide IR, [Glow IR](https://github.com/pytorch/glow/blob/master/docs/IR.md), TVM IR, SPIR-V or LLVM IR directly

##### LLVM backend

Assuming we want to support lots of framework, I don't see any alternative to LLVM for the backend both from a license, a platform and a maturity point of view which comes with it's own cost, notably compilation overhead and the space LLVM takes.

Regarding API, I'm not yet sure which one Halide uses but LLVM offers 2:

- The stable MCJIT: https://llvm.org/docs/MCJITDesignAndImplementation.html
- The new Orc JIT: https://llvm.org/docs/tutorial/BuildingAJIT1.html

##### Written from scratch backends

###### Jit backend

Assuming we target a couple specific platforms that shares a lot of similarity in the programming model (for example X86, ARM, Cuda, ROCm), this is also possible. Furthermore as you said, using Nim offers a tremendous metaprogramming advantage as evidenced by the very clean way i could implement my own JIT assembler in Laser:

https://github.com/numforge/laser/blob/56df4643530ada0a01d29116feb626d93ac11379/laser/photon_jit/x86_64/x86_64_ops.nim

```Nim
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
```

JIT is the approach taken for x86-only libs:
  - Intel libxsmm - JIT from scratch: https://github.com/hfp/libxsmm
  - MKLDNN - [Xbyak JIT](https://github.com/herumi/xbyak): https://github.com/intel/mkl-dnn
  - PyTorch FBGEMM - [asmjit](https://github.com/pytorch/FBGEMM)

There are significant challenges though:

  - Making that robust (need fuzzing)
  - Caching (Intel and AMD are still not caching their JIT-ted OpenCL kernels for reference)
  - Thread-safe
  - Register allocations
  - Optimizing

###### AOT backend

We can also do ahead of time compilation with runtime dispatch depending on the target. This is the approach I take with [Laser GEMM](https://github.com/numforge/laser/tree/56df4643530ada0a01d29116feb626d93ac11379/laser/primitives/matrix_multiplication) with a version for SSE, SSE2, SSE4.1, AVX, AVX+FMA, AVX2, AVX512, ... and we can add Cuda, OpenCL, etc.

# @timotheecour - 2019-02-01

note that we can generate LLVM IR from Nim (or a DSL if we have a macro DSL=>nim); see this demo https://github.com/timotheecour/vitanim/blob/master/testcases/tests/t0127.nim shows how we can generate LLVM IR from a DSL

* the DSL could be either pure nim:
```
DSLtoLLVMIR:
  proc entryPoint() {.exportc.} =
    for i in 0..<10:
      echo i*i
```

* or something custom for image processing, as you were referring to, eg
```
DSLtoLLVMIR: # front-end that converts DSL to nim is needed here
  forEach x in a, y in b, z in c:
    z = x + y
```

# @mratsim - 2019-02-01

I find the need to go through a text file quite clunky and slow in the long term.

Since LLVM has a [bit code representation ](https://llvm.org/docs/BitCodeFormat.html) of its IR I was thinking of targeting that directly using bindings akin to [LLVM-D](https://github.com/MoritzMaxeiner/llvm-d).

This is also [what Halide is doing.](https://github.com/halide/Halide/blob/6169020460efed0cff6973d4c29e81c1ceedccdc/src/CodeGen_Internal.h#L94-L98).

From an interface point of view, I probably can re-use the JIT Assembler for x86-64 [I created here](https://github.com/numforge/laser/tree/27e4a77c7808ac5b5dcee08ddf3baa320aed47c8/laser/photon_jit/x86_64) except that instead of assembling x86 I'm assembling LLVM IR. [(Usage)](https://github.com/numforge/laser/blob/27e4a77c7808ac5b5dcee08ddf3baa320aed47c8/examples/ex07_jit_brainfuck_vm.nim#L45-L85).

It can be restricted to the [subset valid for Nvidia targets.](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html)

Alternatively, it seems like we may be able [to directly JIT an AST representation](https://llvm.org/docs/tutorial/BuildingAJIT4.html) without having to go through LLVM IR phase.

## @mratsim - 2019-02-06

Some advances in my research (will find a place to store that in the future, probably in [Laser](https://github.com/numforge/laser))

## Code generation

- Kai Nacke, maintainer of LDC (LLVM D Compiler), did a very nice talk about what he learned by implementing his own compiler from scratch at FOSDEM 2019. [Slides](https://fosdem.org/2019/schedule/event/llvm_irgen/attachments/paper/3170/export/events/attachments/llvm_irgen/paper/3170/FOSDEM_2019_slides.pdf).

## Alternative HPC compiler

- http://okmij.org/ftp/meta-programming/HPC.html and the Shonan challenge, several HPC problems with a known tuned manual solution that would be nice to automatically optimize:
    - Paper - test cases: http://okmij.org/ftp/meta-programming/Shonan-challenge.pdf
    - Report: http://okmij.org/ftp/meta-programming/shonan2-report.pdf
    - 1D Conv optimisation in OCaml: http://okmij.org/ftp/meta-programming/lift/convolution.ml

## Semantic pass and the extension problem

Halide and PyTorch Glow are using the visitor pattern which looks to me like a workaround for a not expressive enough language.

We need to be able to extend functions without touching the core ones and if possible data types as well.

For example we could have the functions `Add` and `Mul` defined on `Tensor` and `CudaTensor` today and a user library implement `Div` and also extend `Add` and `Mul` on `VulkanTensor`.

Data types are probably not that critical, i.e. we could use ADTs + Generics for Tensor[float32], VulkanTensor[float16], ...

In research, there are several approaches to handle that:

  - Tagless Final, embedding DSL into a typed language: http://okmij.org/ftp/tagless-final/ and http://okmij.org/ftp/tagless-final/course/lecture.pdf
  - Object Algebra: http://ponies.io/posts/2015-07-15-solving-the-expression-problem-in-python-object-algebras-and-mypy-static-types.html
  - Attribute Grammar (needs lazy eval):
        - https://www.reddit.com/r/haskell/comments/6ugzhe/walking_the_ast_without_the_visitor_pattern/
        - https://play.rust-lang.org/?gist=1f7e5a01ec3d0010c50ecbccf93ad66b&version=nightly
        - [Functional Implementation of Multiple Traversals Program with Attribute Grammars in Scala](www.scholarpublishing.org/index.php/TMLAI/article/download/3397/1944)
  - Catamorphisms: https://medium.com/@olxc/catamorphisms-and-f-algebras-b4e91380d134
  - Transducers: http://sixty-north.com/blog/deriving-transducers-from-first-principles.html
  - Visitor vs Interpreter Pattern: https://homepages.cwi.nl/~storm/publications/visitor.pdf
  - Datatypes à la carte: http://www.cs.ru.nl/~W.Swierstra/Publications/DataTypesALaCarte.pdf
  - Functional Lenses / Multiplates: http://de.arxiv.org/ftp/arxiv/papers/1103/1103.2841.pdf
  - Monads (and Free Monads?):
    - [Generic Monadic Construcs for Embedded Languages](https://pdfs.semanticscholar.org/0c92/4ead3a6d833e049d8f224fd0526f47760336.pdf)
  - Object Algebra, traversals, visitor patterns, church encoding: [paper](http://www.informatik.uni-marburg.de/~rendel/oa2ag/rendel14object.pdf) and [slides](http://ps.informatik.uni-tuebingen.de/2014/10/23/presentation-at-oopsla/slides.pdf).
  - Developping extensible shallow embeddings (functions in host lang) on top of deep embeddings (fixed AST/IR):
      - [Folding Domain Specific Language: Deep and Shallow Embedding](https://www.cs.ox.ac.uk/people/jeremy.gibbons/publications/embedding-short.pdf)
      - [Combining Deep and SHallow Embedding for EDSL](http://www.cse.chalmers.se/~josefs/publications/TFP12.pdf) and [paper](http://www.cse.chalmers.se/~josefs/publications/svenningsson2015combining.pdf).
      - [Yin-Yang: Concealing the Deep Embedding of DSLs](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.697.1002&rep=rep1&type=pdf) and [project report](https://geirsson.com/assets/directembedding-report.pdf).
      - [Uniting Language Embedding for Fast and Friendly DSLs (thesis)](https://infoscience.epfl.ch/record/218036/files/EPFL_TH6882.pdf)
      - [Rewriting a Shallow DSL using a GHC extension](https://ku-fpg.github.io/files/Grebe-17-Transformations.pdf)
      - [An Image Processing Language: External and Shallow/Deep Embeddings](http://www.macs.hw.ac.uk/~rs46/papers/rwdsl2016/rwdsl-2016.pdf)
