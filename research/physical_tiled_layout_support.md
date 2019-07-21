# Physical Layout mapping for accelerators

Some accelerators like TPU or libraries like MKL-DNN
supports tiled layout with significantly improved performance.

GEMM prepacking also uses a tiled/swizzled layout.

## Current Tiled-layout References

- [MKL-DNN Understanding Memory Format](https://intel.github.io/mkl-dnn/understanding_memory_formats.html) and [corresponding paper](https://arxiv.org/pdf/1602.06709v1.pdf).
- [XLA Tiled layout](https://www.tensorflow.org/xla/tiled_layout) for Google's TPUs
- [Tiling to HWC8h8w8c in MLIR](https://groups.google.com/a/tensorflow.org/forum/#!topic/mlir/yskqytmUpOU)
- [MLIR Layout map](https://github.com/tensorflow/mlir/blob/165c2e06810efac989b3c83e146f9dd52144b740/g3doc/LangRef.md#layout-map), note a rework is planned

## Past Tiled-Layout References

- [Programming for Parallelism and Locality with Hierarchically Tiled Arrays](https://www.lrde.epita.fr/~bleton/doc/bikshandi06programming.pdf)
- HTA library: http://polaris.cs.uiuc.edu/hta/index.html
- Intel TBB vs HTA: [Task-parallel versus data-parallel library-based programming in multicore systems](http://www.des.udc.es/~basilio/papers/andradehtas.pdf)

## Space-Filling curve references

- [Alternative Array Storage Layouts for Regular Scientific Programs](https://pdfs.semanticscholar.org/3030/22b1c442f543d6794a2171e3dfcd9ff149cb.pdf), Thiyagalingam Jeyarajan, Thesis, 2005
- Matrix multiplication using SFC, [A Study of Energy and Locality Effects using Space-filling Curves](https://arxiv.org/pdf/1606.06133.pdf)
- Morton-order vs MKL and libxsmm, [https://hal.inria.fr/hal-02082524/document](https://hal.inria.fr/hal-02082524/document), 2019
- [Recursive Array Layouts and Fast Parallel Matrix Multiplication](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.452&rep=rep1&type=pdf), 1999

Note focused on HPC:
- [A Parallel N-Dimensional Space-Filling Curve library](https://www.mdpi.com/2220-9964/7/8/327/pdf)

## Implementation notes

This brings overhead on every array access unless
this is a static parameter and so part of the Tensor type.
Having it part of the type will hurt tensor operations' composability however.
On another hand, if that attribute is dynamic, it will lead to code-bloat with a code branch for tiled layout and a code branch for "normal" tensor that only needs shape + strides.

Given the unclear advantage at the moment, it might be best when only used with JIT code generation.
