# Physical Layout mapping for accelerators

Some accelerators like TPU or libraries like MKL-DNN
supports tiled layout with significantly improved performance.

GEMM prepacking also uses a tiled/swizzled layout.

## References

- [MKL-DNN Understanding Memory Format](https://intel.github.io/mkl-dnn/understanding_memory_formats.html) and [corresponding paper](https://arxiv.org/pdf/1602.06709v1.pdf).
- [XLA Tiled layout](https://www.tensorflow.org/xla/tiled_layout) for Google's TPUs
- [Tiling to HWC8h8w8c in MLIR](https://groups.google.com/a/tensorflow.org/forum/#!topic/mlir/yskqytmUpOU)
- [MLIR Layout map](https://github.com/tensorflow/mlir/blob/165c2e06810efac989b3c83e146f9dd52144b740/g3doc/LangRef.md#layout-map), note a rework is planned

## Implementation notes

This brings overhead on every array access unless
this is a static parameter and so part of the Tensor type.
Having it part of the type will hurt tensor operations' composability however.
On another hand, if that attribute is dynamic, it will lead to code-bloat with a code branch for tiled layout and a code branch for "normal" tensor that only needs shape + strides.

Given the unclear advantage at the moment, it might be best when only used with JIT code generation.
