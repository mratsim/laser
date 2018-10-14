# Laser - Data-parallel primitives for high performance tensor and image processing

Carefully-tuned primitives for running tensor and image-processing code
on CPU, GPUs and accelerators.

The package includes `cpuinfo` by Facebook for runtime CPU detection:
  - supports x86, x86_64, ARM and ARM64
  - Detection of cache (L1, L2, L3), size and associativity
  - Detection of hyperthreading

This will be the low-level backend of [Arraymancer](https://github.com/mratsim/Arraymancer).

## License

Laser is licensed under the Apache License version 2
Facebook's cpuinfo is licensed under Simplified BSD (BSD 2 clauses)
