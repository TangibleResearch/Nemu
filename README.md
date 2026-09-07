# Tangible Nemu

[![CI/CD](https://github.com/TangibleResearch/Neuron/actions/workflows/ci.yml/badge.svg)](https://github.com/TangibleResearch/Neuron/actions/workflows/ci.yml)

**Tangible Neuron** is an experimental AI-first processor architecture developed as a research project by Tangible Research.

Neuron explores how a processor can be designed around modern artificial intelligence workloads rather than treating AI acceleration as an afterthought. The project currently focuses on processor architecture, instruction-set design, vector and matrix execution, dataflow acceleration, compiler optimization, and hardware/software co-design.

Tangible Nemu is the Emulator project for Neuron to test how Neuron would possibly handle instructions.

## Current Direction

Neuron is being designed as a heterogeneous architecture containing multiple forms of computation:

* Scalar processing for general-purpose control and program execution
* Vector processing for highly parallel operations
* Matrix and tensor processing for AI workloads
* A programmable dataflow-style AI execution fabric
* Dedicated architectural state for AI execution and accelerator control

The current experimental register architecture includes:

* `R0-R15` — 32-bit scalar registers
* `V0-V7` — vector registers
* `M0-M3` — matrix/tensor registers
* `P0-P3` — predicate registers
* `PC` — program counter
* `SP` — stack pointer
* `FP` — frame pointer
* `STATUS` — processor status and condition flags
* AI, quantization, sparsity, and tensor control registers

## Research Goals

Tangible Neuron is intended to investigate questions such as:

* How should an AI-first ISA differ from a conventional CPU ISA?
* Which operations belong in scalar, vector, matrix, or dataflow hardware?
* Can frequently repeated AI computation graphs be mapped onto interconnected processing elements?
* How should a compiler automatically discover and schedule parallel AI workloads?
* How can data movement and memory bandwidth be reduced?
* How should hardware threading interact with vector and AI execution?
* Which numerical formats should be efficiently supported for training and inference?
* How should the architecture balance specialization with long-term programmability?

## Current Work

The project currently includes an early Neuron32 instruction-set simulator written in Rust.

Development is progressing through research and experimentation in:

1. ISA design
2. Scalar execution
3. Vector execution
4. Matrix/tensor execution
5. Dataflow acceleration
6. Memory architecture
7. Hardware threading
8. Compiler and optimizer design
9. Display and device architecture
10. Architecture simulation and benchmarking


## Status

**Early research and architecture development.**

The ISA, execution model, accelerator architecture, and compiler design are experimental and subject to significant change.

## Development

Run the same checks used by CI locally:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
cargo build --release --locked
```

GitHub Actions runs these checks for every push and pull request. After a successful push to `main`, it publishes `index.html` and `Neuron.png` to GitHub Pages. In the repository's **Settings → Pages**, select **GitHub Actions** as the publishing source once before the first deployment.

