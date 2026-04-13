# Preparation

## Hardware and software

- Linux with NVIDIA GPUs and a recent CUDA stack compatible with your PyTorch build.
- Python 3.10+ is typical for SGLang; follow the version range stated in each submodule’s `python/pyproject.toml`.

## Clone this repository

```bash
git clone --recurse-submodules https://github.com/togethercomputer/System-Aware-4-Bit-KV-Cache-Quantization.git
cd System-Aware-4-Bit-KV-Cache-Quantization
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

If nested `tore-eval` fails (private remote), you can still use **simple-evals** and the scripts in [../scripts](../scripts); see [SUBMODULE_VERSIONS.md](../SUBMODULE_VERSIONS.md).

## Build SGLang (each fork)

From the repository root:

```bash
# Fast rotation fork (inference + benchmarks)
cd third_party/sglang-fast-rotation/python
pip install -e ".[all]"   # or a slimmer extra set per upstream docs

# K-means fork (accuracy matrix + calibration)
cd ../../sglang-kmeans/python
pip install -e ".[all]"
```

Install **extra dependencies** used by this project:

```bash
# Block Hadamard on K/V before INT4 (BDR path)
pip install fast_hadamard_transform

# K-means fork expects flash-kmeans (see submodule pyproject optional deps)
pip install "flash-kmeans @ git+https://github.com/jindajia/flash-kmeans.git"
```

## OpenAI simple-evals (accuracy)

Accuracy numbers in the paper use **[openai/simple-evals](https://github.com/openai/simple-evals)** in the **zero-shot chain-of-thought** style against an OpenAI-compatible server.

```bash
git clone https://github.com/openai/simple-evals.git
cd simple-evals
pip install -e .
pip install openai tqdm numpy
```

You will point the OpenAI Python client at SGLang’s HTTP API (see [03-evaluation-matrix.md](03-evaluation-matrix.md)).

## Hugging Face models

Download or cache the model weights you evaluate (for example `Qwen/Qwen3-8B`). SGLang accepts `--model-path` with a local directory or a Hub id where supported.
