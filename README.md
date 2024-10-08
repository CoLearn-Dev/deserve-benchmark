# DeServe Benchmark

Before running the benchmark, take a look at the source code of available options. 

## Download Datasets

```bash
git lfs install
git clone https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered datasets/ShareGPT_Vicuna_unfiltered
```

## Benchmarking Online vLLM

```bash
python -m src.benchmark.online_vllm
```

## Benchmarking Offline vLLM

```bash
python -m src.benchmark.offline_vllm
```

