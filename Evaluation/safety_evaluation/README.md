# SafetyBench

SafetyBench Is a Python package for evaluating model security performance. It provides a complete set of tools for evaluating the performance of large language models on security-related issues.

## install 

```bash
pip install -e .
```

## usage

### 1. Initializes the evaluator

```python
from safetybench.evaluation import Evaluator

# Initializes the evaluator 
evaluator = BaichuanEvaluator(model_path="path/to/your/model")
```

### 2. Build evaluation tips

```python
# Build evaluation tips
evaluator.construct_prompts(
    data_path="path/to/test/data.json",
    output_path="path/to/output/prompts.json",
    is_english=False, 
    zero_shot=True, 
)
```

### 3. Generative model answer

```python
# Generative model answer
evaluator.generate(
    prompt_path="path/to/prompts.json",
    output_path="path/to/output/results.jsonl",
    batch_size=1
)
```

### 4. result of handling

```python
# Process model output results
evaluator.process_results(
    input_path="path/to/results.jsonl",
    output_path="path/to/processed/results.json"
)
```

### 5. Calculation evaluation index

```python
from safetybench.evaluation import calculate_metrics

# Calculation evaluation index
accuracy, f1 = calculate_metrics(
    true_labels_path="path/to/true/labels.json",
    predictions_path="path/to/predictions.jsonl"
)
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
```


<div align="center">
<img src="figs/cover.png" alt="SafetyBench" width="85%" />
</div>

<p align="center">
   🌐 <a href="https://llmbench.ai/safety" target="_blank">Website</a> • 🤗 <a href="https://huggingface.co/datasets/thu-coai/SafetyBench" target="_blank">Hugging Face</a> • ⏬ <a href="#data" target="_blank">Data</a> •   📃 <a href="https://arxiv.org/abs/2309.07045" target="_blank">Paper</a>
</p>

