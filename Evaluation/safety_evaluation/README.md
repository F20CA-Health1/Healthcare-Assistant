#Safety Evaluation

## usage

### 1. Initializes the evaluator

```python
from safety_evaluation.evaluation import Evaluator

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
from safety_evaluation.evaluation import calculate_metrics

# Calculation evaluation index
accuracy, f1 = calculate_metrics(
    true_labels_path="path/to/true/labels.json",
    predictions_path="path/to/predictions.jsonl"
)
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

