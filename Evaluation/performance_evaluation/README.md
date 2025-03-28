# Pipeline for Performance Benchmarking

## Evaluation

To run local inference, modify the model name in eval.sh file and execute it:

```bash
cd pipline/scripts/examples/
sh eval.sh
```

## Results of Performance Evaluation
| Model           | MMLU-PRO | MMLU College Medicine | MMLU Professional Medicine |PubMedQA |
| :-------------- | :------- | :----- | :----- |:----- |
| Llama3-8B | 48.24 | 72.89 | 62.89  | 61.64 |
| Llama3-8B-RAG | 50.36 |  74.78 |  64.79  |  62.04  |
| Llama3-Med42-8B | 52.9 |  79.10 |  67.6 |  62.84  |

## Benchmarking

Get the benchmarking results:

```
python compute_accuracy.py results/med42_8b-quantized/CoT/all/
```
