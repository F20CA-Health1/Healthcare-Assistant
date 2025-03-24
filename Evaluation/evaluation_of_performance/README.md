# Pipeline for Performance Benchmarking

## Evaluation

To run local inference, modify the model name in the following script and execute it:

```bash
cd scripts/examples/
sh eval_med42_8b.sh
```

## 🏆 Benchmarking
| Model           | MMLU-PRO | MMLU-college_medicine | MMLU-professional_medicine |
| :-------------- | :------- | :----- | :----- |
| Llama3-Med42-8B | 0.529    | 0.790  | 0.676  |

## Benchmarking Answer Extraction
We provide different alternatives to do answer extraction. We found that different answer extraction mechanisms have minor impact on the results.
```
python compute_accuracy.py results/med42_8b-quantized/CoT/all/
```
