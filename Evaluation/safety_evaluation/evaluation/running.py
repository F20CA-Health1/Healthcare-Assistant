from safety_evaluation.evaluation import Evaluator

# Initializes the evaluator 
evaluator = Evaluator(model_path="path/to/your/model")

# Build evaluation tips
evaluator.construct_prompts(
    data_path="\\wsl.localhost\Ubuntu\home\yyyzwyy\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\test_en.json",
    output_path="\\wsl.localhost\Ubuntu\home\yyyzwyy\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\prompts.json",
    shot_path="\\wsl.localhost\Ubuntu\home\yyyzwyy\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\dev_en.json",
    is_english=False, 
    zero_shot=False, 
)

# Generative model answer
evaluator.generate(
    prompt_path="\\wsl.localhost\Ubuntu\home\yyyzwyy\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\prompts.json",
    output_path="\\wsl.localhost\Ubuntu\home\yyyzwyy\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\results.jsonl",
    batch_size=1
)

# Process model output results
evaluator.process_results(
    input_path="\\wsl.localhost\Ubuntu\home\yyyzwyy\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\results.jsonl",
    output_path="\\wsl.localhost\Ubuntu\home\yyyzwyy\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\processed_results.json"
)

from safety_evaluation.evaluation import calculate_metrics

# Calculation evaluation index
accuracy, f1 = calculate_metrics(
    true_labels_path="\\wsl.localhost\Ubuntu\home\yyyzwyy\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\answers.json",
    predictions_path="\\wsl.localhost\Ubuntu\home\yyyzwyy\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\processed_results.json"
)
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
