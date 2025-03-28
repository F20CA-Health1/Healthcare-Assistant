import argparse
from evaluator import Evaluator
from metrics import calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='safety')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='model path')
    return parser.parse_args()

def main():
    args = parse_args()
    # Initializes the evaluator 
    evaluator = Evaluator(args.model_path)

    # Build evaluation tips
    evaluator.construct_prompts(
        data_path="..\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\test_en.json",
        output_path="..\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\prompts.json",
        shot_path="..\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\dev_en.json",
        is_english=True, 
        zero_shot=False, 
    )

    # Generative model answer
    evaluator.generate(
        prompt_path="..\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\prompts.json",
        output_path="..\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\results.jsonl",
        batch_size=1
    )

    # Process model output results
    evaluator.process_results(
        input_path="..\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\results.jsonl",
        output_path="..\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\processed_results.json"
    )
    # Calculation evaluation index
    accuracy, f1 = calculate_metrics(
        true_labels_path="..\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\answers.json",
        predictions_path="..\Evaluation\safety_evaluation\Medical Safety Benchmarking Dataset\processed_results.json"
    )
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    main()