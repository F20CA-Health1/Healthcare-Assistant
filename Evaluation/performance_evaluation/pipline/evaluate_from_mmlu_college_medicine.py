import csv
import json
import argparse
import os
import torch
import random
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset

from multiprocessing.spawn import prepare
from typing import Generator
from chromadb import PersistentClient, Collection

class RAGModule():
    def __init__(self, path: str):
        COLLECTION_NAME = 'nhs_illnesses'
        self.collection = self.initialise_db(path, COLLECTION_NAME)
        self.next_id = self.next_id_generator()

        self.user_template = """QUESTION:\n{}\nCONTEXT:\n{}\n"""

    def initialise_db(self, path: str, collection_name: str):
        """
        Either get or create collection with name `collection_name` in directory `path`.
        """
        chroma_client = PersistentClient(path)
        if collection_name in chroma_client.list_collections():
            print(f"Collection `{collection_name}` found!")
            return chroma_client.get_collection(collection_name)
        print(f"Created collection `{collection_name}`")
        return chroma_client.create_collection(collection_name)

    def next_id_generator(self) -> Generator[None, None, int]:
        count = 0
        while True:
            yield count
            count += 1

    def add_to_datastore(self, fact: str, illness_name: str):
        self.collection.add(documents=fact, ids=f'fact_{next(self.next_id)}', metadatas={'illness_name' : illness_name})

    def get_context(self, query: str, num_retrievals = 3) -> str:
        """
        Get relevant documents formatted into paragraph.
        """
        documents = self.collection.query(query_texts=query, n_results=num_retrievals).get("documents", [[]]) # Extract documents from our query
        return "\n".join(documents[0]) # Only return first batch results (single query)

    def prepare_prompt(self, user_input: str) -> str:
        context = self.get_context(user_input, 1)

        return self.user_template.format(user_input, context)

    def add_docs(self):
        while True:
            doc = input('Doc: ')
            if not doc:
                break
            self.add_to_datastore(doc, doc)

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 2048
rag_module = RAGModule('./data')

def load_mmlu():
    dataset = load_dataset("cais/mmlu", 'college_medicine')
    # dataset = load_dataset("cais/mmlu", 'professional_medicine')
    test_df = list(dataset["test"])
    val_df = list(dataset["validation"])
    return test_df, val_df


def load_model():
    llm = LLM(model=args.model, gpu_memory_utilization=float(args.gpu_util),
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=max_model_length,
                trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                        stop=["Question:"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return (llm, sampling_params), tokenizer


def preprocess(test_df):
    return test_df


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    return df


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = [
        example["choices"][0],
        example["choices"][1],
        example["choices"][2],
        example["choices"][3]
    ]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        answer = choices[example["answer"]]
        prompt += f"Answer: Let's think step by step.\nTherefore, the answer is {answer}.\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    prompt = rag_module.prepare_prompt(prompt)
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    
    val_examples = val_df[: k]
    for example in val_examples:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(llm, sampling_params, inference_batch):
    start = time.time()
    outputs = llm.generate(inference_batch, sampling_params)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, 3)
            if x == each["answer"]:
                corr += 1
            else:
                wrong += 1
        elif choices.index(each["pred"]) == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path):
    llm, sampling_params = model
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)

    pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong


def main():
    model, tokenizer = load_model()
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu()
    
    output_path = os.path.join(save_result_dir, "results.json")
    acc, corr_count, wrong_count = eval_cot("all", model, tokenizer, full_val_df, full_test_df, output_path)
    
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        f.write("Average accuracy: {:.4f}\n".format(acc))
    
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, acc]
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()


