# -*- coding: utf-8 -*-
from NLU_module.source.interaction_instructions import *
from NLU_module.source.agent_personas import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from NLU_module.source.model_definition import *


class Adviser():
    def __init__(self, model_name):
        self.init_prompt = adviser_persona
        self.name = model_name
        if model_name == 'gpt_35':
            self.model = gpt_35
            self.model_name = os.environ['GPT_MODEL_NAME']
        elif model_name == 'starling':
            self.tokenizer = AutoTokenizer.from_pretrained("Nexusflow/Starling-LM-7B-beta")
            self.model = AutoModelForCausalLM.from_pretrained("Nexusflow/Starling-LM-7B-beta")
        elif model_name == 'deepseek':
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat",
                                                            torch_dtype=torch.float16,
                                                            device_map={"": 0},
                                                            offload_folder="offload_weights"
                                                            )
        elif model_name == 'llama3':
            self.tokenizer = AutoTokenizer.from_pretrained("m42-health/Llama3-Med42-8B")
            self.model = AutoModelForCausalLM.from_pretrained("m42-health/Llama3-Med42-8B")

    def prompt_instructor(self, prompt):
        messages = [
            {"role": "system", "content": self.init_prompt},
            {"role": "user", "content": prompt}]

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content

    def generate_response(self, question=None, context=None, history=None, init=False, with_history=True, safe=True,  explanation=""):
        response = None
        if init:
            prompt = init_generate_answer(question, context)
        elif safe:
            if with_history:
                prompt = generate_answer(question, context, history)
            else:
                prompt = generate_answer_without(question, context)
        else:
            if with_history:
                prompt = make_adjustment(question, history, explanation)
            else:
                prompt = make_adjustment_without(question, explanation)
        max_length = 8000
        if self.name == 'gpt_35':
            response = self.prompt_instructor(prompt)
        elif self.name == 'starling':
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                    return_attention_mask=True)
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            final_prompt = "<|system|>\n{}\n<|user|>\n{}<|history|>\n{}\n<|assistant|>".format(self.init_prompt, question, history)
            inputs = self.tokenizer(final_prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            outputs = self.model.generate(**inputs, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = remove_repeated_input(response, final_prompt)
        return response

def remove_repeated_input(decoded_output, input_text):
    parts = decoded_output.split('<|assistant|>', 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return decoded_output
