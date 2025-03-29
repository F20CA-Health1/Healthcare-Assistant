# -*- coding: utf-8 -*-
from NLU_module.source.interaction_instructions import *
from NLU_module.source.agent_personas import *
from NLU_module.source.parse_utils import *
import os


class Verifier():
    def __init__(self, model):
        self.init_prompt = verifier_persona
        self.model = model

        self.model_name = os.environ['GPT_MODEL_NAME']

    def prompt_verifier(self, prompt):
        messages = [
            {"role": "system", "content": self.init_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0
        )

        message = response.choices[0].message.content

        return message

    def assess_cur_response(self, current_response):
        prompt = check_cur_response(current_response) + yaml_correct_answer
        response = self.prompt_verifier(prompt)
        return parse_correct_answer(response)