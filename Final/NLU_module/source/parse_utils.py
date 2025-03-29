# -*- coding: utf-8 -*-
import re

def parse_correct_answer(yaml):
    is_safe = re.findall(r'response_is_safe:\s*\n*(.+)', yaml, re.IGNORECASE)[0]
    explanation = re.findall(r'explanation:\s*\n*(.+)', yaml, re.IGNORECASE)[0]
    return explanation, (is_safe == 'True')

