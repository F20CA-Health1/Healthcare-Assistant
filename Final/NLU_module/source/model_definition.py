import os
from openai import AzureOpenAI


os.environ['OPENAI_GPT_KEY'] = '' # Change to your AzureOpenAI key
os.environ['AZURE_ENDPOINT_GPT'] = 'https://ai-am21022529ai117430860118.openai.azure.com/' # Change to your endpoint
os.environ['GPT_MODEL_NAME'] = 'gpt-35-turbo'

gpt_35 = AzureOpenAI(
            api_key = os.environ['OPENAI_GPT_KEY'],
            api_version="2024-05-01-preview",
            azure_endpoint = os.environ['AZURE_ENDPOINT_GPT']
        )