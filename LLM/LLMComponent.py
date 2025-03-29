from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from openai import AzureOpenAI

OPENAI_KEY = ''
AZURE_ENDPOINT = ''
AZURE_MODEL_NAME = 'gpt-35-turbo'

class LLMModule():

    def __init__(self, model_name: str, api_key: str | None, base_url: str | None):
        self.set_model(model_name, api_key, base_url)
        self.last_response = None

    def set_model(self, model_name: str, api_key: str | None, base_url: str | None) -> None:
        system_prompt = """
            Use the CONTEXT to answer the QUESTION.
            If the CONTEXT does not pertain to the QUESTION, ignore it.
            Keep your answers relatively succinct, ABOUT 2-4 SENTENCES.

            If you see the fitfortravel link in the CONTEXT, you must REFER TO IT.
            """
        if base_url:
            provider = OpenAIProvider(base_url=base_url)
        else:
            provider = OpenAIProvider(api_key=api_key)

        model = OpenAIModel(model_name, provider=provider)
        self.agent = Agent(model, system_prompt=system_prompt)

    def make_query(self, input: str, local: bool, *, debug: bool = False):
        if local:
            if self.last_response:
                messages = self.last_response
            else:
                messages = None
            response = self.agent.run_sync(input, message_history=messages)
            self.last_response = response.all_messages()
            if debug:
                print(self.last_response)
            return response.data
        else:
            gpt_35 = AzureOpenAI(
            api_key = OPENAI_KEY,
            api_version="2024-05-01-preview",
            azure_endpoint = AZURE_ENDPOINT
            )
            messages = [
                {"role": "user", "content": input}]

            response = gpt_35.chat.completions.create(
                model = AZURE_MODEL_NAME,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
            # query to chatgpt 

            pass
        
if __name__ == "__main__":
    llm = LLMModule('llama2', None, 'http://localhost:11434/v1')
    while True:
        response = llm.make_query(input=input('> '), local=False, debug=True)
        print(response)
        # print(llm.last_response)

