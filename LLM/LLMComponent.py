from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

class LLMComponent():
    
    def __init__(self, model_name: str, api_key: str | None, base_url: str | None):
        self.set_model(model_name, api_key, base_url)
        self.last_response = None

    def set_model(self, model_name: str, api_key: str | None, base_url: str | None) -> None:
        system_prompt = """
            Use the CONTEXT to answer the QUESTION.
            If the CONTEXT does not pertain to the QUESTION, ignore it.
            Keep your answers relatively succinct.
            """
        if base_url:
            provider = OpenAIProvider(base_url=base_url)
        else:
            provider = OpenAIProvider(api_key=api_key)

        model = OpenAIModel(model_name, provider=provider)
        self.agent = Agent(model, system_prompt=system_prompt)

    def make_query(self, input: str):
        if self.agent:
            if self.last_response:
                messages = self.last_response
            else:
                messages = None
            response = self.agent.run_sync(input, message_history=messages)
            self.last_response = response.all_messages()
            return response.data
        
if __name__ == "__main__":
    llm = LLMComponent('qwen2.5:0.5b', None, 'http://localhost:11434/v1')
    while True:
        response = llm.make_query(input('> '))
        print(response)
        print(llm.last_response)

