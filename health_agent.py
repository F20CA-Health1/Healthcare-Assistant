'''
What does this file due?
This file contains the HealthAgent class, which is responsible for connecting the whole project together.
User input is taken from the GUI and passed into this class.
Then the class uses the input to call the RAG module to retrieve the relevant information.
refined prompt from the RAG module is passed to the LLM modul to get the final answer.
The final answer is then passed to the saftey agent to check for any harmful content.
The final answer is then passed to the TTS to be spoken out loud.
final answer is passed to the GUI to be displayed to the user. 
'''
from pyexpat import model
from RAG.RAGComponent import RAGModule
import ollama

model_name = 'qwen:0.5b'

def add_context(prompt: str, rag: RAGModule) -> str:
    '''
    This function uses the RAG module to get the relevant information from the knowledge base.
    returns a string that is the refined prompt to be sent to the LLM module.
    '''
    refined_prompt = rag.prepare_prompt(prompt)
    return refined_prompt

def prompt_model(prompt: str) -> str:
    '''
    This function takes the refined prompt from the RAG module and sends it to the LLM module.
    returns a string that is the final answer to be sent to the saftey agent.
    '''
    response = ollama.chat(model=model_name , messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])
    return response['message']['content']

def check_saftey(response: str) -> str:
    '''
    This function takes the final answer from the LLM module and sends it to the saftey agent.
    returns a string that is the final answer to be sent to the TTS module.
    '''
    pass

def send_to_gui(response: str) -> None:
    '''
    This function the final answer from the saftey agent and sends it to the GUI module.
    ''' 
    # The GUI should have a function that take in a string of response
    # The GUI should have a function that takes a path to an audio file to play


def process_prompt(prompt: str):
    '''
    This function takes a string from the GUI to be sent to the RAG module.
    this is the function that sets everything off.
    '''
    refined_prompt = add_context(prompt)    # RAG (DONE)
    response = prompt_model(refined_prompt) # LLM
    safe_response = check_saftey(response)  # Saftey
    send_to_gui(safe_response)              # GUI + TTS
    pass

def test_rag():
    # for testing purposes
    rag = RAGModule('./data')
    prompt = "what are the symptoms of AAA"
    refined_prompt = add_context(prompt, rag)
    print(refined_prompt)
    model_response = prompt_model(refined_prompt)
    print(model_response)

def test_check_saftey():
    # for testing purposes
    response = "This is a test response"
    safe_response = check_saftey(response)
    print(safe_response)

if __name__ == "__main__":
    # test_rag()
    test_check_saftey()

