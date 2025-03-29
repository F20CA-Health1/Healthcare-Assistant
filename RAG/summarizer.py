import ollama
import pandas as pd 

def prompt_model(prompt: str) -> str:
    '''
    This function takes the refined prompt from the RAG module and sends it to the LLM module.
    returns a string that is the final answer to be sent to the saftey agent.
    '''
    response = ollama.chat(model='gemma3' , messages=[
        {
            'role': 'system',
            'content': 'Summarize the given document. The aim is to extract a brief description, symptomps, treatments, and causes \n The summary will be used to create a knowledge base for the NHS. \n The summary should be in the following format:  \n - Description: \n - Symptoms: \n - Treatments: \n - Causes: \n\n',
        },
        {
            'role': 'user',
            'content': prompt,
        },
            ])
    return response['message']['content']

data = pd.read_json('RAG/nhs_illnesses_structured.json', orient='index')
data['summary'] = ''
for i in range(len(data)):
    prompt = data.iloc[i]['whole_page']
    data.at[i, 'summary'] = prompt_model(prompt)
    print(f'Processed {i+1} out of {len(data)}')
    
data.to_json('RAG/nhs_illnesses_structured_with_summary.json', orient='index') 