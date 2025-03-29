import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../RAG')))
from RAGComponent import RAGModule

if __name__ == "__main__":
    rag = RAGModule('./data/')
    df = pd.read_json('./RAG/nhs_illnesses_structured.json', orient='index')
    for _, row in df.iterrows():
        page = row['whole_page']
        title = row['title']
        rag.add_to_datastore(page, title)
        print(f'Added: {title}')
    print('DONE')
    