import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../RAG')))
from RAGComponent import RAGModule

dataset_path = "./RAG/nhs_illnesses_structured_with_summary.json"

if __name__ == "__main__":
    rag = RAGModule('./data/')
    df = pd.read_json(dataset_path, orient='index')
    for _, row in df.iterrows():
        page = row['summary']
        page += '\nLINK: ' + row['link']
        title = row['title']
        rag.add_to_datastore(page, title)
        print(f'Added: {title}')
    print('DONE')
    