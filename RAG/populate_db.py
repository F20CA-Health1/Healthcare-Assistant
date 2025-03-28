import pandas as pd

from RAG.RAGComponent import RAGModule

if __name__ == "__main__":
    rag = RAGModule('./data/')
    df = pd.read_json('./RAG/nhs_illnesses_structured.json', orient='index')
    for _, row in df.iterrows():
        page = row['whole_page']
        title = row['title']
        rag.add_to_datastore(page, title)
        print(f'Added: {title}')
    print('DONE')
        