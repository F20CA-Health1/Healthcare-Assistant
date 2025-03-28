from multiprocessing.spawn import prepare
from typing import Generator
from chromadb import PersistentClient, Collection

class RAGModule():
    def __init__(self, path: str):
        COLLECTION_NAME = 'nhs_illnesses'
        self.collection = self.initialise_db(path, COLLECTION_NAME)
        self.next_id = self.next_id_generator()

        self.user_template = """QUESTION:\n{}\nCONTEXT:\n{}\n"""

    def initialise_db(self, path: str, collection_name: str):
        """
        Either get or create collection with name `collection_name` in directory `path`.
        """
        chroma_client = PersistentClient(path)
        if collection_name in chroma_client.list_collections():
            print(f"Collection `{collection_name}` found!")
            return chroma_client.get_collection(collection_name)
        print(f"Created collection `{collection_name}`")
        return chroma_client.create_collection(collection_name)

    def next_id_generator(self) -> Generator[None, None, int]:
        count = 0
        while True:
            yield count
            count += 1

    def add_to_datastore(self, fact: str, illness_name: str):
        self.collection.add(documents=fact, ids=f'fact_{next(self.next_id)}', metadatas={'illness_name' : illness_name})

    def get_context(self, query: str, num_retrievals = 3) -> str:
        """
        Get relevant documents formatted into paragraph.
        """
        documents = self.collection.query(query_texts=query, n_results=num_retrievals).get("documents", [[]]) # Extract documents from our query
        return "\n".join(documents[0]) # Only return first batch results (single query)

    def prepare_prompt(self, user_input: str) -> str:
        context = self.get_context(user_input, 1)

        return self.user_template.format(user_input, context)

    def add_docs(self):
        while True:
            doc = input('Doc: ')
            if not doc:
                break
            self.add_to_datastore(doc, doc)

if __name__ == "__main__":
    RAG = RAGModule('./data')
    while True:
        user_input = input('> ')
        print(RAG.prepare_prompt(user_input))