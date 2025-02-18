import json
import pdfplumber
import openai
import chromadb


class ExtractorEm():


    def __init__(self,pdf_path,json_path):

        self.pdf_path = pdf_path
        self.json_path = json_path
        self.client = chromadb.PersistentClient(path="db/")
        self.collection = self.client.get_or_create_collection(name="pdf_chunk")  # Mantém a coleção ativa


    def chunk_text(self,text, chunk_size=500):
        """
        FUNÇÃO QUE SEPARA O TEXTO EM CHUNKS
        ENTRADA: texto e tamanho das chunks
        SAÍDA: lista com as chunks
        """
        words = text.split()
        chunks = []
        chunk = []
        length = 0

        for word in words:
            if length + len(word) + 1 > chunk_size:
                chunks.append(" ".join(chunk))
                chunk = []
                length = 0
            chunk.append(word)
            length += len(word) + 1

        if chunk:
            chunks.append(" ".join(chunk))

        return chunks


    def extract_text(self, skip = 0):
        """
        Extrai texto do PDF, separa em chunks numerados e filtra páginas iniciais.
        ENTRADA: Caminho para o pdf
        SAÍDA: lista de chunks numerados 
        """
        chunks_list = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    page_chunks = self.chunk_text(text)

                    #enumeração de chunks e páginas
                    for i, chunk in enumerate(page_chunks):
                        if page_num >= skip: #filtragem de páginas
                            chunks_list.append({
                                "page": page_num + 1,
                                "chunk_id": i + 1,
                                "text": chunk
                            })
        return chunks_list


    def jsonSaver(self):
        """
        Salva os chunks em um arquivo json no diretório padrão
        """
        chunks_list = self.extract_text()
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(chunks_list, f, indent=4, ensure_ascii=False)
    

    def generate_embedding(self,text):
        """
        Transforma as palavras em vetores
        ENTRADA: texto
        SAÍDA: lista de vetores
        """
        response = openai.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return embedding
    

    def chunks_embedding(self):
        """
        Gera embeddings para os chunks do JSON e os armazena no ChromaDB.
        """

        #carrega o arquivo em json e define uma variável para ele
        with open(self.json_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)


        # Iterar sobre os chunks e adicionar ao ChromaDB
        for chunk in chunks_data:
            text = chunk["text"]
            metadata = {"page": chunk["page"], "chunk_id": chunk["chunk_id"]}
            chunk_id = f"{metadata['page']}_{metadata['chunk_id']}"


            existing = self.collection.get(ids=[chunk_id])
            
            # Verifica se existing não é None e se a chave "ids" contém algum valor
            if existing and existing.get("ids"):
                print(f"Chunk {chunk_id} já está na base, ignorando...")
                continue


            #transforma o texto dos chunks em vetores
            embedding = self.generate_embedding(text)


            #adiciona os vetores na coleção
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[f"{metadata['page']}_{metadata['chunk_id']}"]
                )
            print(f"Chunck {chunk_id} adicionado!")


    def query_collection(self,query: str, top_k: int = 3) -> dict:
        """
        Consulta a coleção ChromaDB com a query fornecida e retorna os top_k resultados mais relevantes.

        Args:
            query (str): A query a ser usada para buscar na coleção.
            top_k (int, optional): O número de resultados a serem retornados. Defaults to 3.

        Returns:
            dict: Um dicionário contendo os resultados da consulta, incluindo os documentos e metadados correspondentes.
        """
        query_embedding = self.generate_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        if not results or not results.get("documents"):
            return {"message": "Nenhum resultado encontrado."}

        return results

