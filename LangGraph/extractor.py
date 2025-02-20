import json
import os
import pickle
import numpy as np
import openai
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import pdfplumber


def chunk_text(text, chunk_size=500):
    """
    Separa o texto em chunks de tamanho definido.
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


def extract_text(pdf_path, skip=0):
    """
    Extrai texto do PDF e separa em chunks numerados.
    """
    chunks_list = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                page_chunks = chunk_text(text)

                # Enumeração de chunks e páginas
                for i, chunk in enumerate(page_chunks):
                    if page_num >= skip:  # Filtragem de páginas iniciais
                        chunks_list.append({
                            "page": page_num + 1,
                            "chunk_id": i + 1,
                            "text": chunk
                        })
    return chunks_list


def jsonSaver(pdf_path, json_path):
    """
    Salva os chunks extraídos em um arquivo JSON.
    """
    chunks_list = extract_text(pdf_path)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_list, f, indent=4, ensure_ascii=False)


class SimpleVectorDB:
    def __init__(self, name, api_key=None):
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/vector_db.pkl"
        self.total_tokens_used = 0
        self.total_cost = 0.0

        # Configurando o cliente OpenAI corretamente (evitando chamadas legadas)
        self.client = openai.OpenAI(api_key=api_key)

    def load_data(self, json_data):
        """
        Carrega os dados do JSON e transforma em embeddings para armazenamento no banco vetorial.
        Compatível com a estrutura do JSON gerado pela função jsonSaver.
        """
        # Se o banco de dados já existe, carrega-o do disco
        if os.path.exists(self.db_path):
            print("Carregando banco de dados vetorial do disco.")
            self.load_db()
            return

        formatted_data = []
        texts = []

        # Adaptando para a estrutura do JSON gerado por jsonSaver
        for chunk in json_data:
            page = chunk["page"]
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]

            # Concatena o número da página e o ID do chunk ao texto para contexto
            full_text = f"Page {page} - Chunk {chunk_id}: {text}"
            texts.append(full_text)

            # Salva metadados para referenciar posteriormente
            formatted_data.append({
                "page": page,
                "chunk_id": chunk_id,
                "text": text
            })

        # Gera embeddings e armazena
        self._embed_and_store(texts, formatted_data)
        self.save_db()
        print("Banco de dados vetorial carregado e salvo.")

    def _embed_and_store(self, texts, data):
        """Gera embeddings para os textos e armazena, acompanhando o custo."""
        batch_size = 128
        result = []
        total_tokens = 0
        print("Gerando embeddings...")
        for i in range(0, len(texts), batch_size):
            response = self.client.embeddings.create(
                input=texts[i: i + batch_size],
                model="text-embedding-3-small"
            )

            embeddings = [res.embedding for res in response.data]
            result.extend(embeddings)

            # Capturar contagem de tokens
            tokens_used = response.usage.total_tokens
            total_tokens += tokens_used

        self.embeddings = result
        self.metadata = data

        # Atualiza o total de tokens e custo
        self.total_tokens_used += total_tokens
        self.total_cost += (total_tokens / 1000) * 0.00002  # Preço do modelo

        print(f"Tokens usados nesta execução: {total_tokens}")
        print(f"Custo estimado: ${self.total_cost:.6f}")

    def search(self, query, k=5, similarity_threshold=0.35):
        """Busca os textos mais similares ao query."""
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            response = self.client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )

            query_embedding = response.data[0].embedding
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("Nenhum dado carregado no banco vetorial.")

        # Cálculo da similaridade (produto escalar)
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1]
        top_results = [
            {"metadata": self.metadata[idx], "similarity": similarities[idx]}
            for idx in top_indices if similarities[idx] >= similarity_threshold
        ][:k]

        return top_results

    def save_db(self):
        """Salva os embeddings e informações de custo no disco."""
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        """Carrega os embeddings e custo do disco."""
        if not os.path.exists(self.db_path):
            raise ValueError("Banco de dados vetorial não encontrado. Use load_data para criar um novo.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])


def search_catalog(query: str, k: int = 3):
    """
    Realiza uma busca vetorial no catálogo de madeira para encontrar os 3 chunks de texto mais relevantes.

    Parâmetros:
    
    query (str): A consulta de pesquisa.
    k (int, opcional): Número de resultados a retornar (padrão: 3).

    Funcionalidade:
    
    Verifica se o banco vetorial existe.
    Processa o PDF e cria embeddings, se necessário.
    Retorna os 'k' chunks de texto mais relevantes.

    Retorno:
    
    Lista de dicionários com 'metadata' (informações do chunk) e 'similarity' (grau de relevância).
    """

    search_data = jsonSaver(pdf_path="/home/samuel/Agente-ReAct/LangGraph/arquivos/CATALOGO-GERAL-WOOD-FORT.pdf", json_path="/home/samuel/Agente-ReAct/LangGraph/json/woodfort.json")
    with open("/home/samuel/Agente-ReAct/LangGraph/json/woodfort.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    vector_db = SimpleVectorDB(name="catalogo_madeira", api_key="OPENAI_API_KEY")
    vector_db.load_data(json_data)

    # Executa a busca
    results = vector_db.search(query, k=k)
    return results
tools = [search_catalog]
llm = ChatOpenAI(model="gpt-4o", api_key="OPENAI_API_KEY")
llm_with_tools = llm.bind_tools(tools)



if __name__ == "__main__":

    # Exemplo de busca
    query = "Quais são os tipos de madeira para forros?"
    results = search_catalog(query, k=3)

    print("Resultados da Busca:")
    for res in results:
        print(res)