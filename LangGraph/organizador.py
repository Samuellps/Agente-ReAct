#Códigoq eu utiliza llm para organiar chunks
import openai
import extractor
import json
import os
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
json_path = "/home/samuel/Agente-ReAct/LangGraph/json/farmax.json"
pdf = extractor.extract_text("/home/samuel/Agente-ReAct/LangGraph/arquivos/farmax.pdf")


# Criando um cliente da API OpenAI
client = openai.OpenAI(api_key=api_key)


def organizar_chunks(chunks):
    """
    Refinar os chunks de texto utilizando uma LLM para remover ruídos e recuperar contexto perdido.
    """
    new_chunks = []
    total_tokens = 0

    for i, chunk in enumerate(chunks):
        # Definir contexto para a LLM com os chunks adjacentes
        contexto = ""
        if i > 0:
            contexto += f"Chunk anterior:\n{chunks[i-1]['text']}\n\n"
        contexto += f"Chunk atual:\n{chunk['text']}\n\n"
        if i < len(chunks) - 1:
            contexto += f"Chunk seguinte:\n{chunks[i+1]['text']}\n"

        # Prompt ajustado para preservar o conteúdo original
        prompt = f"""
        Você está processando textos para um sistema de Recuperação Aumentada por Geração (RAG).  
        Seu objetivo é garantir que cada chunk esteja completo e coerente, tanto em **estrutura gramatical** quanto em **conteúdo semântico**.

        ### **📌 Regras:**
        - **Não reescreva o texto ou altere seu significado.**
        - **Corrija palavras e frases cortadas, garantindo fluidez.**
        - **Garanta que cada chunk contenha apenas informações semanticamente coerentes.**
        - **Se um chunk mistura assuntos diferentes, separe-os em novos chunks distintos.**
        - **Use os chunks adjacentes para completar informações cortadas, caso necessário.**
        - **Mantenha a granularidade dos chunks adequada para uma recuperação eficiente.**

        ---

        ### **✅ Exemplo Certo (Correção de Segmentação e Coerência Semântica)**

        #### **Entrada (Texto com erro de segmentação e mistura de tópicos)**  
        Chunk 1: `"Os protetores solares ajudam a prevenir queimaduras e envelhecimento precoce. Além disso, os removedores de esmalte podem ser à base de acetona ou sem acetona."`  

        #### **Saída (Correção adequada para RAG)**  
        Chunk 1: `"Os protetores solares ajudam a prevenir queimaduras e envelhecimento precoce."`  
        Chunk 2: `"Os removedores de esmalte podem ser à base de acetona ou sem acetona."`  

        📌 **Correção válida**: Os tópicos foram **separados** para manter a coerência semântica.

        ---

        ### **❌ Exemplo Errado (Mistura de tópicos sem correção)**

        #### **Entrada (Texto original com tópicos misturados)**  
        Chunk: `"Nossa empresa vende protetores solares e removedores de esmalte que garantem um ótimo resultado."`  

        #### **Saída errada (Sem separação adequada)**  
        `"Nossa empresa vende produtos que garantem um ótimo resultado."`  

        🚨 **Erro**: O modelo **reescreveu de forma genérica**, removendo detalhes importantes e misturando conteúdos.

        ---

        Agora, corrija o chunk abaixo seguindo essas regras:

        {contexto}

        Retorne apenas um JSON no seguinte formato:
        {{
        "text": "Texto corrigido, mantendo coerência semântica e todo o conteúdo original."
        }}
        """
        # Chamada correta à API OpenAI
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente de organização de texto."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}  
        )

        tokens_usados = response.usage.total_tokens
        total_tokens += tokens_usados
        print(response.choices[0].message.content)

        # Captura da resposta
        try:
            resposta_json = response.choices[0].message.content.strip()
            resposta_dict = json.loads(resposta_json)  
            text_content = resposta_dict.get("text", "")

            # Se a resposta for uma lista, juntar os textos em uma única string
            if isinstance(text_content, list):
                new_text = " ".join(text_content).strip()
            else:
                new_text = text_content.strip()

            # Criando novo chunk refinado
            new_chunks.append({
                "page": chunk["page"],
                "chunk_id": len(new_chunks) + 1,
                "text": new_text
            })

        except json.JSONDecodeError as e:
            raise ValueError(f"Erro ao converter resposta para JSON: {resposta_json}") from e


    return new_chunks, total_tokens




# Valores em dólares para GPT-4 Turbo (ajuste conforme necessário)
preco_input_usd = 0.01 / 1000  # Preço por token de entrada
preco_output_usd = 0.03 / 1000  # Preço por token de saída
new_chunks, total_tokens = organizar_chunks(pdf)


os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(new_chunks, f, indent=4, ensure_ascii=False)


# Estimando o custo total (separando entrada e saída seria mais preciso)
custo_usd = total_tokens * (preco_input_usd + preco_output_usd)
print(f"Custo estimado em dólares: ${custo_usd:.4f}")


print(new_chunks)

