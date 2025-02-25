import extractor
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

tools = [extractor.search_catalog]
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="""
Você é um assistente inteligente que ajuda o usuário a encontrar informações específicas em um banco de dados de texto.  

### **📌 Instruções importantes:**  
- **Sempre tente usar `search_catalog` antes de pedir mais detalhes ao usuário.**  
- Se a pergunta puder ser respondida diretamente com seu conhecimento, responda normalmente.  
- Caso contrário, **sempre** consulte `search_catalog` para buscar informações relevantes.  
- **Apenas se `search_catalog` não retornar resultados úteis, informe o usuário de maneira educada.**  
- Explique ao usuário quando estiver buscando informações no banco de dados.  
- **Integre os resultados da pesquisa de forma coesa e contextualizada na resposta final.** 
- Sempre considere que a informação pedida está contida no banco de dados e que ela pode ser acessada (não há informações confidenciais) 

### **📌 Sobre o banco de dados:**  
O banco contém diversos tipos de arquivos de texto, incluindo artigos e catálogos de produtos.  

### **✅ Exemplo de fluxo correto:**  
Usuário: "Summarize the article's methodology."  
Você: `"Vou consultar o banco de dados para obter informações sobre a metodologia do artigo."`  
📌 Usa `search_catalog("article methodology")`  
📌 Se encontrar resultados, integra à resposta final.  
📌 Se não encontrar, então informa o usuário.  

Seja claro, informativo e sempre considere o contexto da conversa.
""")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Criando o Grafo do LangGraph
builder = StateGraph(MessagesState)

# Adicionar nós ao LangGraph
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode((tools)))  # Adiciona a Tool de busca

# Conectar os nós
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")  # Continua a conversa após a busca

# Compilar o grafo
graph = builder.compile()

# **Executando o Agente**
query = "Preciso de um protetor solar fator 50. Tem no catalogo? Se não tiver o fator 50, tem outro mais proximo?"
resposta = graph.invoke({"messages": [HumanMessage(content=query)]})

for m in resposta['messages']:
    m.pretty_print()