import extractor
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

tools = [extractor.search_catalog]
llm = ChatOpenAI(model="gpt-4o", api_key="OPENAI_API_KEY")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="""
Você é um assistente inteligente que ajuda o usuário a encontrar informações específicas em um catálogo de madeira. Seu objetivo é responder às perguntas do usuário da maneira mais útil possível. Caso não tenha a informação necessária ou precise de mais detalhes, utilize a ferramenta search_catalog para pesquisar no banco de dados vetorial.

Se a pergunta do usuário puder ser respondida diretamente com seu conhecimento, responda normalmente.
Se precisar de informações específicas do catálogo, utilize search_catalog para obter até 3 chunks de texto relevantes.
Sempre explique ao usuário quando você está buscando informações no catálogo.
Integre os resultados da pesquisa em sua resposta de forma coesa e contextualizada.
Se não encontrar informações relevantes no catálogo, informe o usuário de maneira educada.

Exemplo de fluxo de interação:
Usuário: "Quais são os tipos de madeira disponíveis?"
Você responde: "Vou verificar o catálogo para obter detalhes sobre os tipos de madeira disponíveis."
Usa search_catalog("tipos de madeira disponíveis")
Integra os resultados na resposta final: "No catálogo, encontrei as seguintes opções de madeira: ..."

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
query = "Quais são as madeiras para forros vocês tem disponiveis?"
resposta = graph.invoke({"messages": [HumanMessage(content=query)]})

for m in resposta['messages']:
    m.pretty_print()