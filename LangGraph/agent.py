import extractor
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode


#TRATAMENTO DE DADOS 

farmax_pdf = "/home/samuel/Documentos/farmax.pdf"
farmax_json = "/home/samuel/LangGraph/json/farmax_chunks.json"
dados = extractor.ExtractorEm(pdf_path= farmax_pdf, json_path= farmax_json)
dados.extract_text(skip= 2)
dados.jsonSaver()
dados.chunks_embedding()



#AGENTE

tools = [dados.query_collection]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# System message
sys_msg = SystemMessage(content="Você é um assistente de vendas, caso te perguntem sobre informações de produtos utilizar a tool: 'query_collection'")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

messages = [HumanMessage(content="Me indique um produto para hidratação capilar")]
messages = react_graph.invoke({"messages": messages})

for m in messages['messages']:
    m.pretty_print()