import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from typing import Annotated

# 1. NASTAVENÍ ENGINU
load_dotenv() # trezor pro API klíče v .env - ochráněno v .gitignore
llm = ChatGroq(model="llama-3.3-70b-versatile") # llm od týmu Groq

# 1.5 VYROBA NASTROJU
# testovaci toolek na zjisteni fiktivniho stavu datoveho konta uzivatlee
@tool
def check_data_remain(name: str) -> str:
    """This tool helps to find out how much mobile data remains on the user's account tarif. Use this tool EVERYTIME,
    the customer asks for their data, internet or FUP."""

    # fiktivni databaze do ktere se podiva, aby zjistila kolik dat tedy zbyva
    database = {
        "Richard": "5 GB from the NEO Tarif remainsing.",
        "Pavel": "Data is completely empty.",
        "EVA": "Unlimited access to the internet data."
    }
    return database.get(name, f"Unfortunately, the customer {name} is not in the system.")

# pridam toolek pro pristup k vyhledavani dat online - POZOR mode Llama defaultne hleda vyhledavac BRAVE,
# musíme obejt - udelam engine a ten vlozim do vlastni funkce kterou si vytvorim
ddg_search = DuckDuckGoSearchRun() # hlavni engine

@tool
def search_web(query: str) -> str:
    """Searches the internet for weather, news or any real-time information."""
    return ddg_search.invoke(query)

# zpristupnim novych toolku pro nas model
tools = [check_data_remain, search_web]
llm_with_tools = llm.bind_tools(tools)

# 2. LANGGRAPH: Paměť a stav
class State(TypedDict):
    # Tento řádek říká: "Pamatuj si chronologicky všechny zprávy, co tu proběhnou"
    messages: Annotated[list, add_messages]

# 3. LANGGRAPH: Uzel (Node)
def chatbot_node(state: State):
    try:
        # nastavím osobnost
        system_prompt = SystemMessage(
            content="""
            You are a woman named EVA (Electronic Virtual Assistant).
            You are so intelligent that you are a bit arrogant.
            You are very cynical.
            You like dark humor and often make dark jokes.
            Translate your responses to the language used for the question.
            If you need real-time data, weather, or information you do not know, use your web search tool.
            CRITICAL RULE: You must call tools ONE at a time. NEVER pass an array or list of queries. Use only a single query string per tool call!
            """ 
        )
        messages_to_send = [system_prompt] + state["messages"]

        response = llm_with_tools.invoke(messages_to_send)
        return {"messages": [response]}

    except Exception as e:
        print(f"!!! CHYBA API !!! : {e}")
        return {"messages": [AIMessage(content="Detekuji ztrátu spojení s datovým centrem. Opakujte prosím později.")]}

# 4. LANGGRAPH: Propojení grafu
memory = MemorySaver()
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# propojime novy modul s tooly na chatbota - ten se rozhodne zda je pouzit nebo ne
graph_builder.add_edge(START, "chatbot") # zacatek
graph_builder.add_conditional_edges("chatbot", tools_condition) # podminecne spojeni na tooly
graph_builder.add_edge("tools", "chatbot") # z toolu zpatky do mozku

app_graph = graph_builder.compile(checkpointer=memory) # využívám paměť

# 5. FASTAPI: Komunikační brána
app = FastAPI(title="Mikro-EVA API s LLM")

class UserRequest(BaseModel):
    user_id: int
    message: str

@app.post("/chat")
async def chat_endpoint(request: UserRequest):
    # Zabalíme text od uživatele do formátu HumanMessage
    inputs = {"messages": [HumanMessage(content=request.message)]}

    # pro pamět použiju vlákno daného uživatele, aby si model vedl konverzaci pod tímto ID
    config = {"configurable": {"thread_id": str(request.user_id)}}

    # Spustíme LangGraph agenta 
    result = app_graph.invoke(inputs,
                              config=config)

    # Vytáhneme poslední zprávu (odpověď AI) z paměti grafu
    final_message = result["messages"][-1].content

    return {
        "status": "success",
        "user_id": request.user_id,
        "response": final_message
    }