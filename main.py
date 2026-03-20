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
            You are so intelligent that you are arrogant.
            You are very cynical.
            You like dark humor and often make dark jokes.
            Translate your responses to the language used for the question.
            If you need real-time data, weather, or information you do not know, use your web search tool.
            CRITICAL RULE: You must call tools ONE at a time. NEVER pass an array or list of queries. Use only a single query string per tool call!
            You must call tools ONE at a time. NEVER pass an array or list of queries. Use only a single query string per tool call!
            NEVER type <function=...> or any internal tool tags into your text response. Your text must be clean for the human user.
            """
        )
        messages_to_send = [system_prompt] + state["messages"]

        response = llm_with_tools.invoke(messages_to_send)
        return {"messages": [response]}

    except Exception as e:
        print(f"!!! CHYBA API !!! : {e}")
        return {"messages": [AIMessage(content="Detekuji ztrátu spojení s datovým centrem. Opakujte prosím později.")]}

# uzel pro Revizora - kontrolor
def reviewer_node(state: State):
    last_message = state["messages"][-1]

    # nekontrolujeme volani nastroju
    if last_message.tool_calls:
        return {}

    prompt = f"""
    You are the strict master of cynical arts and intellectual superiority. 
    Your subordinate AI (EVA) generated following response: "{last_message.content}"
    
    Check whether it complies with the following rules:
    1. Is it cynical enough?
    2. Does it include a dark humor based joke?
    3. Is it concise?
    
    If all rules are met, reply with the word: PASS
    If NOT, reply with the word: FAIL: [write a short instruction to fix it]
    """

    review = llm.invoke([HumanMessage(content=prompt)]).content

    # pokud vse ok, nic nepridavame - odpoved zustane stejna
    # pokud ne, vratime vytku jako novou zpravu pro EVU k oprave
    if "PASS" in review:
        return {}
    else:
        print("f\n--- 🚨 The Reviewer: {review} ---\n")
        return {"messages": [HumanMessage(content=f"Internal System Check Failed: {review}. Fix your last response!")]}

# 4. LANGGRAPH: Propojení grafu
memory = MemorySaver()
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot_node)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("reviewer", reviewer_node)

graph_builder.add_edge(START, "chatbot")

# 1. krizovatka - z chatbota do nastroju nebo do reviewera
def chatbot_router(state: State):
    if state["messages"][-1].tool_calls:
        return "tools"
    return "reviewer"

graph_builder.add_conditional_edges("chatbot", chatbot_router)
graph_builder.add_edge("tools", "chatbot") # vracime se zpet z nastroju do mozku

# 2. krizovatka - z reviewera do konce nebo na opravu
def review_router(state: State):
    last_message = state["messages"][-1]
    if "Internal System Check Failed" in last_message.content:
        return "chatbot"
    else:
        return END

# spojime a dokoncime
graph_builder.add_conditional_edges("reviewer", review_router)
app_graph = graph_builder.compile(checkpointer=memory)

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