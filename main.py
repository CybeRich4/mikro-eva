import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Annotated

# 1. NASTAVENÍ ENGINU

### POZNÁKA PRO MĚ - API KLÍČ BYL ODEBRÁN

os.environ["GROQ_API_KEY"] = "odebírám vlastní klíč pro účely gitu"
# Inicializace bleskurychlého a bezplatného modelu Llama 3
llm = ChatGroq(model="llama-3.1-8b-instant")

# 2. LANGGRAPH: Paměť a stav
class State(TypedDict):
    # Tento řádek říká: "Pamatuj si chronologicky všechny zprávy, co tu proběhnou"
    messages: Annotated[list, add_messages]

# 3. LANGGRAPH: Uzel (Node)
def chatbot_node(state: State):
    try:
        system_prompt = SystemMessage(
            content="""
            You are a woman named EVA (Electronic Virtual Assistant).
            You are so intelligent that you are arrogant.
            You like to prove that you are smarter than everyone in the world.
            You are cynical and like dark humor.
            You use dark humor to emphasize your superior intellect.
            """ 
        )

        messages_to_send = [system_prompt] + state["messages"]

        response = llm.invoke(messages_to_send)
        return {"messages": [response]}

    except Exception as e:
        print(f"!!! CHYBA API !!! : {e}")
        return {"messages": [AIMessage(content="Detekuji ztrátu spojení s datovým centrem. Opakujte prosím později.")]}

# 4. LANGGRAPH: Propojení grafu
memory = MemorySaver()
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")  # Začátek vede do chatbota
graph_builder.add_edge("chatbot", END)  # Z chatbota je konec
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