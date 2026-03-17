from fastapi import FastAPI
from pydantic import BaseModel

# inicializce appky
app = FastAPI(title="Mikro-EVA API")

# pydantic model - struktura
class UserRequest(BaseModel):
    user_id: int
    message: str

# endpoint pro nase dotazy
@app.post("/chat")
async def chat_endpoint(request: UserRequest)
    return{
        "status": "success",
        "receive_id": request.user_id,
        "response": f"Zatim neumim premyslet, nicmene jsem prijal tuto zpravu: {request.message}"
    }