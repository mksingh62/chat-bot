from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import get_response_portfolio, get_response_codearenas

app = FastAPI()

# Allow JS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myportfolio-s5g7.onrender.com","https://codearenas.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str


@app.post("/chat-portfolio")
async def chat_portfolio_endpoint(req: ChatRequest):
    response = get_response_portfolio(req.message)
    return {"response": response}
@app.post("/chat-codearenas")
async def chat_codearenas_endpoint(req: ChatRequest):
    response = get_response_codearenas(req.message)
    return {"response": response}

