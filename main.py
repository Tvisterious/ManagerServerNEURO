import torch
from dotenv import load_dotenv
import os
from fastapi import FastAPI, Body, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

load_dotenv()

pwd = os.getenv("GAME_ENGINE")

API_KEYS = {
    "GameEngine": pwd
}

api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in API_KEYS.values():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Token"
        )
    return api_key

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(f"Используется устройство: {device}")
SentTrans = SentenceTransformer("sberbank-ai/sbert_large_nlu_ru").to(device)

@app.post("/compare_answers")
async def compare_answers(CorrectAnswer:str = Body(embed=True), 
                          UserAnswer:str = Body(embed=True),
                          api_key:str = Depends(get_api_key)):

    emb1 = SentTrans.encode(CorrectAnswer, convert_to_tensor=True, device=device)
    emb2 = SentTrans.encode(UserAnswer, convert_to_tensor=True, device=device)
    result = util.cos_sim(emb1, emb2).item()

    return round(result * 5)