#BEFORE START DOWNLOAD FASTAPI, CTRANFORMERS, SENTANCE_TRANSFORMERS
from fastapi import FastAPI, Body
from sentence_transformers import SentenceTransformer, util
#from ctransformers import AutoModelForCausalLM

"""llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", 
                                           model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", 
                                           model_type="mistral", 
                                           gpu_layers=25, 
                                           threads=10, 
                                           batch_size=1,
                                           stream=False)"""

SentTrans = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

"""@app.post("/ask_question")
async def ask_question(prompt:str = Body(embed=True), 
                 temperature: float = Body(embed=True),
                 max_tokens: int = Body(embed=True)):
    new_prompt = "[INST]"+prompt+"[/INST]"
    return(llm(new_prompt, temperature=temperature, max_new_tokens=max_tokens))"""

@app.post("/compare_answers")
async def compare_answers(CorrectAnswer:str = Body(embed=True), UserAnswer:str = Body(embed=True)):
    emb1 = SentTrans.encode(CorrectAnswer, convert_to_tensor=True)
    emb2 = SentTrans.encode(UserAnswer, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    return round(similarity * 5)