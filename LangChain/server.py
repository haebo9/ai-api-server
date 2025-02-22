from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sql_chat_model import TranslationModel  # 모델 가져오기

app = FastAPI() # 앱 인스턴스 설정
model = TranslationModel()  # 모델 인스턴스 생성

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat")
async def chat(message: str, session_id: str = "default_session"):
    response = await model.chat(message, session_id)
    return {"response": response}

@app.get("/translates")
async def translates(text: str, language: str = "en", session_id: str = "default_thread"):
    response = await model.translate(text, language, session_id)
    return {"response": response}