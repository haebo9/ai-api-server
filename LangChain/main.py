import uvicorn
import server

app = server.app

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
# uvicorn main:app --reload
# 채팅 모델 실행
### $ MODEL_TYPE=chat uvicorn main:app --reload
# 번역 모델 실행
### $ MODEL_TYPE=translate uvicorn main:app --reload