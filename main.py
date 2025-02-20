from typing import Union
from fastapi import FastAPI
import model

app = FastAPI()
model = model.AndModel() # default model_type

# 학습에 사용할 모델을 선택하는 함수 선언
def model_select(model_type):
    global model
    
    if model_type == "and":
        model = model.AndModel()
    elif model_type == "xor":
        model = model.XorModel()
    elif model_type == "not":
        model = model.NotModel()
    else: 
        raise ValueError("Invalid model type")
    return model
    
# endpoint 엔드포인트를 선언하며 GET으로 요청을 받고 경로는 /이다.
@app.get("/")
def read_root():
    return {"name": "heabo"}

# items/ 다음 경로로 드러오는 GET 요청을 처리
    # q : 쿼리 매개변수로 url에서 ?뒤에 오는 값을 의미 /items/123?q=search
    # Union[str, None]는 q의 문자열이 str혹은 None일것을 의미
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# 모델의 학습을 요청. 생성 기능은 POST
@app.post("/train/{model_type}")
def train(model_type : str): # 여기에 모델 종류를 선택하는 것을 추가
    model.train()
    return {"result": "OK"}

# 모델의 예측 기능을 호출. 조회 기능은 GET
@app.get("/predict/left/{left}/right/{right}") 
def predict(left: int, right: int):
    result = model.predict([left, right])
    return {"result": result}