import torch
import torch.nn as nn
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel

app = FastAPI()
router = APIRouter()

class LogicGateModel(nn.Module):
    def __init__(self):
        super(LogicGateModel, self).__init__()
        self.linear1 = nn.Linear(2, 4)  # 2개의 입력 (x1, x2), 4개의 중간 노드
        self.linear2 = nn.Linear(4, 1)  # 4개의 중간 노드, 1개의 출력 (예측값)
        self.sigmoid = nn.Sigmoid()  # 활성화 함수 (sigmoid)

    def forward(self, x):
        x = torch.relu(self.linear1(x))  # ReLU 활성화 함수 적용
        x = self.sigmoid(self.linear2(x))  # Sigmoid 활성화 함수 적용
        return x


