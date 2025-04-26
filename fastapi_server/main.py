# simplechat/fastapi_server/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import boto3
import json
import os

from dotenv import load_dotenv

# 環境変数を読み込む（.envがある場合）
load_dotenv()

app = FastAPI()

# モデルID（デフォルトは Nova Lite）
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

# Bedrockクライアントを作成
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# リクエストBodyの定義
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversationHistory: Optional[List[ChatMessage]] = []

# /api/conversation エンドポイント
@app.post("/api/conversation")
def conversation(req: ChatRequest):
    messages = req.conversationHistory.copy()
    messages.append({"role": "user", "content": req.message})

    bedrock_messages = [
        {
            "role": msg["role"],
            "content": [{"text": msg["content"]}]
        } for msg in messages
    ]

    payload = {
        "messages": bedrock_messages,
        "inferenceConfig": {
            "maxTokens": 512,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 0.9
        }
    }

    # Bedrockへリクエスト送信
    response = bedrock_client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json"
    )

    response_body = json.loads(response['body'].read())
    assistant_response = response_body['output']['message']['content'][0]['text']
    messages.append({"role": "assistant", "content": assistant_response})

    return {
        "success": True,
        "response": assistant_response,
        "conversationHistory": messages
    }
