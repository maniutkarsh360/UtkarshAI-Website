```
import torch
from transformers import LLaMAForConditionalGeneration, LLaMATokenizer
import numpy as np

api_key = "YOUR_META_AI_API_KEY"
model_name = "llama"

tokenizer = LLaMATokenizer.from_pretrained(model_name)
model = LLaMAForConditionalGeneration.from_pretrained(model_name)

def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

class ChatBot:
    def __init__(self):
        self.context = []

    def respond(self, input_text):
        self.context.append(input_text)
        response = generate_response(input_text)
        self.context.append(response)
        return response

chatbot = ChatBot()

while True:
    user_input = input("User: ")
    response = chatbot.respond(user_input)
    print("ChatBot: ", response)
```
