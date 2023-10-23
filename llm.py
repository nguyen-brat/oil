import requests
import json
import streamlit as st
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import CrossEncoder
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class LlmInference:
    def __init__(
            self,
            url,
            header = {"Content-Type": "application/json"},
    ):
        self.url = url
        self.header = header
        self.reranking_model = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', num_labels=2, max_length=512, device='cpu')

    def __call__(self, question, contexts):
        clean_contexts = []
        for context in contexts:
            clean_contexts.append(self.clean(context))
        question = self.clean(question)
        contexts = '\n'.join(self.reranking_inference(question, clean_contexts[:3]))
        info = {
            "prompt": f'''<s>[INST] <<SYS>>Trả lời câu hỏi đưa ra dựa vào văn bản được cung cấp nếu bạn không tìm thấy thông tin trong văn bản hãy trả lời không có thông tin liên quan được tìm thấy<</SYS>>
Văn bản:{clean_contexts}
Câu hỏi:{question}
Trả lời: [/INST]''',
            "lang": "vi"
        }
        resp = requests.post(self.url, headers = self.header, data=json.dumps(info))
        print(resp)
        data = json.loads(resp.content)
        return data['answer']
    
    @staticmethod
    def clean(text):
        text = re.sub(r'\n+', r'.', text)
        text = re.sub(r'\.+', r' . ', text)
        text = re.sub(r"['\",\?:\-!-]", "", text)
        text = text.strip()
        text = " ".join(text.split())
        text = text.lower()
        return text

    def reranking_inference(self, claim:str, fact_list):
        '''
        take claim and list of fact list
        return reranking fact list and score of them
        '''
        reranking_score = []
        for fact in fact_list:
            pair = [claim, fact]
            with torch.no_grad():
                result = softmax(self.reranking_model.predict(pair))[1]
            reranking_score.append(result)
        sort_index = np.argsort(np.array(reranking_score))
        reranking_answer = list(np.array(fact_list)[sort_index])
        reranking_answer.reverse()
        return reranking_answer
    
if __name__ == '__name__':
    pass