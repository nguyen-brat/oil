import requests
import json
import streamlit as st
import re

class LlmInference:
    def __init__(
            self,
            url,
            header = {"Content-Type": "application/json"},
    ):
        self.url = url
        self.header = header

    def __call__(self, question, contexts):
        question = self.clean(question)
        contexts = self.clean(contexts)
        info = {
            "prompt": f'''<s>[INST] <<SYS>>
Trả lời câu hỏi đưa ra dựa vào văn bản được cung cấp nếu bạn không tìm thấy thông tin trong văn bản hãy trả lời không có thông tin liên quan được tìm thấy
<</SYS>>\n\n\
Văn bản:{contexts}
Câu hỏi:{question}
Trả lời: [/INST]''',
            "lang": "vi"
        }
        resp = requests.post(self.url, headers = self.header, data=json.dumps(info))
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
''' Yeu cau

Văn bản:
Câu hỏi:
Trả lời:

Văn bản:
Câu hỏi:
Trả lời: ''' 
if __name__ == '__name__':
    pass