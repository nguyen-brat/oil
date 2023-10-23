from llm import LlmInference
from retrieval.tfidf import DocIR
import googletrans
import streamlit as st

class Inference:
    def __init__(
            self,
            url,
            header = {"Content-Type": "application/json"},
            data_path='oil_crawl/*/*',
            output_path='retrieval/saved',
            reset=False
    ):
        self.llm = LlmInference(url=url, header=header)
        self.doc_retrieval = DocIR(data_path=data_path, output_path=output_path,reset=reset)
        #self.translator = googletrans.Translator()

    def __call__(self, question):
        informations = self.doc_retrieval(query=question, k=10)
        #joint_information = '\n'.join(informations)
        answer = self.llm(question=question, contexts=informations)
        #answer = self.translator.translate(answer ,src='en' ,dest='vi').text
        return answer, informations[:3]