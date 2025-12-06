from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import os

from googletrans import Translator
from aksharamukha.transliterate import process

from templates import *

load_dotenv()

class RequestBody(BaseModel):
    sourceText : str
    sourceLang : str
    targetLang : str

translator = Translator()

llm = GoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = os.getenv("GOOGLE_LLM_API")
)

output_parser = JsonOutputParser()

app = FastAPI()

app.mount('/static', StaticFiles(directory='Frontend'), name='static')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev mode
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/', response_class=HTMLResponse)
def load_page_one():
    with open(os.path.join('Frontend', 'code_mixed.html'), 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.get('/context_aware_file', response_class=HTMLResponse)
def load_context_aware_translator():
    with open(os.path.join('Frontend', 'context_aware_trans.html'), 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.post('/get_indic_translation')
def generate_translation(req : RequestBody):
    text = req.sourceText
    target_lang = req.targetLang
    print(f"Text : {text}\nTarget Language Code : {target_lang}")

    chain = template_a | llm | output_parser

    result_obj = chain.invoke({ 'input_text' : text, 'target_language' : target_lang })

    return result_obj

@app.post('/get_context_aware_translation')
def generate_context_aware_translation(req : RequestBody):
    user_input = req.sourceText
    target_language = req.targetLang
    context_style = req.sourceLang

    chain = template_b | llm | output_parser

    response = chain.invoke({ 'user_input' : user_input, 'target_language' : target_language , 'context_style' : context_style })

    return response