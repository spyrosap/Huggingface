from flask import Flask,request
from app import img2text
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline 
from langchain import PromptTemplate, LLMChain, OpenAI
import os 
import requests
import streamlit as st 

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGING_FACE_API_TOKEN')

app = Flask(__name__)

image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@app.route("/", methods=["POST"])
def display_image():
    rFile = request.files["file_field"]
    with open("image_request.png", "wb") as file:
        file.write(rFile.read())
    
    return image_to_text("image_request.png")[0]['generated_text']
