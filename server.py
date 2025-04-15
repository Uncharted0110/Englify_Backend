from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import zipfile
import requests
import gdown

# === Model Download Config ===
model_folder = "model_finale"
model_zip = "mymod.zip"
gdrive_file_id = "1i_g__SfEPEUOV7htXnQj2FCBEd-px74y"



def download_model_from_gdrive():
    if os.path.exists(model_folder):
        print("✅ Model already exists.")
        return

    print("⬇️ Downloading model from Google Drive using gdown...")
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    gdown.download(url, model_zip, quiet=False)

    print("📦 Extracting model...")
    with zipfile.ZipFile(model_zip, 'r') as zip_ref:
        zip_ref.extractall()

    os.remove(model_zip)
    print("✅ Model is ready.")


# === Prepare Model ===
download_model_from_gdrive()
tokenizer = AutoTokenizer.from_pretrained(model_folder)
model = AutoModelForSeq2SeqLM.from_pretrained(model_folder)
model.eval()

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    output_tokens = model.generate(**inputs, max_length=512, num_return_sequences=1)
    translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return translated_text

@app.get("/")
async def root():
    return {"message": "Translation API is running!"}

@app.post("/generate")
async def generate(request: TranslationRequest):
    translated_text = translate_text(request.text)
    return {"translation": translated_text}
