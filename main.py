from fastapi import FastAPI, File, UploadFile
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PIL import Image
import pytesseract
import io
import re
from bson import ObjectId
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# ENV variables
MONGO_URI = os.getenv("MONGO_URI")
TESSERACT_PATH = os.getenv("TESSERACT_PATH")

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

app = FastAPI()

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client["business_cards"]
collection = db["contacts"]

# CORS Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_details(text: str):
    """Extract structured details from OCR text"""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    full_text = " ".join(lines)

    data = {
        "name": "",
        "designation": "",
        "company": "",
        "phone_numbers": [],
        "email": "",
        "website": "",
        "address": "",
        "social_links": [],
        "additional_notes": full_text
    }

    email_match = re.search(r"[\w\.-]+@[\w\.-]+", full_text)
    website_match = re.search(r"(https?://\S+|www\.\S+)", full_text)
    phones = re.findall(r"\+?\d[\d\- ]{7,}\d", full_text)

    data["email"] = email_match.group(0) if email_match else ""
    data["website"] = website_match.group(0) if website_match else ""
    data["phone_numbers"] = list(set(phones))

    # Detect designation → name is line above
    for i, line in enumerate(lines):
        if re.search(r"manager|director|engineer|founder|ceo|head|lead|consultant", line, re.I):
            data["designation"] = line
            if i > 0:
                data["name"] = lines[i - 1]
            break

    # Company detection
    company_candidates = [l for l in lines if re.search(r"Pvt|Ltd|Inc|Corporation|Company", l, re.I)]
    if company_candidates:
        data["company"] = company_candidates[0]

    # Address detection
    addr_candidates = [
        l for l in lines if re.search(r"\d.*(Street|St|Road|Ave|City|State|Avenue)", l, re.I)
    ]
    if addr_candidates:
        data["address"] = " ".join(addr_candidates)

    return data


class JSONEncoder:
    @staticmethod
    def encode_document(doc):
        if isinstance(doc, list):
            return [JSONEncoder.encode_document(x) for x in doc]
        if isinstance(doc, dict):
            return {k: JSONEncoder.encode_document(v) for k, v in doc.items()}
        if isinstance(doc, ObjectId):
            return str(doc)
        return doc

@app.get("/")
def root():
    return {"message": "OCR Backend Running Successfully 🚀"}
@app.post("/upload_card")
async def upload_card(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        text = pytesseract.image_to_string(image)
        data = extract_details(text)

        result = collection.insert_one(data)
        inserted_data = collection.find_one({"_id": result.inserted_id})

        return {
            "message": "Card inserted successfully",
            "data": JSONEncoder.encode_document(inserted_data)
        }

    except Exception as e:
        return {"error": str(e)}

