from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openocr import OpenOCR
from PIL import Image
from datetime import datetime
import io
import re
import json
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# --------------------------
# Load ENV variables
# --------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# --------------------------
# Initialize App
# --------------------------
app = FastAPI(title="Business Card OCR API (OpenOCR)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Initialize OpenOCR
# --------------------------
print("Loading OpenOCR model...")
engine = OpenOCR(backend="torch", device="cpu")  # use backend='onnx' for lightweight mode
print("✅ OpenOCR model loaded successfully.")

# --------------------------
# MongoDB Setup (optional)
# --------------------------
client = None
db = None
if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI)
        db = client["ocr_db"]
        print("✅ MongoDB connected.")
    except Exception as e:
        print("⚠️ MongoDB connection failed:", e)


# --------------------------
# Helper: Extract business card fields
# --------------------------
def extract_details(text: str):
    details = {
        "name": "",
        "designation": "",
        "company": "",
        "phone_numbers": [],
        "email": "",
        "website": "",
        "address": "",
        "social_links": []
    }

    # Regex patterns
    phone_pattern = re.compile(r'(\+?\d[\d\s\-]{7,}\d)')
    email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    website_pattern = re.compile(r'(https?://\S+|www\.\S+)')
    social_pattern = re.compile(r'(linkedin\.com/\S+|facebook\.com/\S+|instagram\.com/\S+|twitter\.com/\S+)')

    # Extract using regex
    details["phone_numbers"] = phone_pattern.findall(text)
    details["email"] = next(iter(email_pattern.findall(text)), "")
    details["website"] = next(iter(website_pattern.findall(text)), "")
    details["social_links"] = social_pattern.findall(text)

    # Split lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return details

    # Guess name (usually first line)
    details["name"] = lines[0] if len(lines[0].split()) <= 4 else ""

    # Guess designation (common roles)
    if len(lines) >= 2:
        if any(word in lines[1].lower() for word in ["ceo", "founder", "manager", "director", "lead", "officer", "head"]):
            details["designation"] = lines[1]

    # Guess company (third line, if not email/website)
    for line in lines[1:]:
        if "@" not in line and not re.search(r'https?://|www\.', line) and len(line.split()) > 1:
            if not any(x in line.lower() for x in ["ceo", "manager", "director", "founder"]):
                details["company"] = line
                break

    # Address (look for indicators)
    for line in reversed(lines):
        if any(x in line.lower() for x in ["road", "st", "street", "nagar", "lane", "city", "block", "avenue", "india"]):
            details["address"] = line
            break

    return details


# --------------------------
# OCR API Endpoint
# --------------------------
@app.post("/ocr/business-card")
async def extract_business_card(file: UploadFile = File(...)):
    try:
        # Load image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Run OCR
        result_text, elapsed = engine(image)
        print(f"OCR done in {elapsed:.2f}s")

        # Extract structured info
        data = extract_details(result_text)

        # Optional: save to MongoDB
        if db:
            db.cards.insert_one({
                "filename": file.filename,
                "text": result_text,
                "data": data,
                "created_at": datetime.utcnow()
            })

        return {
            "status": "success",
            "elapsed_time": f"{elapsed:.2f}s",
            "extracted_data": data
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# --------------------------
# Root
# --------------------------
@app.get("/")
def home():
    return {"message": "Welcome to OpenOCR Business Card API 🚀"}


# --------------------------
# Run: uvicorn server:app --reload
# --------------------------
