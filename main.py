from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
import pytesseract
import cv2
import numpy as np
import io
import os
import re
from dotenv import load_dotenv
from datetime import datetime
import pytz

# --------------------------
# Load ENV
# --------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# --------------------------
# FastAPI & MongoDB Setup
# --------------------------
app = FastAPI()

client = MongoClient(MONGO_URI)
db = client["business_cards"]
collection = db["contacts"]

# --------------------------
# CORS Middleware
# --------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# JSON Encoder for ObjectId
# --------------------------
class JSONEncoder:
    @staticmethod
    def encode(doc):
        if isinstance(doc, ObjectId):
            return str(doc)
        if isinstance(doc, dict):
            return {k: JSONEncoder.encode(v) for k, v in doc.items()}
        if isinstance(doc, list):
            return [JSONEncoder.encode(x) for x in doc]
        return doc

# --------------------------
# OCR Extraction Logic
# --------------------------
def preprocess_image(content: bytes) -> np.ndarray:
    """Enhanced preprocessing for better OCR on address text."""
    file_bytes = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    gray = cv2.equalizeHist(gray)

    # Adaptive threshold (better for faint text)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Morph closing to connect thin text
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph


def extract_details(text: str):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    raw_text = " ".join(lines)

    data = {
        "name": "",
        "designation": "",
        "company": "",
        "phone_numbers": [],
        "email": "",
        "website": "",
        "address": "",
        "social_links": [],
        "additional_notes": raw_text
    }

    # EMAIL
    email = re.search(r"[\w\.-]+@[\w\.-]+", raw_text)
    data["email"] = email.group(0) if email else ""

    # WEBSITE — fix OCR spacing like "WWW. ceiyone. com"
    website_match = re.search(
        r"(?:https?://)?(?:www[\s\.]*)?[A-Za-z0-9\-]+\s*(?:\.\s*[A-Za-z]{2,})(?:\s*\.\s*[A-Za-z]{2,})?",
        raw_text,
        flags=re.I
    )
    if website_match:
        cleaned = website_match.group(0)
        cleaned = re.sub(r"\s*([\.])\s*", r"\1", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)
        if not cleaned.lower().startswith("http"):
            cleaned = "https://" + cleaned.lower().replace("https://https://", "https://")
        data["website"] = cleaned

    # PHONE NUMBERS
    phones = re.findall(r"\+?\d[\d \-]{8,}\d", raw_text)
    data["phone_numbers"] = list(set(phones))

    # SOCIAL LINKS
    for l in lines:
        if "linkedin" in l.lower() or "in/" in l.lower():
            data["social_links"].append(l)

    # DESIGNATION
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager",
        "director", "engineer", "consultant", "head", "lead"
    ]
    for line in lines:
        if any(kw in line.lower() for kw in designation_keywords):
            data["designation"] = line.strip()
            break

    # COMPANY
    for line in lines:
        if re.search(r"(pvt|private|ltd|llp|inc|corporation|company|works)", line, re.I):
            data["company"] = line.strip()
            break

    # NAME
    company_words = data["company"].lower().split() if data["company"] else []
    for l in lines:
        clean = re.sub(r"[^A-Za-z ]", "", l).strip()
        if not clean:
            continue
        if any(w in clean.lower() for w in company_words):
            continue
        if "@" in clean or "www" in clean.lower():
            continue
        if any(kw in clean.lower() for kw in designation_keywords):
            continue
        alpha_ratio = len(re.findall(r"[A-Za-z]", clean)) / max(1, len(clean))
        if alpha_ratio < 0.7:
            continue
        if clean.replace(" ", "").isupper():
            data["name"] = clean
            break

    # ------------------------
    # ADDRESS (Enhanced Logic)
    # ------------------------
    address_keywords = [
        "road", "street", "st", "lane", "nagar", "layout", "block", "phase",
        "colony", "avenue", "main", "cross", "near", "opp", "building",
        "coimbatore", "chennai", "bangalore", "delhi", "mumbai", "pune",
        "hyderabad", "india", "tamil", "nadu", "district", "pin", "zip"
    ]

    address_candidates = []
    for l in lines:
        l_clean = l.lower()
        # skip lines that look like names, emails, or phones
        if any(x in l_clean for x in ["@", "www", "http", "linkedin", "+91", "phone", "tel"]):
            continue
        if any(kw in l_clean for kw in address_keywords):
            address_candidates.append(l.strip())

    # If address still not found, try lines near the end of the card
    if not address_candidates and len(lines) > 2:
        possible_bottom_lines = lines[-3:]
        for l in possible_bottom_lines:
            if len(l.split()) > 3:  # long lines = probable address
                address_candidates.append(l)

    if address_candidates:
        data["address"] = ", ".join(address_candidates)

    return data


# --------------------------
# API ROUTES
# --------------------------
@app.get("/")
def root():
    return {"message": "OCR Backend Running ✅"}


@app.post("/upload_card")
async def upload_card(file: UploadFile = File(...)):
    try:
        # Read and preprocess using OpenCV
        content = await file.read()
        processed_img = preprocess_image(content)

        # Run Tesseract OCR
        text = pytesseract.image_to_string(processed_img)

        extracted = extract_details(text)

        # Add created_at in IST
        ist = pytz.timezone("Asia/Kolkata")
        extracted["created_at"] = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

        # Insert to MongoDB
        result = collection.insert_one(extracted)
        inserted = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}

    except Exception as e:
        return {"error": str(e)}


@app.get("/all_cards")
def get_all_cards():
    try:
        docs = list(collection.find())
        return {"data": JSONEncoder.encode(docs)}
    except Exception as e:
        return {"error": str(e)}


@app.put("/update_notes/{card_id}")
def update_notes(card_id: str, payload: dict = Body(...)):
    try:
        new_notes = payload.get("additional_notes", "")
        result = collection.update_one(
            {"_id": ObjectId(card_id)},
            {"$set": {"additional_notes": new_notes}}
        )
        if result.modified_count:
            return {"message": "Notes updated successfully", "data": new_notes}
        return {"message": "No changes made", "data": new_notes}
    except Exception as e:
        return {"error": str(e)}



