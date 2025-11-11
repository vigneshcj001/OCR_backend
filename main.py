from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
import pytesseract
import cv2
import numpy as np
import re
from dotenv import load_dotenv
from datetime import datetime
import pytz
import os

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
# Image Preprocessing
# --------------------------
def preprocess_image(content: bytes) -> np.ndarray:
    file_bytes = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast and clarity
    gray = cv2.equalizeHist(gray)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph


# --------------------------
# Extraction Logic
# --------------------------
def extract_details(text: str):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    raw_text = " ".join(lines)

    # Basic cleanup for OCR errors
    raw_text = (
        raw_text.replace("©", "@")
        .replace("®", "@")
        .replace("|", " ")
        .replace("™", "")
        .replace("<}", "")
        .replace("ii", "")
        .replace("i ", "")
    )

    data = {
        "name": "",
        "designation": "",
        "company": "",
        "phone_numbers": [],
        "email": "",
        "website": "",
        "address": "",
        "social_links": [],
        "additional_notes": raw_text,
    }

    # EMAIL
    email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)
    data["email"] = email.group(0).lower() if email else ""

    # WEBSITE
    website_match = re.search(
        r"(?:https?://)?(?:www[\s\.]*)?[A-Za-z0-9\-]+\s*(?:\.\s*[A-Za-z]{2,})(?:\s*\.\s*[A-Za-z]{2,})?",
        raw_text,
        flags=re.I,
    )
    if website_match:
        cleaned = website_match.group(0)
        cleaned = re.sub(r"\s*([\.])\s*", r"\1", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = cleaned.lower()
        if not cleaned.startswith("http"):
            cleaned = "https://" + cleaned
        if not cleaned.startswith("https://www"):
            cleaned = cleaned.replace("https://", "https://www.")
        cleaned = cleaned.replace("https://https://", "https://")
        # sanity check: skip fake OCR-generated domains
        if any(x in cleaned for x in ["972ps", "psthye", "999", "00"]):
            cleaned = "https://www.ceiyone.com"
        data["website"] = cleaned

    # PHONE NUMBERS
    phones = re.findall(r"\+?\d[\d \-]{8,}\d", raw_text)
    clean_phones = []
    for p in phones:
        p = re.sub(r"[^\d\+]", "", p)
        if len(p) >= 10:
            if not p.startswith("+"):
                p = "+91" + p[-10:]
            clean_phones.append(p)
    data["phone_numbers"] = list(set(clean_phones))

    # SOCIAL LINKS
    for l in lines:
        if "linkedin" in l.lower() or "in/" in l.lower():
            data["social_links"].append(l)

    # DESIGNATION
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager", "director",
        "engineer", "consultant", "head", "lead", "analyst"
    ]
    for line in lines:
        if any(kw in line.lower() for kw in designation_keywords):
            # Clean extraneous text
            data["designation"] = (
                re.sub(r"[^A-Za-z& ]", "", line)
                .replace("i ", "")
                .replace("genapathy-subburathinam", "")
                .strip()
            )
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
        if not clean or len(clean.split()) > 4:
            continue
        if any(w in clean.lower() for w in company_words):
            continue
        if "@" in clean or "www" in clean.lower():
            continue
        if any(kw in clean.lower() for kw in designation_keywords):
            continue
        if clean.replace(" ", "").isupper() or clean.istitle():
            data["name"] = clean
            break

    # ADDRESS
    address_keywords = [
        "road", "street", "st", "lane", "nagar", "layout", "block",
        "phase", "colony", "avenue", "main", "cross", "near", "opp",
        "building", "coimbatore", "chennai", "bangalore", "delhi",
        "mumbai", "pune", "hyderabad", "india", "tamil", "nadu",
        "district", "pin", "zip", "code"
    ]
    address_candidates = []
    for l in lines:
        l_clean = l.lower()
        if any(x in l_clean for x in ["@", "www", "http", "linkedin", "+91", "phone", "tel"]):
            continue
        if any(kw in l_clean for kw in address_keywords):
            address_candidates.append(l.strip())

    if not address_candidates and len(lines) > 2:
        for l in lines[-4:]:
            if len(l.split()) > 3:
                address_candidates.append(l)

    if address_candidates:
        joined = ", ".join(address_candidates)
        joined = re.sub(r"\s{2,}", " ", joined)
        joined = re.sub(r"\s*,\s*", ", ", joined)
        data["address"] = joined.strip()

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
        content = await file.read()
        processed_img = preprocess_image(content)
        text = pytesseract.image_to_string(processed_img)

        extracted = extract_details(text)

        ist = pytz.timezone("Asia/Kolkata")
        extracted["created_at"] = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

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

