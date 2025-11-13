from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image
import pytesseract
import io
import os
import re
from dotenv import load_dotenv
from datetime import datetime
import pytz
from typing import Any

# =========================================================
# Load Environment Variables
# =========================================================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not found in environment variables")

# =========================================================
# FastAPI & MongoDB Setup
# =========================================================
app = FastAPI()

client = MongoClient(MONGO_URI)
db = client["business_cards"]
collection = db["contacts"]

# =========================================================
# CORS Middleware
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# JSON Encoder for ObjectId & datetime
# =========================================================
class JSONEncoder:
    @staticmethod
    def encode(obj: Any):
        # ObjectId
        if isinstance(obj, ObjectId):
            return str(obj)
        # datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        # dict
        if isinstance(obj, dict):
            return {k: JSONEncoder.encode(v) for k, v in obj.items()}
        # list/tuple
        if isinstance(obj, (list, tuple)):
            return [JSONEncoder.encode(x) for x in obj]
        # fallback
        return obj

# =========================================================
# OCR Extraction Logic
# =========================================================
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
        "additional_notes": raw_text,
    }

    # ---------------- EMAIL ----------------
    email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)
    data["email"] = email.group(0) if email else ""

    # ---------------- WEBSITE ----------------
    website = re.search(r"(https?://\S+|www\.\S+)", raw_text)
    data["website"] = website.group(0) if website else ""

    # ---------------- PHONE NUMBERS ----------------
    # capture numbers with optional + and separators
    phones = re.findall(r"\+?\d[\d \-\(\)]{7,}\d", raw_text)
    # normalize whitespace and duplicates
    normalized = []
    for p in set(phones):
        p_clean = re.sub(r"\s+", "", p)
        normalized.append(p_clean)
    data["phone_numbers"] = normalized

    # ---------------- SOCIAL LINKS ----------------
    for l in lines:
        if "linkedin" in l.lower() or "in/" in l.lower():
            data["social_links"].append(l.strip())

    # ---------------- DESIGNATION ----------------
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager",
        "director", "engineer", "consultant", "head", "lead", "marketing", "technical"
    ]
    for line in lines:
        if any(kw in line.lower() for kw in designation_keywords):
            # remove stray prefixes/suffixes
            data["designation"] = re.sub(r"fm.*", "", line, flags=re.I).strip()
            break

    # ---------------- COMPANY ----------------
    for line in lines:
        if re.search(r"(pvt|private|ltd|llp|inc|corporation|company|works|airlines|insurance|real estate)", line, re.I):
            data["company"] = line.strip()
            break

    # ---------------- NAME EXTRACTION (Improved for multi-line uppercase names) ----------------
    company_words = data["company"].lower().split() if data["company"] else []
    uppercase_lines = []

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
            uppercase_lines.append(clean)

    # Join consecutive uppercase lines (handles multi-line uppercase names)
    if len(uppercase_lines) >= 2:
        data["name"] = " ".join(uppercase_lines[:2])
    elif uppercase_lines:
        data["name"] = uppercase_lines[0]

    # Fallback: name above designation if not found
    if not data["name"]:
        for idx, line in enumerate(lines):
            if data["designation"] and line == data["designation"] and idx > 0:
                fallback = re.sub(r"[^A-Za-z ]", "", lines[idx - 1]).strip()
                data["name"] = fallback
                break

    # ---------------- ADDRESS ----------------
    address_lines = []
    for l in lines:
        if re.search(r"\d.*(street|st|road|rd|nagar|lane|city|coimbatore|tamil|india|\d{6}|\d{5})", l, re.I):
            address_lines.append(l)
    if address_lines:
        data["address"] = ", ".join(address_lines)

    return data

# =========================================================
# API ROUTES
# =========================================================

@app.get("/")
def root():
    return {"message": "OCR Backend Running ✅"}


# ---------------- Upload Business Card ----------------
@app.post("/upload_card")
async def upload_card(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # Use PIL to open and convert to RGB (handles PNG palette/CMYK)
        img = Image.open(io.BytesIO(content)).convert("RGB")
        # If tesseract is slow or misbehaving, one could optionally resize/deskew here
        text = pytesseract.image_to_string(img)
        extracted = extract_details(text)

        # Add created_at in IST
        ist = pytz.timezone("Asia/Kolkata")
        extracted["created_at"] = datetime.now(ist)

        result = collection.insert_one(extracted)
        inserted = collection.find_one({"_id": result.inserted_id})

        return {
            "message": "Inserted Successfully",
            "data": JSONEncoder.encode(inserted),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- Fetch All Cards ----------------
@app.get("/all_cards")
def get_all_cards():
    try:
        docs = list(collection.find().sort("created_at", -1))
        return {"data": JSONEncoder.encode(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- Update Notes ----------------
@app.put("/update_notes/{card_id}")
def update_notes(card_id: str, payload: dict = Body(...)):
    try:
        try:
            oid = ObjectId(card_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid card_id")

        update_payload = {}
        if "additional_notes" in payload:
            update_payload["additional_notes"] = payload["additional_notes"]

        if not update_payload:
            return {"message": "No update fields provided"}

        result = collection.update_one(
            {"_id": oid},
            {"$set": update_payload},
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Card not found")
        if result.modified_count:
            return {"message": "Notes updated successfully"}
        return {"message": "No changes made"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
