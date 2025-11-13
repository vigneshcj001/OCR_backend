from fastapi import FastAPI, File, UploadFile, Body, HTTPException, status
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

# =========================================================
# Load Environment Variables
# =========================================================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

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
# JSON Encoder
# =========================================================
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

# =========================================================
# OCR Extraction Logic (Improved)
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

    # EMAIL
    email = re.search(r"[\w\.-]+@[\w\.-]+", raw_text)
    data["email"] = email.group(0) if email else ""

    # WEBSITE
    website = re.search(r"(https?://\S+|www\.\S+)", raw_text)
    data["website"] = website.group(0) if website else ""

    # PHONE NUMBERS
    phones = re.findall(r"\+?\d[\d \-]{8,}\d", raw_text)
    data["phone_numbers"] = list(set(phones))

    # SOCIAL LINKS
    for l in lines:
        if "linkedin" in l.lower() or "in/" in l.lower():
            data["social_links"].append(l)

    # DESIGNATION — clean and remove OCR noise
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager",
        "director", "engineer", "consultant", "head", "lead"
    ]
    for line in lines:
        if any(kw in line.lower() for kw in designation_keywords):
            clean_line = re.split(r"(\s+fm\s+|\s+from\s+)", line, flags=re.I)[0]
            clean_line = re.sub(r"[^A-Za-z& ]+", "", clean_line).strip()
            data["designation"] = clean_line
            break

    # COMPANY
    for line in lines:
        if re.search(r"(pvt|private|ltd|llp|inc|corporation|company|works)", line, re.I):
            data["company"] = line.strip()
            break

    # NAME — usually uppercase and short
    uppercase_lines = []
    for l in lines:
        clean = re.sub(r"[^A-Za-z ]", "", l).strip()
        if clean and clean.replace(" ", "").isupper():
            uppercase_lines.append(clean)

    if len(uppercase_lines) >= 2:
        data["name"] = " ".join(uppercase_lines[:2])
    elif uppercase_lines:
        data["name"] = uppercase_lines[0]

    # ADDRESS
    address_lines = []
    for l in lines:
        if re.search(r"\d.*(street|st|road|rd|nagar|lane|city|tamilnadu|india|641)", l, re.I):
            address_lines.append(l)
    if address_lines:
        data["address"] = ", ".join(address_lines)

    # Final Cleanups
    data["name"] = data["name"].strip()
    data["designation"] = data["designation"].strip()
    data["company"] = data["company"].strip()
    data["address"] = data["address"].strip()

    return data

# =========================================================
# API ROUTES
# =========================================================

@app.get("/")
def root():
    return {"message": "OCR Backend Running ✅"}


# Upload Business Card
@app.post("/upload_card")
async def upload_card(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))

        # Run OCR
        text = pytesseract.image_to_string(img)
        extracted = extract_details(text)

        # Add timestamps
        ist = pytz.timezone("Asia/Kolkata")
        extracted["created_at"] = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
        extracted["edited_at"] = ""

        # Save to MongoDB
        result = collection.insert_one(extracted)
        inserted = collection.find_one({"_id": result.inserted_id})

        return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}

    except Exception as e:
        return {"error": str(e)}


# Fetch All Cards
@app.get("/all_cards")
def get_all_cards():
    try:
        docs = list(collection.find())
        return {"data": JSONEncoder.encode(docs)}
    except Exception as e:
        return {"error": str(e)}


# Update Notes
@app.put("/update_notes/{card_id}")
def update_notes(card_id: str, payload: dict = Body(...)):
    try:
        ist = pytz.timezone("Asia/Kolkata")
        ts = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

        update_payload = {
            "additional_notes": payload.get("additional_notes", ""),
            "edited_at": ts
        }

        collection.update_one({"_id": ObjectId(card_id)}, {"$set": update_payload})
        updated = collection.find_one({"_id": ObjectId(card_id)})

        return {"message": "Updated", "data": JSONEncoder.encode(updated)}

    except Exception as e:
        return {"error": str(e)}


# PATCH Update Route (inline edits)
@app.patch("/update_card/{card_id}")
def update_card(card_id: str, payload: dict = Body(...)):
    try:
        allowed_fields = {
            "name", "designation", "company", "phone_numbers",
            "email", "website", "address", "social_links",
            "additional_notes"
        }

        update_data = {k: v for k, v in payload.items() if k in allowed_fields}

        if not update_data:
            raise HTTPException(400, "No valid fields to update.")

        ist = pytz.timezone("Asia/Kolkata")
        update_data["edited_at"] = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

        collection.update_one({"_id": ObjectId(card_id)}, {"$set": update_data})
        updated = collection.find_one({"_id": ObjectId(card_id)})

        return {"message": "Updated", "data": JSONEncoder.encode(updated)}

    except Exception as e:
        raise HTTPException(500, str(e))
