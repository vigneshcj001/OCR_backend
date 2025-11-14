import os
import io
import re
from datetime import datetime
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, File, UploadFile, Body, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import pytz

# Load .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "business_cards")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contacts")

# FastAPI setup
app = FastAPI(title="Business Card OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to allowed origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# -----------------------------------------
# JSON Encoder helper
# -----------------------------------------
class JSONEncoder:
    @staticmethod
    def encode(doc: Any):
        if isinstance(doc, ObjectId):
            return str(doc)
        if isinstance(doc, dict):
            return {k: JSONEncoder.encode(v) for k, v in doc.items()}
        if isinstance(doc, list):
            return [JSONEncoder.encode(x) for x in doc]
        return doc

# -----------------------------------------
# Pydantic models
# -----------------------------------------
class ContactCreate(BaseModel):
    name: Optional[str] = ""
    designation: Optional[str] = ""
    company: Optional[str] = ""
    phone_numbers: Optional[List[str]] = []
    email: Optional[EmailStr] = ""
    website: Optional[str] = ""
    address: Optional[str] = ""
    social_links: Optional[List[str]] = []
    additional_notes: Optional[str] = ""

    @validator("phone_numbers", pre=True)
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            items = [x.strip() for x in v.split(",") if x.strip()]
            return items
        return v

    @validator("social_links", pre=True)
    def ensure_social_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            items = [x.strip() for x in v.split(",") if x.strip()]
            return items
        return v

# -----------------------------------------
# Utilities: extract_details (OCR parsing)
# -----------------------------------------
def extract_details(text: str) -> Dict[str, Any]:
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
    email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)
    data["email"] = email.group(0) if email else ""

    # WEBSITE
    website = re.search(r"(https?://\S+|www\.\S+)", raw_text)
    data["website"] = website.group(0) if website else ""

    # PHONE NUMBERS
    phones = re.findall(r"\+?\d[\d \-\(\)]{6,}\d", raw_text)
    phones = [re.sub(r"[^\d\+]", "", p) for p in phones]
    data["phone_numbers"] = list(dict.fromkeys(phones))  # dedupe

    # SOCIAL LINKS
    for l in lines:
        if "linkedin" in l.lower() or "in/" in l.lower() or "twitter" in l.lower() or "instagram" in l.lower():
            data["social_links"].append(l.strip())

    # -----------------------------------------
    # IMPROVED DESIGNATION LOGIC
    # -----------------------------------------
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager",
        "director", "engineer", "consultant", "head", "lead",
        "president", "vp", "vice"
    ]

    for line in lines:
        low = line.lower()

        if any(kw in low for kw in designation_keywords):

            # Extract first few words (designation usually small)
            words = line.split()
            limited = " ".join(words[:4])  # keep only first 4 words

            # Clean extra chars
            clean = re.sub(r"[^A-Za-z&\s\-]", "", limited).strip()

            data["designation"] = clean
            break

    # COMPANY heuristics
    for line in lines:
        if re.search(r"\b(pvt\b|private|ltd|llp|inc\b|corporation|company|works|solutions|technologies)\b", line, re.I):
            data["company"] = line.strip()
            break

    # NAME heuristics
    uppercase_lines = []
    for l in lines:
        clean = re.sub(r"[^A-Za-z\s]", "", l).strip()
        if clean and clean.replace(" ", "").isupper() and len(clean.split()) <= 4:
            uppercase_lines.append(clean)
    if uppercase_lines:
        data["name"] = uppercase_lines[0]
    else:
        for l in lines:
            if l == data["company"] or l == data["designation"]:
                continue
            if re.search(r"[\w\.-]+@[\w\.-]+", l):
                continue
            if re.search(r"\+?\d", l):
                continue
            if 1 <= len(l.split()) <= 4 and len(l) < 60:
                data["name"] = re.sub(r"[^A-Za-z\s]", "", l).strip()
                break

    # ADDRESS heuristics
    address_lines = []
    for l in lines:
        if re.search(r"\d.*(street|st|road|rd|nagar|lane|city|tamilnadu|india|pincode|pin|641|\bnear\b|\bopp\b)", l, re.I):
            address_lines.append(l)
    if address_lines:
        data["address"] = ", ".join(address_lines)

    # Trim
    for k in ["name", "designation", "company", "address", "email", "website"]:
        if isinstance(data.get(k), str):
            data[k] = data[k].strip()

    return data

# -----------------------------------------
# Helper timestamp
# -----------------------------------------
def now_ist() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------------------
# Routes
# -----------------------------------------
@app.get("/")
def root():
    return {"message": "OCR Backend Running ✅"}

@app.post("/upload_card", status_code=status.HTTP_201_CREATED)
async def upload_card(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(img)
        extracted = extract_details(text)

        extracted["created_at"] = now_ist()
        extracted["edited_at"] = ""

        result = collection.insert_one(extracted)
        inserted = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_card", status_code=status.HTTP_201_CREATED)
async def create_card(payload: ContactCreate):
    try:
        doc = payload.dict()
        doc["created_at"] = now_ist()
        doc["edited_at"] = ""

        # If email exists, update that doc instead of creating a duplicate
        if doc.get("email"):
            existing = collection.find_one({"email": doc["email"]})
            if existing:
                # merge/overwrite fields and set edited_at
                doc["edited_at"] = now_ist()
                collection.update_one({"_id": existing["_id"]}, {"$set": doc})
                updated = collection.find_one({"_id": existing["_id"]})
                return {"message": "Updated existing contact", "data": JSONEncoder.encode(updated)}

        result = collection.insert_one(doc)
        inserted = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/all_cards")
def get_all_cards():
    try:
        docs = list(collection.find().sort([("_id", -1)]))
        return {"data": JSONEncoder.encode(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/update_notes/{card_id}")
def update_notes(card_id: str, payload: dict = Body(...)):
    try:
        ts = now_ist()
        update_payload = {
            "additional_notes": payload.get("additional_notes", ""),
            "edited_at": ts
        }
        collection.update_one({"_id": ObjectId(card_id)}, {"$set": update_payload})
        updated = collection.find_one({"_id": ObjectId(card_id)})
        return {"message": "Updated", "data": JSONEncoder.encode(updated)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
            raise HTTPException(status_code=400, detail="No valid fields to update.")

        update_data["edited_at"] = now_ist()
        collection.update_one({"_id": ObjectId(card_id)}, {"$set": update_data})
        updated = collection.find_one({"_id": ObjectId(card_id)})
        return {"message": "Updated", "data": JSONEncoder.encode(updated)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_card/{card_id}", status_code=status.HTTP_200_OK)
def delete_card(card_id: str):
    try:
        result = collection.delete_one({"_id": ObjectId(card_id)})
        if result.deleted_count == 1:
            return {"message": "Deleted"}
        else:
            raise HTTPException(status_code=404, detail="Card not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
