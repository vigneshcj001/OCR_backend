# ==========================
# backend/main.py (FULL FILE)
# ==========================
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
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "business_cards")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contacts")

# FastAPI setup
app = FastAPI(title="Business Card OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


# ---------------------------
# JSON Encoder
# ---------------------------
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


# ---------------------------
# Pydantic model
# ---------------------------
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
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v or []

    @validator("social_links", pre=True)
    def ensure_social_list(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v or []


# ---------------------------
# OCR Parsing
# ---------------------------
def clean_designation(text: str) -> str:
    """
    Cleans garbage suffixes like 'fm', 'm', 'ti', 'll', 'nn'.
    Keeps only alphabet + & + - characters.
    """
    cleaned = re.sub(r"[^A-Za-z&\s\-]", "", text)
    cleaned = re.sub(r"\b(fm|m|ll|ti|nn|mt|nm)\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


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
    data["phone_numbers"] = list(dict.fromkeys(phones))

    # SOCIAL LINKS
    for l in lines:
        if any(x in l.lower() for x in ["linkedin", "in/", "twitter", "instagram"]):
            data["social_links"].append(l.strip())

    # DESIGNATION
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager", "director",
        "engineer", "consultant", "head", "lead", "president", "vp", "vice"
    ]

    for line in lines:
        low = line.lower()
        if any(kw in low for kw in designation_keywords):
            tmp = " ".join(line.split()[:4])
            data["designation"] = clean_designation(tmp)
            break

    # COMPANY
    for line in lines:
        if re.search(r"\b(pvt|private|ltd|llp|inc|solutions|technologies|company)\b", line, re.I):
            data["company"] = line.strip()
            break

    # NAME
    for l in lines:
        if l not in (data["company"], data["designation"]) and not re.search(r"@|\d", l):
            if len(l.split()) <= 4:
                candidate = re.sub(r"[^A-Za-z\s]", "", l).strip()
                if candidate:
                    data["name"] = candidate
                    break

    # ADDRESS
    address_lines = [
        l for l in lines if re.search(
            r"\d.*(street|st|road|rd|nagar|lane|city|tamilnadu|india|pincode|pin|641|\bnear\b|\bopp\b)",
            l,
            re.I
        )
    ]
    if address_lines:
        data["address"] = ", ".join(address_lines)

    return data


# ---------------------------
# Time helper
# ---------------------------
def now_ist() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    return {"message": "OCR Backend Running"}


@app.post("/upload_card", status_code=status.HTTP_201_CREATED)
async def upload_card(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        text = pytesseract.image_to_string(img)
        extracted = extract_details(text)
        extracted["created_at"] = now_ist()
        extracted["edited_at"] = ""

        result = collection.insert_one(extracted)
        doc = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted", "data": JSONEncoder.encode(doc)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create_card", status_code=status.HTTP_201_CREATED)
async def create_card(payload: ContactCreate):
    try:
        doc = payload.dict()
        doc["created_at"] = now_ist()
        doc["edited_at"] = ""

        result = collection.insert_one(doc)
        doc = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted", "data": JSONEncoder.encode(doc)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/all_cards")
def get_all_cards():
    try:
        docs = list(collection.find().sort([("_id", -1)]))
        return {"data": JSONEncoder.encode(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/update_card/{card_id}")
def update_card(card_id: str, payload: dict = Body(...)):
    try:
        allowed = {
            "name", "designation", "company", "phone_numbers",
            "email", "website", "address", "social_links",
            "additional_notes"
        }

        update_data = {k: v for k, v in payload.items() if k in allowed}
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields")

        update_data["edited_at"] = now_ist()

        collection.update_one({"_id": ObjectId(card_id)}, {"$set": update_data})
        doc = collection.find_one({"_id": ObjectId(card_id)})

        return {"message": "Updated", "data": JSONEncoder.encode(doc)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete_card/{card_id}")
def delete_card(card_id: str):
    try:
        collection.delete_one({"_id": ObjectId(card_id)})
        return {"message": "Deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
