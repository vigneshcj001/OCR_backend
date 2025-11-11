from fastapi import FastAPI, File, UploadFile, Body
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
# JSON Encoder for ObjectId
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
    email = re.search(r"[\w\.-]+@[\w\.-]+", raw_text)
    data["email"] = email.group(0) if email else ""

    # ---------------- WEBSITE ----------------
    website = re.search(r"(https?://\S+|www\.\S+)", raw_text)
    data["website"] = website.group(0) if website else ""

    # ---------------- PHONE NUMBERS ----------------
    phones = re.findall(r"\+?\d[\d \-]{8,}\d", raw_text)
    data["phone_numbers"] = list(set(phones))

    # ---------------- SOCIAL LINKS ----------------
    for l in lines:
        if "linkedin" in l.lower() or "in/" in l.lower():
            data["social_links"].append(l)

    # ---------------- DESIGNATION ----------------
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager",
        "director", "engineer", "consultant", "head", "lead"
    ]
    for line in lines:
        if any(kw in line.lower() for kw in designation_keywords):
            data["designation"] = re.sub(r"fm.*", "", line, flags=re.I).strip()
            break

    # ---------------- COMPANY ----------------
    for line in lines:
        if re.search(r"(pvt|private|ltd|llp|inc|corporation|company|works)", line, re.I):
            data["company"] = line
            break

    # ---------------- NAME EXTRACTION ----------------
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
            data["name"] = clean.split()[0]
            break

    # Fallback: name above designation if not found
    if not data["name"]:
        for idx, line in enumerate(lines):
            if data["designation"] and line == data["designation"] and idx > 0:
                fallback = re.sub(r"[^A-Za-z ]", "", lines[idx - 1]).strip()
                data["name"] = fallback.split()[0]
                break

    # ---------------- ADDRESS ----------------
    address_lines = []
    for l in lines:
        if re.search(r"\d.*(street|st|road|rd|nagar|lane|city|coimbatore|tamil|india|641)", l, re.I):
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
        img = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(img)
        extracted = extract_details(text)

        # Add created_at in IST
        ist = pytz.timezone("Asia/Kolkata")
        extracted["created_at"] = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

        result = collection.insert_one(extracted)
        inserted = collection.find_one({"_id": result.inserted_id})

        return {
            "message": "Inserted Successfully",
            "data": JSONEncoder.encode(inserted),
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------- Fetch All Cards ----------------
@app.get("/all_cards")
def get_all_cards():
    try:
        docs = list(collection.find())
        return {"data": JSONEncoder.encode(docs)}
    except Exception as e:
        return {"error": str(e)}


# ---------------- Update Notes ----------------
@app.put("/update_notes/{card_id}")
def update_notes(card_id: str, payload: dict = Body(...)):
    try:
        result = collection.update_one(
            {"_id": ObjectId(card_id)},
            {"$set": {"additional_notes": payload.get("additional_notes", "")}},
        )
        if result.modified_count:
            return {"message": "Notes updated successfully"}
        return {"message": "No changes made"}
    except Exception as e:
        return {"error": str(e)}
