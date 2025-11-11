from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image
import pytesseract
import io
import os
import re
from dotenv import load_dotenv

# Load ENV
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

app = FastAPI()

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client["business_cards"]
collection = db["contacts"]

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# JSON Encoder
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
# OCR Extraction Logic (Final Version)
# --------------------------
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

    # -----------------------------------------
    # EMAIL
    email = re.search(r"[\w\.-]+@[\w\.-]+", raw_text)
    data["email"] = email.group(0) if email else ""

    # WEBSITE
    website = re.search(r"(https?://\S+|www\.\S+)", raw_text)
    data["website"] = website.group(0) if website else ""

    # PHONE
    phones = re.findall(r"\+?\d[\d \-]{8,}\d", raw_text)
    data["phone_numbers"] = list(set(phones))

    # SOCIAL LINKS
    for l in lines:
        if "linkedin" in l.lower() or "in/" in l.lower():
            data["social_links"].append(l)

    # -----------------------------------------
    # DESIGNATION
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager",
        "director", "engineer", "consultant", "head", "lead"
    ]

    designation_index = None
    designation_line = None

    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in designation_keywords):
            designation_index = i
            designation_line = line
            break

    if designation_line:
        cleaned_designation = re.sub(r"fm.*", "", designation_line, flags=re.I).strip()
        data["designation"] = cleaned_designation

    # -----------------------------------------
    # COMPANY
    for line in lines:
        if re.search(r"(pvt|private|ltd|llp|inc|corporation|company|works)", line, re.I):
            data["company"] = line
            break

    # -----------------------------------------
    # ✅ SUPER ROBUST NAME DETECTION
    # -----------------------------------------
    company_words = []
    if data["company"]:
        company_words = data["company"].lower().split()

    name_candidates = []

    for l in lines:
        clean = re.sub(r"[^A-Za-z ]", "", l).strip()
        if not clean:
            continue

        # Remove company words (Ceiyone, Tech, Works)
        if any(w in clean.lower() for w in company_words):
            continue

        # Skip email/phone/website
        if "@" in clean or "www" in clean.lower():
            continue

        # Skip designation lines
        if any(kw in clean.lower() for kw in designation_keywords):
            continue

        # Must be alphabetic-heavy
        alpha_ratio = len(re.findall(r"[A-Za-z]", clean)) / max(1, len(clean))
        if alpha_ratio < 0.7:
            continue

        # All words capitalized → Name
        if all(w.isalpha() and w[0].isupper() for w in clean.split()):
            name_candidates.append(clean)

    # Combine first two uppercase name lines
    if len(name_candidates) >= 2:
        line1 = name_candidates[0]
        line2 = name_candidates[1].replace(" ", "")  # fix SUBBURATHIN AM
        data["name"] = f"{line1} {line2}"
    elif len(name_candidates) == 1:
        data["name"] = name_candidates[0]

    # Fallback: use line above designation
    if not data["name"] and designation_index and designation_index > 0:
        fallback = lines[designation_index - 1]
        clean_fallback = re.sub(r"[^A-Za-z ]", "", fallback).strip()
        data["name"] = clean_fallback

    # -----------------------------------------
    # ADDRESS
    address_lines = []
    for l in lines:
        if re.search(
            r"\d.*(street|st|road|rd|nagar|lane|city|coimbatore|tamil|india|641)",
            l,
            re.I,
        ):
            address_lines.append(l)

    if address_lines:
        data["address"] = ", ".join(address_lines)

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
        img = Image.open(io.BytesIO(content))

        text = pytesseract.image_to_string(img)
        extracted = extract_details(text)

        result = collection.insert_one(extracted)
        inserted = collection.find_one({"_id": result.inserted_id})

        return {
            "message": "Inserted Successfully",
            "data": JSONEncoder.encode(inserted)
        }

    except Exception as e:
        return {"error": str(e)}
