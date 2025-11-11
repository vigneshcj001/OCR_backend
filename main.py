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
# OCR Extraction Logic (Only FIRST NAME)
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

    # -----------------------------------------
    # SOCIAL
    for l in lines:
        if "linkedin" in l.lower() or "in/" in l.lower():
            data["social_links"].append(l)

    # -----------------------------------------
    # DESIGNATION
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager",
        "director", "engineer", "consultant", "head", "lead"
    ]

    for line in lines:
        if any(kw in line.lower() for kw in designation_keywords):
            data["designation"] = re.sub(r"fm.*", "", line, flags=re.I).strip()
            break

    # -----------------------------------------
    # COMPANY
    for line in lines:
        if re.search(r"(pvt|private|ltd|llp|inc|corporation|company|works)", line, re.I):
            data["company"] = line
            break

    # -----------------------------------------
    # ✅✅ ONLY FIRST NAME EXTRACTION
    # -----------------------------------------
    company_words = []
    if data["company"]:
        company_words = data["company"].lower().split()

    for l in lines:
        clean = re.sub(r"[^A-Za-z ]", "", l).strip()
        if not clean:
            continue

        # skip company
        if any(w in clean.lower() for w in company_words):
            continue

        # skip email/phone/website/designation lines
        if "@" in clean or "www" in clean.lower():
            continue
        if any(kw in clean.lower() for kw in designation_keywords):
            continue

        # must be alphabet-heavy
        alpha_ratio = len(re.findall(r"[A-Za-z]", clean)) / max(1, len(clean))
        if alpha_ratio < 0.7:
            continue

        # must be uppercase (typical name style)
        if clean.replace(" ", "").isupper():
            # ✅ pick ONLY the FIRST NAME
            data["name"] = clean.split()[0]
            break

    # fallback above designation
    if not data["name"]:
        for line in lines:
            if data["designation"] and line == data["designation"]:
                idx = lines.index(line)
                if idx > 0:
                    fallback = re.sub(r"[^A-Za-z ]", "", lines[idx - 1]).strip()
                    data["name"] = fallback.split()[0]  # first word only
                break

    # -----------------------------------------
    # ADDRESS
    address_lines = []
    for l in lines:
        if re.search(r"\d.*(street|st|road|rd|nagar|lane|city|coimbatore|tamil|india|641)", l, re.I):
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
