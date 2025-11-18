# main.py
# OpenAI-vision-based OCR backend + CRUD routes
# Requires: openai, fastapi, uvicorn, python-dotenv, pymongo, pillow, numpy, opencv-python

import os
import io
import re
import json
import logging
from datetime import datetime
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, File, UploadFile, Body, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image, ImageFilter, ImageEnhance
from dotenv import load_dotenv

# Image preprocessing libs (optional but recommended)
import numpy as np
import cv2

# OpenAI client
import openai

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in env")
openai.api_key = OPENAI_API_KEY

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "business_cards")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contacts")
PHONE_DEFAULT_REGION = os.getenv("PHONE_DEFAULT_REGION", "IN")

# FastAPI setup
app = FastAPI(title="Business Card OCR API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    email: Optional[str] = ""
    website: Optional[str] = ""
    address: Optional[str] = ""
    social_links: Optional[List[str]] = []
    more_details: Optional[str] = ""
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

    @validator("email", pre=True, always=True)
    def empty_or_valid_email(cls, v):
        v = (v or "").strip()
        if v == "":
            return ""
        # simple regex check
        if re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", v):
            return v
        raise ValueError("email must be a valid email address or empty")

# -----------------------------------------
# Simple normalization & heuristics (no external libs)
# -----------------------------------------
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.I)
PHONE_RE = re.compile(r"\+?\d[\d \-\(\)xextEXT]{6,}\d")
WEBSITE_RE = re.compile(r"(https?://)?(www\.)?([A-Za-z0-9\-]+\.[A-Za-z]{2,})(/[^\s]*)?", re.I)

SOCIAL_PLATFORMS = {
    "linkedin": ["linkedin.com", "linkedin"],
    "twitter": ["twitter.com", "x.com", "t.co", "twitter"],
    "instagram": ["instagram.com", "instagr.am"],
    "facebook": ["facebook.com", "fb.me", "facebook"],
    "telegram": ["t.me", "telegram.me"],
    "whatsapp": ["wa.me", "whatsapp"]
}

COMPANY_KEYWORDS_LABELS = [
    ("Pvt Ltd", ["pvt ltd", "private limited", "pvt. ltd"]),
    ("Ltd", ["ltd", "limited"]),
    ("Inc", ["inc", "inc."]),
    ("LLP", ["llp"]),
    ("Solutions", ["solutions"]),
]

ADDRESS_COUNTRY_KEYWORDS = {
    "India": ["india", "tamilnadu", "mumbai", "delhi", "bangalore", "chennai", "pincode", "pin"],
    "USA": ["usa", "united states", "california", "ny", "new york", "zip"],
    "UK": ["united kingdom", "england", "london", "uk"],
}

def normalize_phones(raw_phone_matches: List[str]) -> List[str]:
    out = []
    for p in raw_phone_matches:
        cleaned = re.sub(r"[^\d\+xX]", "", p)
        digits_only = re.sub(r"[^\d]", "", cleaned)
        if len(digits_only) >= 8:
            out.append(cleaned)
    seen = set()
    ret = []
    for x in out:
        if x not in seen:
            seen.add(x)
            ret.append(x)
    return ret

def extract_website_from_text(text: str) -> str:
    m = WEBSITE_RE.search(text)
    if not m:
        return ""
    # return the matched host + path if present
    return (m.group(0) or "").strip()

# -----------------------------------------
# Preprocessing (PIL + OpenCV)
# -----------------------------------------
from PIL import Image

def preprocess_pil_image(pil_img: Image.Image, upscale: bool = True) -> Image.Image:
    img = pil_img.convert("RGB")
    arr = np.array(img)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    if upscale and max(h, w) < 1200:
        scale = max(1.0, 1200.0 / max(h, w))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    try:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 15)
    except Exception:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    pil_proc = Image.fromarray(processed)
    pil_proc = pil_proc.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    pil_proc = ImageEnhance.Contrast(pil_proc).enhance(1.15)
    return pil_proc

# -----------------------------------------
# OCR via OpenAI Vision (strict JSON output)
# -----------------------------------------

def ocr_with_openai_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    import base64
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    system = (
        "You are an OCR assistant. You will receive an image (data URL). "
        "Extract contact card fields. Reply ONLY with valid JSON (no explanation, no markdown). "
        "Fields: name, designation, company, phone_numbers (list), email, website, address, social_links (list), raw_text. "
        "If a field is not found, return empty string or empty list for phone_numbers/social_links. "
        "Make phone numbers readable and preserve plus/extension characters. Do not invent data."
    )

    user = f"Image: {data_url}\n\nExtract the fields requested."
    model_name = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini-vision")

    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=1500,
            temperature=0.0
        )

        assistant_text = ""
        if resp and "choices" in resp and len(resp["choices"]) > 0:
            assistant_text = resp["choices"][0]["message"]["content"].strip()

        try:
            parsed = json.loads(assistant_text)
        except Exception:
            m = re.search(r"\{.*\}", assistant_text, re.S)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = {"raw_text": assistant_text}
            else:
                parsed = {"raw_text": assistant_text}
        return parsed

    except Exception:
        logging.exception("OpenAI OCR error")
        raise


def extract_details_via_openai(pil_image: Image.Image) -> Dict[str, Any]:
    proc = preprocess_pil_image(pil_image, upscale=True)
    buf = io.BytesIO()
    proc.save(buf, format="JPEG", quality=90)
    image_bytes = buf.getvalue()

    model_output = ocr_with_openai_image_bytes(image_bytes)

    data = {
        "name": model_output.get("name", "") if isinstance(model_output.get("name", ""), str) else "",
        "designation": model_output.get("designation", "") if isinstance(model_output.get("designation", ""), str) else "",
        "company": model_output.get("company", "") if isinstance(model_output.get("company", ""), str) else "",
        "phone_numbers": model_output.get("phone_numbers", []) if isinstance(model_output.get("phone_numbers", []), list) else [],
        "email": model_output.get("email", "") if isinstance(model_output.get("email", ""), str) else "",
        "website": model_output.get("website", "") if isinstance(model_output.get("website", ""), str) else "",
        "address": model_output.get("address", "") if isinstance(model_output.get("address", ""), str) else "",
        "social_links": model_output.get("social_links", []) if isinstance(model_output.get("social_links", []), list) else [],
        "more_details": "",
        "additional_notes": model_output.get("raw_text", "") if isinstance(model_output.get("raw_text", ""), str) else ""
    }

    raw = model_output.get("raw_text", "") or ""
    if not data["email"]:
        m = EMAIL_RE.search(raw)
        if m:
            data["email"] = m.group(0)

    if not data["website"]:
        w = extract_website_from_text(raw)
        if w:
            data["website"] = w

    if not data["phone_numbers"]:
        phones = PHONE_RE.findall(raw)
        data["phone_numbers"] = normalize_phones(phones)

    data["phone_numbers"] = list(dict.fromkeys(data["phone_numbers"]))
    data["social_links"] = list(dict.fromkeys(data["social_links"]))

    return data

# -----------------------------------------
# Lightweight classify_contact (no external libs)
# -----------------------------------------

def _detect_social_platforms(links: List[str]) -> List[str]:
    found = set()
    for link in links or []:
        low = (link or "").lower()
        for platform, tests in SOCIAL_PLATFORMS.items():
            if any(t in low for t in tests):
                found.add(platform)
    return sorted(found)


def _guess_company_type(company: str) -> str:
    if not company:
        return ""
    low = company.lower()
    for label, keywords in COMPANY_KEYWORDS_LABELS:
        for kw in keywords:
            if kw in low:
                return label
    return ""


def _guess_address_country(address: str) -> str:
    if not address:
        return ""
    low = address.lower()
    for country, keys in ADDRESS_COUNTRY_KEYWORDS.items():
        for k in keys:
            if k in low:
                return country
    return ""


def classify_contact(contact: Dict[str, Any]) -> Dict[str, Any]:
    c = dict(contact)

    phones = c.get("phone_numbers") or []
    if isinstance(phones, str):
        phones = [p.strip() for p in phones.split(",") if p.strip()]

    socials = c.get("social_links") or []
    if isinstance(socials, str):
        socials = [s.strip() for s in socials.split(",") if s.strip()]

    email = (c.get("email") or "").strip()
    website = (c.get("website") or "").strip()
    name = (c.get("name") or "").strip()
    company = (c.get("company") or "").strip()
    address = (c.get("address") or "").strip()
    notes = (c.get("additional_notes") or "").strip()
    designation = (c.get("designation") or "").strip()

    email_valid = bool(EMAIL_RE.match(email)) if email else False

    website_valid = False
    domain = ""
    if website:
        m = WEBSITE_RE.search(website)
        website_valid = bool(m)
        if m:
            domain = m.group(3) or ""

    phones_parsed = []
    for raw in phones or []:
        candidate = re.sub(r"[^\d\+xX]", "", raw)
        valid = len(re.sub(r"[^\d]", "", candidate)) >= 8
        phones_parsed.append({
            "raw": raw,
            "normalized": candidate if valid else None,
            "valid": bool(valid)
        })

    social_platforms = _detect_social_platforms(socials + ([website] if website else []) + ([email] if email else []))

    company_type = _guess_company_type(company)
    address_country = _guess_address_country(address)

    name_is_upper = bool(name and name.replace(" ", "").isupper())
    name_word_count = len(name.split()) if name else 0

    notes_len = len(notes)
    lines_in_notes = len([l for l in notes.splitlines() if l.strip()])

    field_validations = {
        "email": {"value": email, "valid": email_valid},
        "website": {"value": website, "valid": website_valid, "domain": domain},
        "phones": phones_parsed,
        "social_platforms_detected": social_platforms,
        "company_type_guess": company_type,
        "address_country_hint": address_country,
        "name": {"value": name, "is_uppercase": name_is_upper, "word_count": name_word_count},
        "notes": {"length": notes_len, "lines": lines_in_notes},
        "designation": {"value": designation}
    }

    c["field_validations"] = field_validations
    if "more_details" not in c or not c.get("more_details"):
        c["more_details"] = ""
    c["phone_numbers"] = phones
    c["social_links"] = socials
    return c

# -----------------------------------------
# Timestamp helper
# -----------------------------------------

def now_ist() -> str:
    try:
        import pytz
        ist = pytz.timezone("Asia/Kolkata")
        return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

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

        try:
            extracted = extract_details_via_openai(img)
        except Exception as e:
            logging.exception("ocr-with-openai failed")
            raise HTTPException(status_code=500, detail=f"OCR with OpenAI failed: {e}")

        extracted["_raw_ocr_text"] = extracted.get("additional_notes", "")
        extracted["_ocr_avg_confidence"] = None
        extracted["_ocr_word_count"] = len(extracted["_raw_ocr_text"].split())

        extracted = classify_contact(extracted)

        extracted["more_details"] = ""
        extracted["created_at"] = now_ist()
        extracted["edited_at"] = ""

        result = collection.insert_one(extracted)
        inserted = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("upload_card error")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------  The routes you asked to add ------------------
@app.post("/create_card", status_code=status.HTTP_201_CREATED)
async def create_card(payload: ContactCreate):
    try:
        doc = payload.dict()

        # classify & enrich before insert/update
        doc = classify_contact(doc)

        # If user provided more_details in payload, keep it (manual create)
        if payload.more_details:
            doc["more_details"] = payload.more_details
        else:
            doc["more_details"] = ""

        doc["created_at"] = now_ist()
        doc["edited_at"] = ""

        # If email exists, update that doc instead of creating a duplicate
        if doc.get("email"):
            existing = collection.find_one({"email": doc["email"]})
            if existing:
                # preserve existing more_details if user didn't provide one
                if not doc.get("more_details"):
                    doc["more_details"] = existing.get("more_details", "")
                doc["edited_at"] = now_ist()
                collection.update_one({"_id": existing["_id"]}, {"$set": doc})
                updated = collection.find_one({"_id": existing["_id"]})
                return {"message": "Updated existing contact", "data": JSONEncoder.encode(updated)}

        result = collection.insert_one(doc)
        inserted = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}
    except Exception as e:
        logging.exception("create_card error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/all_cards")
def get_all_cards():
    try:
        docs = list(collection.find().sort([("_id", -1)]))
        return {"data": JSONEncoder.encode(docs)}
    except Exception as e:
        logging.exception("get_all_cards error")
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
        logging.exception("update_notes error")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/update_card/{card_id}")
def update_card(card_id: str, payload: dict = Body(...)):
    try:
        allowed_fields = {
            "name", "designation", "company", "phone_numbers",
            "email", "website", "address", "social_links",
            "additional_notes", "more_details"
        }

        update_data = {k: v for k, v in payload.items() if k in allowed_fields}
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields to update.")

        # fetch existing doc
        existing = collection.find_one({"_id": ObjectId(card_id)})
        if not existing:
            raise HTTPException(status_code=404, detail="Card not found.")

        existing_more = existing.get("more_details", "")

        merged = dict(existing)
        merged.update(update_data)

        # classify -> returns normalized phone/socials and adds field_validations
        merged = classify_contact(merged)

        # ensure more_details: if user provided it in update_data, keep that; otherwise preserve existing
        if "more_details" in update_data:
            merged["more_details"] = update_data.get("more_details", "")
        else:
            merged["more_details"] = existing_more

        # pick only allowed fields + classification fields to set
        set_payload = {k: merged.get(k) for k in list(allowed_fields) + ["field_validations"]}
        set_payload["edited_at"] = now_ist()

        collection.update_one({"_id": ObjectId(card_id)}, {"$set": set_payload})
        updated = collection.find_one({"_id": ObjectId(card_id)})
        return {"message": "Updated", "data": JSONEncoder.encode(updated)}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("update_card error")
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
        logging.exception("delete_card error")
        raise HTTPException(status_code=500, detail=str(e))
