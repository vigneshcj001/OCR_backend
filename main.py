# main.py
"""
FastAPI Business Card OCR backend (Tesseract OCR + OpenAI chat completions parser)
Requirements:
  pip install fastapi uvicorn python-multipart pillow pytesseract openai python-dotenv pymongo validators phonenumbers tldextract pytz
  Install Tesseract OCR binary (Linux: apt install tesseract-ocr, Mac: brew install tesseract, Windows: UB-Mannheim)

Environment variables:
  OPENAI_API_KEY
  MONGO_URI (optional; default mongodb://localhost:27017)
  DB_NAME, COLLECTION_NAME (optional)
  TESSERACT_PATH (optional, Windows path e.g. 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe')
  PHONE_DEFAULT_REGION (optional; default 'IN')
Run:
  uvicorn main:app --reload --port 8000
"""

import os
import io
import re
import json
import logging
from datetime import datetime
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, File, UploadFile, Body, HTTPException, status, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import pytz
from PIL import Image
import pytesseract

# validation libs
import phonenumbers
import tldextract
import validators

# OpenAI v1 client (chat completions)
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None

# Load .env
load_dotenv()

# Tesseract executable path (optional)
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Mongo config
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "business_cards")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contacts")
PHONE_DEFAULT_REGION = os.getenv("PHONE_DEFAULT_REGION", "IN")

# Initialize OpenAI client lazily when needed
def get_openai_client(api_key: Optional[str] = None):
    if OpenAI is None:
        raise RuntimeError("openai package not installed. pip install openai")
    key = api_key or OPENAI_API_KEY
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (env) and not provided")
    return OpenAI(api_key=key)

# Mongo client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Helper: JSON encoder for ObjectId
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

# Utilities
def now_ist() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

def re_sub_digits_plus_x(s: str) -> str:
    return re.sub(r"[^\d\+xX]", "", s or "")

def parse_phones(phone_list: List[str]) -> List[Dict[str, Any]]:
    out = []
    for raw in phone_list or []:
        raw = str(raw)
        candidate = re_sub_digits_plus_x(raw)
        try:
            parsed = phonenumbers.parse(candidate, PHONE_DEFAULT_REGION)
            is_valid = phonenumbers.is_valid_number(parsed)
            e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164) if is_valid else candidate
            phone_type = None
            try:
                phone_type = phonenumbers.number_type(parsed).name
            except Exception:
                phone_type = None
            out.append({
                "raw": raw,
                "normalized": e164,
                "country_code": getattr(parsed, "country_code", None),
                "national_number": getattr(parsed, "national_number", None),
                "valid": bool(is_valid),
                "type": phone_type
            })
        except Exception:
            out.append({
                "raw": raw,
                "normalized": None,
                "country_code": None,
                "national_number": None,
                "valid": False,
                "type": None
            })
    return out

SOCIAL_PLATFORMS = {
    "linkedin": ["linkedin.com", "linkedin"],
    "twitter": ["twitter.com", "x.com", "t.co", "twitter"],
    "instagram": ["instagram.com", "instagr.am", "instagram"],
    "facebook": ["facebook.com", "fb.me", "facebook"],
    "telegram": ["t.me", "telegram.me", "telegram"],
    "whatsapp": ["wa.me", "whatsapp"]
}

def _detect_social_platforms(links: List[str]) -> List[str]:
    found = set()
    for link in links or []:
        low = (link or "").lower()
        for platform, tests in SOCIAL_PLATFORMS.items():
            if any(t in low for t in tests):
                found.add(platform)
    return sorted(found)

COMPANY_KEYWORDS_LABELS = [
    ("Pvt Ltd", ["pvt ltd", "private limited", "pvt. ltd"]),
    ("Ltd", ["ltd", "limited"]),
    ("Inc", ["inc", "inc."]),
    ("LLP", ["llp"]),
    ("Solutions", ["solutions"]),
    ("Technologies", ["technologies", "tech"]),
    ("Works", ["works"])
]

ADDRESS_COUNTRY_KEYWORDS = {
    "India": ["india", "tamilnadu", "mumbai", "delhi", "bangalore", "chennai", "pincode", "pin"],
    "USA": ["usa", "united states", "california", "ny", "new york", "zip"],
    "UK": ["united kingdom", "england", "london", "uk"],
}

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

    email_valid = bool(validators.email(email)) if email else False

    website_valid = False
    domain = ""
    if website:
        website_try = website if website.startswith(("http://", "https://")) else ("http://" + website)
        website_valid = bool(validators.url(website_try))
        te = tldextract.extract(website)
        if te and te.domain:
            domain = ".".join([p for p in [te.domain, te.suffix] if p])

    phones_parsed = parse_phones(phones)

    social_platforms = _detect_social_platforms(socials + [website] + [email])

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
    c["phone_numbers"] = phones
    c["social_links"] = socials
    if "more_details" not in c or not c.get("more_details"):
        c["more_details"] = ""
    return c

# ------------------------------
# OCR + OpenAI parsing (from first snippet)
# ------------------------------
def clean_ocr_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

PARSER_PROMPT = (
    "You are an assistant that extracts structured contact fields from messy OCR'd text from a business card.\n"
    "Return a JSON object with keys: name, company, title, email, phone, website, address, extra.\n"
    "If a field is not present, set it to null. For 'extra' include any other useful strings (fax, linkedin, notes).\n"
    "Also add a short field 'confidence_notes' describing any ambiguity.\n\n"
    "Respond ONLY with the JSON object.\n"
)

def call_openai_parse(ocr_text: str, api_key: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Uses OpenAI chat completions (v1 style) to parse OCR text into JSON fields.
    Model name default is 'gpt-4o' (change as appropriate).
    """
    client = get_openai_client(api_key)
    prompt = PARSER_PROMPT + "\nOCR_TEXT:\n" + ocr_text + "\n\nRespond with JSON only."

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a JSON-only extractor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=512,
    )

    # extract assistant text
    try:
        assistant_text = resp.choices[0].message.content.strip()
    except Exception:
        assistant_text = str(resp)

    # try parse JSON
    try:
        parsed = json.loads(assistant_text)
        return parsed
    except Exception:
        m = re.search(r"\{[\s\S]*\}$", assistant_text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # fallback
        return {
            "name": None,
            "company": None,
            "title": None,
            "email": None,
            "phone": None,
            "website": None,
            "address": None,
            "extra": {"model_output": assistant_text},
            "confidence_notes": "Model output not parseable as JSON. See extra.model_output."
        }

def generate_vcard(data: Dict[str, Optional[str]]) -> str:
    lines = ["BEGIN:VCARD", "VERSION:3.0"]
    if data.get("name"):
        lines.append(f"FN:{data.get('name')}")
    if data.get("company"):
        lines.append(f"ORG:{data.get('company')}")
    if data.get("title"):
        lines.append(f"TITLE:{data.get('title')}")
    if data.get("phone"):
        lines.append(f"TEL;TYPE=WORK,VOICE:{data.get('phone')}")
    if data.get("email"):
        lines.append(f"EMAIL;TYPE=WORK:{data.get('email')}")
    if data.get("website"):
        lines.append(f"URL:{data.get('website')}")
    if data.get("address"):
        lines.append(f"ADR;TYPE=WORK:;;{data.get('address')}")
    lines.append("END:VCARD")
    return "\n".join(lines)

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Business Card OCR Backend (Tesseract + OpenAI parser)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        if validators.email(v):
            return v
        raise ValueError("email must be a valid email address or empty")

@app.get("/")
def root():
    return {"message": "OCR Backend Running ✅ (Tesseract OCR + OpenAI parser)"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/upload_card", status_code=status.HTTP_201_CREATED)
async def upload_card(file: UploadFile = File(...), authorization: Optional[str] = Header(None), model: Optional[str] = "gpt-4o"):
    """
    Upload an image file (form field 'file').
    Steps:
      - Validate image
      - Run Tesseract OCR (pytesseract)
      - Send OCR text to OpenAI chat completions (call_openai_parse)
      - Normalize, classify, store in MongoDB
    """
    # Obtain API key: Authorization header Bearer > env
    api_key = None
    if authorization and authorization.lower().startswith("bearer "):
        api_key = authorization.split(" ", 1)[1].strip()
    if not api_key:
        api_key = OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not provided. Set OPENAI_API_KEY or provide Authorization: Bearer <key> header")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # optional resize for large images
    max_dim = 1800
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0]*ratio), int(image.size[1]*ratio))
        image = image.resize(new_size)

    # OCR via Tesseract
    try:
        raw_text = pytesseract.image_to_string(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failure: {e}")

    raw_text = clean_ocr_text(raw_text)
    if not raw_text:
        # still proceed - but likely no result
        logging.warning("OCR returned empty text for uploaded image.")

    # Call OpenAI to parse the OCR text -> structured fields
    parsed = call_openai_parse(raw_text, api_key=api_key, model=model)

    # Map parsed fields into our contact schema
    data = {
        "name": parsed.get("name") or "",
        "designation": parsed.get("title") or "",
        "company": parsed.get("company") or "",
        "phone_numbers": [],
        "email": parsed.get("email") or "",
        "website": parsed.get("website") or "",
        "address": parsed.get("address") or "",
        "social_links": [],
        "more_details": "",
        "additional_notes": parsed.get("extra") or ""
    }

    # 'phone' may be single string; normalize to list
    phone = parsed.get("phone")
    if phone:
        if isinstance(phone, list):
            data["phone_numbers"] = phone
        else:
            # split by common separators
            data["phone_numbers"] = [p.strip() for p in re.split(r"[;,/|\\n]+", str(phone)) if p.strip()]

    # If extra contains obvious social/links, try to detect
    extra = parsed.get("extra")
    if isinstance(extra, dict):
        # flatten values to try find links
        vals = []
        for v in extra.values():
            if isinstance(v, (list, dict)):
                vals.extend([str(x) for x in v if x])
            else:
                if v:
                    vals.append(str(v))
        data["social_links"] = [v for v in vals if v and (v.startswith("http") or any(k in v.lower() for k in ["linkedin", "instagram", "twitter", "x.com", "wa.me", "t.me"]))]
    else:
        if extra and isinstance(extra, str):
            data["social_links"] = [s.strip() for s in re.findall(r"(https?://[^\s,;]+|www\.[^\s,;]+|linkedin[^\s,;]+)", extra)]

    # store raw data for debugging
    data["_ocr_raw_text"] = raw_text
    data["_openai_raw"] = parsed.get("extra", {}).get("model_output") if isinstance(parsed.get("extra"), dict) else None
    data["_openai_model"] = model
    data["_processed_with_openai_at"] = now_ist()

    # classify & enrich
    data = classify_contact(data)
    data["created_at"] = now_ist()
    data["edited_at"] = ""

    # insert to MongoDB
    result = collection.insert_one(data)
    inserted = collection.find_one({"_id": result.inserted_id})
    return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}

@app.post("/create_card", status_code=status.HTTP_201_CREATED)
async def create_card(payload: ContactCreate):
    try:
        doc = payload.dict()
        doc = classify_contact(doc)
        if payload.more_details:
            doc["more_details"] = payload.more_details
        else:
            doc["more_details"] = ""
        doc["created_at"] = now_ist()
        doc["edited_at"] = ""
        if doc.get("email"):
            existing = collection.find_one({"email": doc["email"]})
            if existing:
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

        existing = collection.find_one({"_id": ObjectId(card_id)})
        if not existing:
            raise HTTPException(status_code=404, detail="Card not found.")

        existing_more = existing.get("more_details", "")

        merged = dict(existing)
        merged.update(update_data)

        merged = classify_contact(merged)

        if "more_details" in update_data:
            merged["more_details"] = update_data.get("more_details", "")
        else:
            merged["more_details"] = existing_more

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

@app.post("/vcard")
async def vcard_endpoint(payload: dict = Body(...)):
    """
    Accepts JSON body with contact fields and returns a .vcf file for download.
    Expected keys: name, company, title, phone, email, website, address
    """
    data = payload or {}
    vcard = generate_vcard(data)
    return StreamingResponse(io.BytesIO(vcard.encode("utf-8")), media_type="text/vcard",
                            headers={"Content-Disposition": "attachment; filename=contact.vcf"})
