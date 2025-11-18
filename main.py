# File: main.py (FastAPI backend)
import os
import io
import re
import logging
from datetime import datetime
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, File, UploadFile, Body, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract
from pytesseract import Output
from dotenv import load_dotenv
import pytz

# --- Additional libs for classification/validation / preprocessing ---
import numpy as np
import cv2

# --- Additional libs for classification/validation ---
import phonenumbers
import tldextract
import validators

# Try loading spaCy NER (optional)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# Load .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "business_cards")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contacts")
PHONE_DEFAULT_REGION = os.getenv("PHONE_DEFAULT_REGION", "IN")  # default phone region

# Optional: explicitly set tesseract binary if not on PATH
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # adjust path if necessary

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
    email: Optional[str] = ""          # allow empty string; validated below
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
        # Use validators package to check email format; clearer error reporting
        if validators.email(v):
            return v
        raise ValueError("email must be a valid email address or empty")

# -----------------------------------------
# Utilities: OCR preprocessing + parsing + heuristics
# -----------------------------------------
from difflib import SequenceMatcher

def preprocess_pil_image(pil_img: Image.Image, upscale: bool = True) -> Image.Image:
    """
    Convert to RGB, grayscale, upscale if small, denoise, adaptive threshold,
    mild sharpening and contrast. Returns a PIL Image (binary/thresholded)
    suitable for Tesseract.
    """
    # Ensure RGB then convert to numpy
    img = pil_img.convert("RGB")
    arr = np.array(img)

    # Convert to gray
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Optional upscale to improve OCR on small images
    h, w = gray.shape
    if upscale and max(h, w) < 1200:
        scale = max(1.0, 1200.0 / max(h, w))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Denoise (bilateral preserves edges)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive thresholding for uneven lighting
    try:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 15)
    except Exception:
        # fallback to Otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological open (remove speckles)
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    pil_proc = Image.fromarray(processed)

    # Mild sharpening and contrast increase
    pil_proc = pil_proc.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    pil_proc = ImageEnhance.Contrast(pil_proc).enhance(1.15)

    return pil_proc

def extract_details(text: str) -> Dict[str, Any]:
    """
    Enhanced OCR parsing with improved field extraction logic.
    Uses multi-pass strategies and confidence scoring for better accuracy.
    """
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
        "more_details": "",
        "additional_notes": raw_text,
    }
    
    # ====== HELPER PATTERNS & KEYWORDS ======
    company_keywords = [
        "pvt", "private", "ltd", "llp", "inc", "solutions", "technologies", "tech",
        "corporation", "company", "corp", "industries", "works", "enterprises",
        "consulting", "services", "group", "international", "systems"
    ]
    
    designation_keywords = [
        "founder", "co-founder", "ceo", "cto", "coo", "cfo", "manager", "director",
        "engineer", "consultant", "head", "lead", "president", "vp", "vice president",
        "principal", "officer", "specialist", "executive", "developer", "designer",
        "architect", "analyst", "coordinator", "supervisor", "administrator"
    ]
    
    address_tokens = [
        "street", "st", "road", "rd", "avenue", "ave", "nagar", "lane", "city",
        "state", "pin", "pincode", "zip", "near", "opp", "opposite", "building",
        "bldg", "floor", "suite", "ste", "apartment", "apt", "block", "sector"
    ]
    
    # ====== 1. EMAIL EXTRACTION (Enhanced) ======
    email_patterns = [
        r"[\w\.\-\+]+@[\w\.\-]+\.[a-zA-Z]{2,}",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ]
    emails = []
    for pattern in email_patterns:
        emails.extend(re.findall(pattern, raw_text, re.I))
    data["email"] = emails[0] if emails else ""
    
    # ====== 2. PHONE EXTRACTION (Enhanced) ======
    phone_patterns = [
        r"\+\d{1,3}[\s\-]?\d{3,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4}",  # International
        r"\d{3}[\s\-]\d{3}[\s\-]\d{4}",  # US format
        r"\(\d{3}\)[\s\-]?\d{3}[\s\-]?\d{4}",  # (123) 456-7890
        r"\d{5}[\s\-]\d{5}",  # Indian format
        r"\d{10,}",  # 10+ digits
    ]
    phones = set()
    for pattern in phone_patterns:
        matches = re.findall(pattern, raw_text)
        for match in matches:
            cleaned = re.sub(r"[^\d\+]", "", match)
            digits_only = re.sub(r"[^\d]", "", cleaned)
            if 8 <= len(digits_only) <= 15:  # Valid phone length
                phones.add(cleaned)
    data["phone_numbers"] = list(phones)
    
    # ====== 3. WEBSITE EXTRACTION (Enhanced) ======
    website_patterns = [
        r"https?://[\w\.\-]+\.[\w]{2,}(?:/[\w\.\-]*)*",
        r"www\.[\w\.\-]+\.[\w]{2,}",
        r"\b[\w\-]+\.(?:com|in|net|org|co|io|biz|info|xyz|me|ai|tech|dev)\b"
    ]
    websites = []
    for pattern in website_patterns:
        websites.extend(re.findall(pattern, raw_text, re.I))
    if websites:
        # Prefer full URLs, fallback to domain
        full_urls = [w for w in websites if w.startswith(("http", "www"))]
        data["website"] = full_urls[0] if full_urls else websites[0]
    
    # ====== 4. SOCIAL LINKS (Enhanced) ======
    social_patterns = {
        "linkedin": r"(?:linkedin\.com/in/|linkedin\.com/company/)[\w\-]+",
        "twitter": r"(?:twitter\.com/|x\.com/)[\w\-]+",
        "instagram": r"instagram\.com/[\w\-\.]+",
        "facebook": r"facebook\.com/[\w\-\.]+",
        "github": r"github\.com/[\w\-]+",
    }
    
    for platform, pattern in social_patterns.items():
        matches = re.findall(pattern, raw_text, re.I)
        data["social_links"].extend(matches)
    
    # Check for handle patterns (@ mentions)
    handle_matches = re.findall(r"@[\w\-]+", raw_text)
    data["social_links"].extend(handle_matches)
    
    # ====== 5. DESIGNATION EXTRACTION (Multi-strategy) ======
    designation_found = False
    
    # Strategy 1: Lines containing designation keywords
    for line in lines:
        low = line.lower()
        if any(kw in low for kw in designation_keywords):
            # Clean and extract
            cleaned = re.sub(r"[^\w\s\-&/]", " ", line).strip()
            words = cleaned.split()
            if 1 <= len(words) <= 8:
                data["designation"] = " ".join(words)
                designation_found = True
                break
    
    # Strategy 2: Second or third line if capitalized (common card layout)
    if not designation_found and len(lines) >= 2:
        for idx in [1, 2]:
            if idx < len(lines):
                line = lines[idx]
                # Skip if looks like email/phone/website
                if "@" in line or re.search(r"\d{3}", line) or ".com" in line.lower():
                    continue
                cleaned = re.sub(r"[^\w\s\-&/]", " ", line).strip()
                words = cleaned.split()
                if 1 <= len(words) <= 6:
                    data["designation"] = " ".join(words)
                    designation_found = True
                    break
    
    # ====== 6. COMPANY EXTRACTION (Multi-strategy) ======
    company_candidates = []
    
    # Strategy 1: Lines with company keywords
    for idx, line in enumerate(lines):
        low = line.lower()
        if any(kw in low for kw in company_keywords):
            if not re.search(r"@", line) and not re.search(r"\d{3,}", line):
                score = sum(1 for kw in company_keywords if kw in low)
                company_candidates.append((score, idx, line.strip()))
    
    # Strategy 2: Lines with 2-5 capitalized words (brand names)
    if not company_candidates:
        for idx, line in enumerate(lines[:6]):  # Check top 6 lines
            if "@" in line or re.search(r"\+?\d{3}", line):
                continue
            words = re.findall(r"\b[A-Z][a-z]+\b", line)
            if 2 <= len(words) <= 5:
                company_candidates.append((1, idx, line.strip()))
    
    if company_candidates:
        # Sort by score (higher = more likely company)
        company_candidates.sort(key=lambda x: (-x[0], x[1]))
        data["company"] = company_candidates[0][2]
    
    # ====== 7. NAME EXTRACTION (Enhanced Logic) ======
    name_candidates = []
    
    # Strategy 1: First line if ALL CAPS (2-4 words)
    if lines:
        first = lines[0]
        cleaned = re.sub(r"[^\w\s]", "", first).strip()
        words = cleaned.split()
        if 2 <= len(words) <= 4 and cleaned.replace(" ", "").isupper():
            name_candidates.append((5, cleaned.title()))
    
    # Strategy 2: Lines with Title Case pattern
    for idx, line in enumerate(lines[:4]):
        if "@" in line or re.search(r"\d{3}", line):
            continue
        # Check if words start with capital
        words = re.findall(r"\b[A-Z][a-z]+\b", line)
        if 2 <= len(words) <= 4:
            full_name = " ".join(words)
            # Avoid if it's the company or designation
            if full_name != data.get("company") and full_name != data.get("designation"):
                name_candidates.append((3, full_name))
    
    # Strategy 3: SpaCy PERSON entities (if available)
    if nlp:
        try:
            doc = nlp(raw_text)
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            for person in persons[:3]:  # Top 3 person entities
                words = person.split()
                if 2 <= len(words) <= 4:
                    name_candidates.append((4, person))
        except:
            pass
    
    if name_candidates:
        # Sort by score and pick best
        name_candidates.sort(key=lambda x: -x[0])
        data["name"] = name_candidates[0][1]
    
    # ====== 8. ADDRESS EXTRACTION (Enhanced) ======
    address_lines = []
    
    for line in lines:
        low = line.lower()
        # Check for address indicators
        has_pincode = bool(re.search(r"\b\d{5,6}\b", line))
        has_address_token = any(tok in low for tok in address_tokens)
        has_location = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2,}\b", line))
        
        if has_pincode or has_address_token or has_location:
            address_lines.append(line)
    
    if address_lines:
        data["address"] = ", ".join(address_lines)
    
    # ====== 9. CLEANUP & VALIDATION ======
    # Remove leading/trailing special chars
    for field in ["name", "designation", "company", "address"]:
        if data[field]:
            data[field] = re.sub(r"^[\W_]+|[\W_]+$", "", data[field]).strip()
    
    # Ensure name is not same as company
    if data["name"] and data["company"]:
        if SequenceMatcher(None, data["name"].lower(), data["company"].lower()).ratio() > 0.7:
            # Try to find alternative name
            for line in lines[:4]:
                cleaned = re.sub(r"[^\w\s]", "", line).strip()
                words = cleaned.split()
                if 2 <= len(words) <= 4 and cleaned != data["company"]:
                    data["name"] = " ".join([w.capitalize() for w in words])
                    break
    
    # Deduplicate lists
    data["phone_numbers"] = list(dict.fromkeys(data["phone_numbers"]))
    data["social_links"] = list(dict.fromkeys(data["social_links"]))
    
    return data
# -----------------------------------------
# Timestamp helper
# -----------------------------------------
def now_ist() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------------------
# Classification helpers (unchanged)
# -----------------------------------------
SOCIAL_PLATFORMS = {
    "linkedin": ["linkedin.com", "linkedin"],
    "twitter": ["twitter.com", "x.com", "t.co", "twitter"],
    "instagram": ["instagram.com", "instagr.am", "instagram"],
    "facebook": ["facebook.com", "fb.me", "facebook"],
    "telegram": ["t.me", "telegram.me", "telegram"],
    "whatsapp": ["wa.me", "whatsapp"]
}

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

def _detect_social_platforms(links: List[str]) -> List[str]:
    found = set()
    for link in links or []:
        low = link.lower()
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

def parse_phones(phone_list: List[str]) -> List[Dict[str, Any]]:
    out = []
    for raw in phone_list or []:
        candidate = re.sub(r"[^\d\+xX]", "", raw)
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

def classify_contact(contact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate & enrich contact. Keeps field_validations but leaves more_details empty
    unless provided by the caller (frontend controls it).
    """
    c = dict(contact)  # shallow copy

    # Normalize types
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

    # Email validation
    email_valid = bool(validators.email(email)) if email else False

    # Website validation & domain
    website_valid = False
    domain = ""
    if website:
        website_try = website if website.startswith(("http://", "https://")) else ("http://" + website)
        website_valid = bool(validators.url(website_try))
        te = tldextract.extract(website)
        if te and te.domain:
            domain = ".".join([p for p in [te.domain, te.suffix] if p])

    # Phones parse
    phones_parsed = parse_phones(phones)

    # Social platform detection (include email + website)
    social_platforms = _detect_social_platforms(socials + [website] + [email])

    # Company type & address country
    company_type = _guess_company_type(company)
    address_country = _guess_address_country(address)

    # Name heuristics
    name_is_upper = bool(name and name.replace(" ", "").isupper())
    name_word_count = len(name.split()) if name else 0

    # NER quick check using spaCy (optional)
    ner_org = ""
    ner_gpe = ""
    if nlp and (company or address or name):
        try:
            txt = " ".join([x for x in [name, company, address] if x])
            doc = nlp(txt)
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            gpes = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
            ner_org = orgs[0] if orgs else ""
            ner_gpe = gpes[0] if gpes else ""
        except Exception:
            ner_org = ""
            ner_gpe = ""

    # Notes stats
    notes_len = len(notes)
    lines_in_notes = len([l for l in notes.splitlines() if l.strip()])

    # Build structured validation doc
    field_validations = {
        "email": {"value": email, "valid": email_valid},
        "website": {"value": website, "valid": website_valid, "domain": domain},
        "phones": phones_parsed,
        "social_platforms_detected": social_platforms,
        "company_type_guess": company_type,
        "address_country_hint": address_country,
        "name": {"value": name, "is_uppercase": name_is_upper, "word_count": name_word_count},
        "ner_org": ner_org,
        "ner_gpe": ner_gpe,
        "notes": {"length": notes_len, "lines": lines_in_notes},
        "designation": {"value": designation}
    }

    c["field_validations"] = field_validations

    # Keep more_details empty unless provided
    if "more_details" not in c or not c.get("more_details"):
        c["more_details"] = ""

    # Ensure normalized types
    c["phone_numbers"] = phones
    c["social_links"] = socials

    return c

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

        # PREPROCESS + OCR (robust)
        try:
            proc_img = preprocess_pil_image(img, upscale=True)

            # choose OCR config; psm 6 is good for a block of text, adjust for layout
            ocr_config = "--oem 3 --psm 6"
            raw_text = pytesseract.image_to_string(proc_img, config=ocr_config, lang='eng')

            # detailed data (word-level) — gives confidence, bbox, text
            ocr_data = pytesseract.image_to_data(proc_img, config=ocr_config, lang='eng', output_type=Output.DICT)

            # build a simple confidence summary
            confidences = []
            for c in ocr_data.get("conf", []):
                try:
                    # pytesseract may output "-1" or "-1\n"
                    conf = int(float(c))
                    confidences.append(conf)
                except Exception:
                    pass
            avg_conf = sum(confidences)/len(confidences) if confidences else None

        except Exception as e:
            logging.exception("ocr preprocessing error")
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")

        # Extract structured details from OCR text
        extracted = extract_details(raw_text)

        # store debug fields to help triage poor OCR results
        extracted["_raw_ocr_text"] = raw_text
        extracted["_ocr_avg_confidence"] = avg_conf
        extracted["_ocr_word_count"] = len([t for t in ocr_data.get("text", []) if t and t.strip()])

        # optional: store low-confidence words (capped)
        low_conf_words = []
        try:
            for i, w in enumerate(ocr_data.get("text", [])):
                if not w or not w.strip():
                    continue
                try:
                    conf = int(float(ocr_data.get("conf", [])[i]))
                except Exception:
                    conf = -1
                if conf >= 0 and conf < 60:
                    low_conf_words.append({"word": w, "conf": conf})
            extracted["_ocr_low_conf_words"] = low_conf_words[:40]
        except Exception:
            # non-fatal: don't block processing if this debug step fails
            extracted["_ocr_low_conf_words"] = []

        # classify & enrich before storing (computes validations but leaves more_details empty)
        extracted = classify_contact(extracted)

        # ensure more_details is empty for newly created records (user will fill later)
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

