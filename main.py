# backend.py
"""
Business Card OCR Backend (OpenAI required + MongoDB)

Features:
- Image preprocessing (grayscale, upscale, denoise, adaptive threshold, optional deskew)
- Multi-PSM Tesseract attempts, picks best by mean confidence
- Rotation fallback (90/270) for sideways images
- Local regex-based parsing for fallback/augmentation
- Required OpenAI parsing to extract structured fields (phone_numbers and social_links enforced as lists)
- Safer update endpoint: only $set fields intentionally provided by client
- MongoDB persistence and vCard generation

Requirements (Python packages):
    pip install fastapi uvicorn python-multipart pillow pytesseract pymongo openai numpy
System-level:
    - Install Tesseract OCR on host (apt, brew, or Windows installer)
    - If Tesseract binary isn't on PATH, set TESSERACT_PATH env var to full path
"""

import os
import io
import re
import json
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Body, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract
import numpy as np

# OpenAI (required)
from openai import OpenAI

# -----------------------
# Logging & config
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("business-card-backend")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "business_cards")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contacts")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

TESSERACT_PATH = os.getenv("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# -----------------------
# Utilities
# -----------------------
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _ensure_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    if isinstance(v, (int, float)):
        return [str(v)]
    if isinstance(v, str):
        items = [x.strip() for x in re.split(r"[,\n;]+", v) if x.strip()]
        return items
    try:
        return [str(v)]
    except Exception:
        return []

def clean_ocr_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    # trim long leading/trailing whitespace
    return text.strip()

def normalize_payload(payload: dict) -> dict:
    out = {}
    out["name"] = payload.get("name")
    out["designation"] = payload.get("designation") or payload.get("title")
    out["company"] = payload.get("company")
    out["phone_numbers"] = _ensure_list(payload.get("phone_numbers") or payload.get("phone") or payload.get("phones"))
    out["email"] = payload.get("email")
    out["website"] = payload.get("website")
    out["address"] = payload.get("address")
    out["social_links"] = _ensure_list(payload.get("social_links") or payload.get("social") or payload.get("linkedin"))
    out["more_details"] = payload.get("more_details") or ""
    out["additional_notes"] = payload.get("additional_notes") or ""
    return out

def db_doc_to_canonical(doc: dict) -> dict:
    if not doc:
        return {}
    canonical = {
        "_id": str(doc.get("_id")),
        "name": doc.get("name"),
        "designation": doc.get("designation"),
        "company": doc.get("company"),
        "phone_numbers": doc.get("phone_numbers") or [],
        "email": doc.get("email"),
        "website": doc.get("website"),
        "address": doc.get("address"),
        "social_links": doc.get("social_links") or [],
        "more_details": doc.get("more_details") or "",
        "additional_notes": doc.get("additional_notes") or "",
        "created_at": doc.get("created_at"),
        "edited_at": doc.get("edited_at"),
        "field_validations": doc.get("field_validations", {}),
    }
    if "raw_text" in doc:
        canonical["raw_text"] = doc.get("raw_text")
    if "confidence_notes" in doc:
        canonical["confidence_notes"] = doc.get("confidence_notes")
    if "extra" in doc:
        canonical["extra"] = doc.get("extra")
    return canonical

# -----------------------
# Local regex-based extractor
# -----------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.I)

PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-.])?(?:\(?\d{2,4}\)?[\s\-.])?(?:\d{2,4}[\s\-.]?){1,4}\d{2,4}"
)

WWW_RE = re.compile(
    r"(https?://[^\s,;]+|www\.[^\s,;]+|[A-Za-z0-9.-]+\.(?:com|net|org|io|in|co|ai|tech|dev|biz|info)(?:/[^\s,;]*)?)",
    re.I,
)

COMPANY_HINTS = [
    r"\b(Ltd|Pvt|Private|LLP|Limited|Inc|Corporation|Company|Technologies|Tech|Solutions|Works|Consultants|Advisory|Group|Systems|Enterprises|Industries)\b"
]

def _normalize_phone_string(p: str) -> str:
    if not p:
        return ""
    p = p.strip()
    plus = p.startswith("+")
    digits = re.sub(r"[^\d]", "", p)
    if plus:
        return "+" + digits
    return digits

def _dedupe_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def local_parse_from_ocr(ocr_text: str) -> Tuple[Dict[str, Any], str]:
    parsed = {
        "name": None,
        "company": None,
        "title": None,
        "email": None,
        "phone": None,
        "website": None,
        "address": None,
        "extra": {},
        "confidence_notes": None,
    }

    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]

    # Emails
    emails = EMAIL_RE.findall(ocr_text)
    if emails:
        emails_clean = [e.strip().rstrip(".,;") for e in emails]
        parsed["extra"]["emails_all"] = emails_clean
        parsed["email"] = emails_clean[0]

    # Websites
    wwws = WWW_RE.findall(ocr_text)
    if wwws:
        wwws_clean = []
        for w in wwws:
            if "@" in w:
                continue
            w = w.strip().rstrip(".,;")
            if w.lower().startswith("www."):
                w = "http://" + w
            wwws_clean.append(w)
        if wwws_clean:
            parsed["extra"]["websites_all"] = _dedupe_keep_order(w for w in wwws_clean)
            parsed["website"] = parsed["extra"]["websites_all"][0]

    # Phones
    phones_raw = PHONE_RE.findall(ocr_text)
    phones = []
    if phones_raw:
        for m in phones_raw:
            p = m if isinstance(m, str) else "".join(m)
            p_norm = _normalize_phone_string(p)
            digits_only = re.sub(r"[^\d]", "", p_norm)
            if 7 <= len(digits_only) <= 15:
                phones.append(p_norm)
    if phones:
        phones = _dedupe_keep_order(phones)
        parsed["extra"]["phones_all"] = phones
        parsed["phone"] = phones[0]

    # Name heuristics
    def looks_like_name(s: str) -> bool:
        if not s or len(s) < 2 or len(s) > 40:
            return False
        if EMAIL_RE.search(s) or WWW_RE.search(s) or PHONE_RE.search(s):
            return False
        low = s.lower()
        bad_words = ("ltd", "pvt", "private", "company", "technologies", "solutions", "consultants", "inc", "www", "http", "co.", "group", "llp")
        if any(w in low for w in bad_words):
            return False
        if re.search(r"\b(CEO|Founder|Director|Partner|Manager|Consultant|Officer|Advisory|Placement|Head|Chief|Executive|Officer)\b", s, re.I):
            return False
        words = s.split()
        if len(words) > 6:
            return False
        alpha_ratio = sum(c.isalpha() for c in s) / max(1, len(s))
        if s.isupper() and alpha_ratio > 0.6:
            return False
        if alpha_ratio < 0.4:
            return False
        title_like = sum(1 for w in words if w and w[0].isupper())
        if title_like >= 1:
            return True
        return False

    name_candidate = None
    for ln in lines[:6]:
        if looks_like_name(ln):
            name_candidate = ln
            break
    parsed["name"] = name_candidate

    # Title
    if parsed["name"]:
        try:
            idx = lines.index(parsed["name"])
            for ln in lines[idx + 1: idx + 5]:
                if re.search(r"\b(Founder|CEO|Director|Manager|Partner|Consultant|Officer|Advisory|Placement|Head|Chief|Executive|Officer|Placement Officer)\b", ln, re.I):
                    parsed["title"] = ln
                    break
        except ValueError:
            pass

    # Company
    company_candidate = None
    for ln in lines:
        if re.search(r"\b(Ltd|Pvt|Private|LLP|Limited|Inc|Corporation|Company|Technologies|Tech|Solutions|Works|Consultants|Advisory|Group|Systems|Enterprises|Industries)\b", ln, re.I):
            company_candidate = ln
            break
    if not company_candidate and parsed["name"]:
        try:
            idx = lines.index(parsed["name"])
            for cand in lines[idx + 1: idx + 3]:
                if len(cand.split()) <= 6 and not EMAIL_RE.search(cand) and not PHONE_RE.search(cand):
                    company_candidate = cand
                    break
        except ValueError:
            pass
    parsed["company"] = company_candidate

    # Address guess
    addresses = []
    addr_tokens = ("road", "rd", "street", "st", "bengaluru", "bangalore", "kolhapur", "mumbai", "coimbatore", "address", "city", "block", "floor", "lane", "near", "plot", "sector", "outer ring")
    for ln in lines[-8:]:
        low = ln.lower()
        if len(ln) > 30 or any(tok in low for tok in addr_tokens) or re.search(r"\b\d{5,6}\b", ln):
            addresses.append(ln)
    if addresses:
        parsed["address"] = " | ".join(addresses[:3])
    else:
        for ln in lines:
            if re.search(r"\b\d{5,6}\b", ln):
                parsed["address"] = ln
                break

    notes = []
    if parsed.get("email"):
        notes.append("email_ok")
    if parsed.get("phone"):
        notes.append("phone_ok")
    if parsed.get("website"):
        notes.append("website_ok")
    if parsed.get("name"):
        notes.append("name_guess")
    if parsed.get("company"):
        notes.append("company_guess")
    if parsed.get("address"):
        notes.append("address_guess")
    if not notes:
        notes = ["no_fields_parsed_locally"]

    parsed["confidence_notes"] = ";".join(notes)
    return parsed, parsed["confidence_notes"]

# -----------------------
# OCR Preprocessing + Multi-PSM logic
# -----------------------
def preprocess_image_for_ocr(image: Image.Image, upscale_to: int = 1600, do_deskew: bool = True) -> Image.Image:
    """
    Convert to grayscale, upscale if small, denoise, sharpen, simple adaptive threshold and optional deskew via tesseract OSD.
    """
    img = image.convert("L")

    # upscale if small
    try:
        max_dim = max(img.size)
        if max_dim < upscale_to:
            ratio = upscale_to / max_dim
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
    except Exception:
        pass

    # median denoise and mild sharpening
    try:
        img = img.filter(ImageFilter.MedianFilter(size=3))
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
    except Exception:
        pass

    # adaptive-like threshold via numpy (fast & crude)
    try:
        arr = np.array(img).astype(np.uint8)
        mean = int(arr.mean())
        # Slight offset to preserve lighter text
        thresh = mean - 10
        arr = np.where(arr > thresh, 255, 0).astype(np.uint8)
        img = Image.fromarray(arr)
    except Exception:
        pass

    # Optional deskew using tesseract OSD if available
    if do_deskew:
        try:
            osd = pytesseract.image_to_osd(img)
            m = re.search(r"Rotate:\s+(\d+)", osd)
            if m:
                rot = int(m.group(1))
                if rot and rot % 360 != 0:
                    img = img.rotate(360 - rot, expand=True)
        except Exception:
            pass

    return img

def ocr_try_multiple_psm(img: Image.Image, psm_choices=(6, 3, 11)) -> Tuple[str, dict]:
    """
    Try several Tesseract PSMs and return the best text with metadata.
    psm_choices: try order; returns best by mean word confidence.
    """
    best = {"text": "", "mean_conf": -999.0, "raw": None, "psm": None, "data": None}
    for psm in psm_choices:
        config = f"--psm {psm} --oem 3"
        try:
            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
            words = []
            confs = []
            for t, c in zip(data.get("text", []), data.get("conf", [])):
                if t and str(t).strip():
                    words.append(str(t).strip())
                    try:
                        cnum = float(c)
                    except Exception:
                        cnum = -1.0
                    confs.append(cnum)
            text = " ".join(words).strip()
            mean_conf = float(np.mean([c for c in confs if c >= 0])) if any(c >= 0 for c in confs) else -1.0
            # choose higher mean_conf, tie-breaker: longer text
            if mean_conf > best["mean_conf"] or (abs(mean_conf - best["mean_conf"]) < 1e-6 and len(text) > len(best["text"])):
                best.update({"text": text, "mean_conf": mean_conf, "raw": data, "psm": psm, "data": data})
        except Exception:
            # fallback to simpler API if image_to_data fails
            try:
                txt = pytesseract.image_to_string(img, config=config)
                txt = txt.strip()
                if len(txt) > len(best["text"]):
                    best.update({"text": txt, "mean_conf": -1.0, "raw": None, "psm": psm, "data": None})
            except Exception:
                pass

    # Fallback: if nothing returned, try default image_to_string
    if not best["text"]:
        try:
            txt = pytesseract.image_to_string(img)
            best.update({"text": txt.strip(), "mean_conf": -1.0, "raw": None, "psm": None, "data": None})
        except Exception:
            pass

    return best["text"], {"mean_conf": best["mean_conf"], "psm": best["psm"], "raw": best["raw"]}

# -----------------------
# OpenAI parsing wrapper (required)
# -----------------------
PARSER_PROMPT = (
    "You are an assistant that extracts structured contact fields from messy OCR'd text from a business card.\n"
    "Return a JSON object with keys exactly: name, company, title, email, phone, phone_numbers, website, address, social_links, extra, confidence_notes.\n"
    "Rules:\n"
    " - phone_numbers must be a JSON list of phone strings (can be empty list if none).\n"
    " - social_links must be a JSON list of URLs or handles (empty list if none).\n"
    " - phone may be the primary single phone string or null.\n"
    " - If a field is not present, set it to null (or an empty list for phone_numbers/social_links).\n"
    " - For 'extra' include any other useful strings (fax, linkedin, notes) as an object.\n"
    " - Respond with JSON only, no surrounding text.\n"
    "Example:\n"
    '{"name": "Alice Doe", "company": "Acme Pvt Ltd", "title": "CTO", "email": "alice@acme.com", "phone": null, "phone_numbers": ["+911234567890"], "website": "https://acme.com", "address": "123, Road, City", "social_links": ["https://www.linkedin.com/in/alicedoe"], "extra": {}, "confidence_notes": "email_ok;phone_ok"}\n'
)

def call_openai_parse(ocr_text: str, api_key: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Calls OpenAI (via OpenAI Python client) to parse OCR text into structured JSON.
    Raises RuntimeError if api_key is missing.
    Returns a dict with expected keys (phone_numbers and social_links ensured as lists).
    """
    if not api_key:
        raise RuntimeError("OpenAI API key is required for parsing.")
    client = OpenAI(api_key=api_key)
    prompt = PARSER_PROMPT + "\nOCR_TEXT:\n" + ocr_text + "\n\nRespond with JSON only."

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a JSON-only extractor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as e:
        logger.exception("OpenAI API call failed")
        return {
            "name": None,
            "company": None,
            "title": None,
            "email": None,
            "phone": None,
            "phone_numbers": [],
            "website": None,
            "address": None,
            "social_links": [],
            "extra": {"openai_error": str(e), "traceback": traceback.format_exc()},
            "confidence_notes": f"OpenAI call failed: {e}"
        }

    try:
        assistant_text = resp.choices[0].message.content.strip()
    except Exception:
        assistant_text = str(resp)

    try:
        parsed = json.loads(assistant_text)
        # normalize presence of lists
        if "phone_numbers" not in parsed:
            if parsed.get("phone"):
                parsed["phone_numbers"] = _ensure_list(parsed.get("phone"))
            else:
                parsed["phone_numbers"] = []
        if "social_links" not in parsed:
            parsed["social_links"] = parsed.get("social_links") or []
        return parsed
    except Exception:
        # try to extract JSON object at end of text
        m = re.search(r"\{[\s\S]*\}$", assistant_text)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if "phone_numbers" not in parsed:
                    parsed["phone_numbers"] = parsed.get("phone") and _ensure_list(parsed.get("phone")) or []
                if "social_links" not in parsed:
                    parsed["social_links"] = parsed.get("social_links") or []
                return parsed
            except Exception:
                pass

        return {
            "name": None,
            "company": None,
            "title": None,
            "email": None,
            "phone": None,
            "phone_numbers": [],
            "website": None,
            "address": None,
            "social_links": [],
            "extra": {"model_output": assistant_text},
            "confidence_notes": "Model output not parseable as JSON. See extra.model_output."
        }

# -----------------------
# FastAPI app & endpoints
# -----------------------
app = FastAPI(title="Business Card OCR Backend (OpenAI required + MongoDB)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContactBase(BaseModel):
    name: Optional[str] = None
    designation: Optional[str] = None
    company: Optional[str] = None
    phone_numbers: Optional[List[str]] = []
    email: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    social_links: Optional[List[str]] = []
    more_details: Optional[str] = ""
    additional_notes: Optional[str] = ""

class ExtractedContact(ContactBase):
    raw_text: Optional[str] = None
    confidence_notes: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

@app.get("/ping")
async def ping():
    return {"status": "ok", "time": now_iso()}

@app.post("/extract", response_model=ExtractedContact)
async def extract_card(file: UploadFile = File(...), authorization: Optional[str] = Header(None), model: Optional[str] = "gpt-4o"):
    """
    Upload image -> preprocess -> OCR (multi-psm, rotation fallback) -> local parse -> OpenAI parse (required) -> merge -> return structured fields.
    """
    # require api key via header or env
    api_key = None
    if authorization and authorization.lower().startswith("bearer "):
        api_key = authorization.split(" ", 1)[1].strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(status_code=401, detail="OpenAI API key required. Provide via Authorization: Bearer <KEY> header or OPENAI_API_KEY environment variable.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png/etc)")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.exception("Invalid image uploaded")
        return JSONResponse(status_code=400, content={"detail": f"Invalid image: {e}", "traceback": traceback.format_exc()})

    # Preprocess image (deskew, enhance, threshold, upscale)
    try:
        pre = preprocess_image_for_ocr(image, upscale_to=1600, do_deskew=True)
    except Exception:
        pre = image.convert("L")

    # Try multiple Tesseract PSMs and pick best by mean confidence
    ocr_text, ocr_meta = ocr_try_multiple_psm(pre, psm_choices=(6, 3, 11))

    # If OCR text is short or low confidence, try additional rotations (portrait / transposed)
    try:
        if (not ocr_text or len(ocr_text.strip()) < 20) or (ocr_meta.get("mean_conf", -1) != -1 and ocr_meta.get("mean_conf") < 40):
            for rot in (90, 270):
                try:
                    rot_img = pre.rotate(rot, expand=True)
                    ttext, tmeta = ocr_try_multiple_psm(rot_img, psm_choices=(6, 3, 11))
                    if len(ttext) > len(ocr_text) and (tmeta.get("mean_conf", -1) > ocr_meta.get("mean_conf", -1)):
                        ocr_text = ttext
                        ocr_meta = tmeta
                except Exception:
                    pass
    except Exception:
        pass

    raw_text_clean = clean_ocr_text(ocr_text)

    # local parse (always)
    local_parsed, local_notes = local_parse_from_ocr(raw_text_clean)

    # REQUIRED: call OpenAI parser (send the cleaned OCR text)
    openai_parsed = call_openai_parse(raw_text_clean, api_key=api_key, model=model)

    # Merge: prefer OpenAI value if present and non-empty, otherwise local
    merged = {}
    def pick(key):
        v = openai_parsed.get(key)
        if v is not None and v != "" and v != []:
            return v
        return local_parsed.get(key)

    merged["name"] = pick("name")
    merged["designation"] = pick("title") or pick("designation")
    merged["company"] = pick("company")

    # Phones: prioritize OpenAI phone_numbers (list); otherwise local extras
    phones_candidate = None
    if openai_parsed.get("phone_numbers"):
        phones_candidate = openai_parsed.get("phone_numbers")
    elif openai_parsed.get("phone"):
        phones_candidate = _ensure_list(openai_parsed.get("phone"))
    else:
        phones_candidate = local_parsed.get("extra", {}).get("phones_all") or local_parsed.get("phone")

    merged["phone_numbers"] = _ensure_list(phones_candidate)

    merged["email"] = pick("email")
    merged["website"] = pick("website")
    merged["address"] = pick("address")

    social_candidate = None
    if openai_parsed.get("social_links"):
        social_candidate = openai_parsed.get("social_links")
    else:
        social_candidate = local_parsed.get("extra", {}).get("linkedin") or local_parsed.get("extra", {}).get("websites_all")
    merged["social_links"] = _ensure_list(social_candidate)

    merged["more_details"] = openai_parsed.get("more_details") or local_parsed.get("more_details") or ""
    merged["additional_notes"] = openai_parsed.get("additional_notes") or local_parsed.get("additional_notes") or ""
    merged["raw_text"] = raw_text_clean

    cn = []
    if openai_parsed.get("confidence_notes"):
        cn.append(f"openai:{openai_parsed.get('confidence_notes')}")
    if local_parsed.get("confidence_notes"):
        cn.append(f"local:{local_parsed.get('confidence_notes')}")
    # include OCR meta confidence for diagnostics
    cn.append(f"ocr_mean_conf:{int(ocr_meta.get('mean_conf', -1))}")
    if ocr_meta.get("psm"):
        cn.append(f"ocr_psm:{ocr_meta.get('psm')}")
    merged["confidence_notes"] = ";".join(cn) if cn else "none"

    merged["extra"] = {
        "local": local_parsed.get("extra", {}),
        "openai": openai_parsed.get("extra", {}),
        "ocr_meta": ocr_meta,
    }

    return JSONResponse(status_code=200, content=merged)

@app.post("/vcard")
async def vcard_endpoint(payload: ContactBase = Body(...)):
    payload_dict = payload.dict()
    phone = payload_dict.get("phone_numbers", [None])[0] if payload_dict.get("phone_numbers") else None
    vcard_data = {
        "name": payload_dict.get("name"),
        "company": payload_dict.get("company"),
        "title": payload_dict.get("designation"),
        "phone": phone,
        "email": payload_dict.get("email"),
        "website": payload_dict.get("website"),
        "address": payload_dict.get("address"),
    }
    vcard = generate_vcard(vcard_data)
    return StreamingResponse(io.BytesIO(vcard.encode("utf-8")),
                             media_type="text/vcard",
                             headers={"Content-Disposition": "attachment; filename=contact.vcf"})

@app.post("/create_card", status_code=status.HTTP_201_CREATED)
async def create_card(payload: ContactBase = Body(...)):
    try:
        doc = normalize_payload(payload.dict())
        doc["created_at"] = now_iso()
        doc["edited_at"] = ""
        doc.setdefault("field_validations", {})

        # dedupe by email
        if doc.get("email"):
            existing = collection.find_one({"email": doc["email"]})
            if existing:
                if not doc.get("more_details"):
                    doc["more_details"] = existing.get("more_details", "")
                doc["edited_at"] = now_iso()
                collection.update_one({"_id": existing["_id"]}, {"$set": doc})
                updated = collection.find_one({"_id": existing["_id"]})
                return {"message": "Updated existing contact", "data": db_doc_to_canonical(updated)}

        result = collection.insert_one(doc)
        inserted = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted Successfully", "data": db_doc_to_canonical(inserted)}
    except Exception as e:
        logger.exception("create_card error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/all_cards")
def get_all_cards():
    try:
        docs = list(collection.find().sort([("_id", -1)]))
        canonical_list = [db_doc_to_canonical(d) for d in docs]
        return {"data": canonical_list}
    except Exception as e:
        logger.exception("get_all_cards error")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/update_notes/{card_id}")
def update_notes(card_id: str, payload: dict = Body(...)):
    try:
        ts = now_iso()
        update_payload = {
            "additional_notes": payload.get("additional_notes", ""),
            "edited_at": ts
        }
        collection.update_one({"_id": ObjectId(card_id)}, {"$set": update_payload})
        updated = collection.find_one({"_id": ObjectId(card_id)})
        return {"message": "Updated", "data": db_doc_to_canonical(updated)}
    except Exception as e:
        logger.exception("update_notes error")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/update_card/{card_id}")
def update_card(card_id: str, payload: dict = Body(...)):
    """
    Safer update: normalize payload and only $set fields that are intentionally provided.
    Empty lists are respected (they clear fields). None/empty strings are ignored unless user explicitly sets.
    """
    try:
        allowed_fields = {
            "name", "designation", "company", "phone_numbers",
            "email", "website", "address", "social_links",
            "additional_notes", "more_details"
        }
        normalized = normalize_payload(payload)
        # Keep fields that are allowed and not None/empty-string.
        # Note: we DO allow empty lists for phone_numbers/social_links to clear them intentionally.
        update_data: Dict[str, Any] = {}
        for k, v in normalized.items():
            if k not in allowed_fields:
                continue
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            # Accept empty lists
            update_data[k] = v

        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields to update.")

        existing = collection.find_one({"_id": ObjectId(card_id)})
        if not existing:
            raise HTTPException(status_code=404, detail="Card not found.")

        update_data["edited_at"] = now_iso()
        collection.update_one({"_id": ObjectId(card_id)}, {"$set": update_data})
        updated = collection.find_one({"_id": ObjectId(card_id)})
        return {"message": "Updated", "data": db_doc_to_canonical(updated)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("update_card error")
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
        logger.exception("delete_card error")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# vCard helper
# -----------------------
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
