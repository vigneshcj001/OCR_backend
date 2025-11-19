"""
Merged FastAPI app (updated):
 - /extract        : multipart image -> Tesseract OCR -> OpenAI chat completions to extract structured contact fields.
 - /upload_card    : now reuses /extract flow (OCR -> OpenAI parse), normalizes result, and saves to MongoDB (or in-memory).
 - /vcard, /create_card, /all_cards, /update_notes, /update_card, /delete_card, /ping, / root remain.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Body, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import io
import os
import re
import json
import base64
import logging
import pytesseract
from datetime import datetime
import pytz

# OpenAI v1 client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Optional MongoDB
try:
    from pymongo import MongoClient
    from bson import ObjectId, json_util
except Exception:
    MongoClient = None
    ObjectId = None
    json_util = None

# Configure Tesseract path from env if provided (Windows)
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------- Helpers ----------
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

def call_openai_parse(ocr_text: str, api_key: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Calls OpenAI chat completions style (OpenAI v1 client). Returns parsed JSON dict (best effort).
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed (pip install openai)")

    client = OpenAI(api_key=api_key)
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

    # parse JSON or fallback
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

# small helper to get API key from header or env
def get_api_key_from_header_or_env(authorization: Optional[str]) -> Optional[str]:
    api_key = None
    if authorization:
        if authorization.lower().startswith("bearer "):
            api_key = authorization.split(" ", 1)[1].strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    return api_key

# Minimal "classify_contact" placeholder (replace with real logic)
def classify_contact(d: Dict[str, Any]) -> Dict[str, Any]:
    # Here you can add phone/email normalization, validation flags, entity detection, etc.
    if "phone_numbers" not in d:
        if d.get("phone"):
            d["phone_numbers"] = [d.get("phone")]
        else:
            d["phone_numbers"] = []
    # keep compatibility: ensure strings exist
    for k in ["name", "designation", "company", "email", "website", "address", "more_details", "additional_notes"]:
        if k not in d:
            d[k] = "" if k != "phone_numbers" else []
    if "social_links" not in d:
        d["social_links"] = []
    return d

# Time helper using IST (Asia/Kolkata)
def now_ist() -> str:
    tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(tz).isoformat()

# JSON encoder helpers for BSON -> JSON (Mongo)
class JSONEncoder:
    @staticmethod
    def encode(doc):
        if json_util:
            return json.loads(json_util.dumps(doc))
        try:
            return json.loads(json.dumps(doc, default=str))
        except Exception:
            return str(doc)

# ---------- FastAPI app ----------
app = FastAPI(title="Business Card OCR Backend (Merged)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Optional DB setup ----------
MONGO_URI = os.getenv("MONGO_URI")
collection = None
if MONGO_URI and MongoClient is not None:
    try:
        client = MongoClient(MONGO_URI)
        db = client.get_default_database() or client["businesscards"]
        collection = db.get_collection("contacts")
        logging.info("Connected to MongoDB collection for contacts.")
    except Exception:
        logging.exception("Failed to connect to MongoDB. Falling back to in-memory store.")
        collection = None

# fallback in-memory store (list of dicts)
_in_memory_store = []
def _insert_in_memory(doc):
    doc = dict(doc)
    doc["_id"] = str(len(_in_memory_store) + 1)
    _in_memory_store.append(doc)
    return doc
def _find_all_in_memory():
    return list(reversed(_in_memory_store))
def _find_one_in_memory_by_id(_id):
    for d in _in_memory_store:
        if str(d.get("_id")) == str(_id):
            return d
    return None
def _delete_in_memory(_id):
    global _in_memory_store
    before = len(_in_memory_store)
    _in_memory_store = [d for d in _in_memory_store if str(d.get("_id")) != str(_id)]
    return before - len(_in_memory_store)

# ---------- Pydantic models ----------
class ExtractedContact(BaseModel):
    name: Optional[str]
    company: Optional[str]
    title: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    website: Optional[str]
    address: Optional[str]
    raw_text: Optional[str]
    confidence_notes: Optional[str]
    extra: Optional[Dict[str, Any]]

class VCardRequest(BaseModel):
    name: Optional[str]
    company: Optional[str]
    title: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    website: Optional[str]
    address: Optional[str]

class ContactCreate(BaseModel):
    name: Optional[str] = ""
    designation: Optional[str] = ""
    company: Optional[str] = ""
    phone_numbers: Optional[List[str]] = Field(default_factory=list)
    email: Optional[str] = ""
    website: Optional[str] = ""
    address: Optional[str] = ""
    social_links: Optional[List[str]] = Field(default_factory=list)
    more_details: Optional[str] = ""
    additional_notes: Optional[str] = ""

# ---------- Shared image-processing helper ----------
def process_image_bytes(content_bytes: bytes, content_type: str, api_key: str, model: str = "gpt-4o") -> Tuple[Dict[str, Any], str]:
    """
    Shared helper: open image from bytes, optionally resize, run Tesseract OCR, then call OpenAI parser.
    Returns (parsed_dict, raw_text).
    """
    try:
        image = Image.open(io.BytesIO(content_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image: {e}")

    # optional resize for very large images
    max_dim = 1800
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0]*ratio), int(image.size[1]*ratio))
        image = image.resize(new_size)

    try:
        raw_text = pytesseract.image_to_string(image)
    except Exception as e:
        raise RuntimeError(f"OCR failure: {e}")

    raw_text = clean_ocr_text(raw_text)
    parsed = call_openai_parse(raw_text, api_key=api_key, model=model)
    return parsed, raw_text

# ---------- Routes (merged) ----------
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "OCR Backend Running ✅ (OpenAI Vision)"}

@app.post("/extract", response_model=ExtractedContact)
async def extract_card(file: UploadFile = File(...), authorization: Optional[str] = Header(None), model: Optional[str] = "gpt-4o"):
    api_key = get_api_key_from_header_or_env(authorization)
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not provided. Set OPENAI_API_KEY or provide Authorization: Bearer <key> header")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        parsed, raw_text = process_image_bytes(contents, file.content_type, api_key, model=model)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as rexc:
        raise HTTPException(status_code=500, detail=str(rexc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = {
        "name": parsed.get("name"),
        "company": parsed.get("company"),
        "title": parsed.get("title"),
        "email": parsed.get("email"),
        "phone": parsed.get("phone"),
        "website": parsed.get("website"),
        "address": parsed.get("address"),
        "raw_text": raw_text,
        "confidence_notes": parsed.get("confidence_notes"),
        "extra": parsed.get("extra"),
    }
    return JSONResponse(status_code=200, content=result)

@app.post("/vcard")
async def vcard_endpoint(payload: VCardRequest = Body(...)):
    data = payload.dict()
    vcard = generate_vcard(data)
    # return as downloadable text stream
    return StreamingResponse(io.BytesIO(vcard.encode("utf-8")), media_type="text/vcard", headers={"Content-Disposition": "attachment; filename=contact.vcf"})

# ------------------------------
# Upload route: now reuses OCR+parse and saves to DB
# ------------------------------
@app.post("/upload_card", status_code=status.HTTP_201_CREATED)
async def upload_card(file: UploadFile = File(...), authorization: Optional[str] = Header(None), model: Optional[str] = "gpt-4o"):
    """
    New behavior:
      - Uses the shared OCR+OpenAI parsing flow (same as /extract).
      - Normalizes parsed output into DB schema and persists (MongoDB if configured, else in-memory).
    """
    api_key = get_api_key_from_header_or_env(authorization)
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not provided. Set OPENAI_API_KEY or provide Authorization: Bearer <key> header")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()

    # Process image (shared code)
    try:
        parsed, raw_text = process_image_bytes(contents, file.content_type, api_key, model=model)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as rexc:
        raise HTTPException(status_code=500, detail=str(rexc))
    except Exception as e:
        logging.exception("upload_card processing error")
        raise HTTPException(status_code=500, detail=str(e))

    # Normalize fields into DB document format used by create_card/upload_card flows
    doc = {
        "name": parsed.get("name") or "",
        "designation": parsed.get("title") or "",
        "company": parsed.get("company") or "",
        # phone_numbers as list (if model returned single 'phone', turn into list)
        "phone_numbers": ( [parsed.get("phone")] if parsed.get("phone") else [] ),
        "email": parsed.get("email") or "",
        "website": parsed.get("website") or "",
        "address": parsed.get("address") or "",
        # try to pluck social links from 'extra' if available (best-effort)
        "social_links": [],
        "more_details": "",
        "additional_notes": parsed.get("confidence_notes") or "",
        # keep raw info for debugging
        "_openai_parsed": parsed,
        "_ocr_raw_text": raw_text,
        "_processed_at": now_ist(),
        "_openai_model_used": model
    }

    # If 'extra' contains common handles or links, append to social_links
    extra = parsed.get("extra")
    if isinstance(extra, dict):
        # look for linkedin/twitter/telegram/instagram/whatsapp keys or URLs
        for v in extra.values():
            if isinstance(v, str):
                if "linkedin.com" in v or "twitter.com" in v or "instagram.com" in v or "wa.me" in v or "telegram.me" in v:
                    doc["social_links"].append(v)
    elif isinstance(extra, str):
        if "http" in extra:
            doc["social_links"].append(extra)

    doc = classify_contact(doc)

    # persist
    try:
        if collection:
            result = collection.insert_one(doc)
            inserted = collection.find_one({"_id": result.inserted_id})
            out = JSONEncoder.encode(inserted)
        else:
            inserted = _insert_in_memory(doc)
            out = inserted
    except Exception as e:
        logging.exception("upload_card DB insert error")
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Inserted Successfully", "data": out}

# ------------------------------
# Create card route (manual creation)
# ------------------------------
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
            # attempt uniqueness by email
            existing = None
            if collection:
                existing = collection.find_one({"email": doc["email"]})
            else:
                for d in _in_memory_store:
                    if d.get("email") == doc["email"]:
                        existing = d
                        break
            if existing:
                if not doc.get("more_details"):
                    doc["more_details"] = existing.get("more_details", "")
                doc["edited_at"] = now_ist()
                if collection:
                    collection.update_one({"_id": existing["_id"]}, {"$set": doc})
                    updated = collection.find_one({"_id": existing["_id"]})
                    return {"message": "Updated existing contact", "data": JSONEncoder.encode(updated)}
                else:
                    # update in-memory
                    existing.update(doc)
                    return {"message": "Updated existing contact", "data": existing}
        # insert new
        if collection:
            result = collection.insert_one(doc)
            inserted = collection.find_one({"_id": result.inserted_id})
            return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}
        else:
            inserted = _insert_in_memory(doc)
            return {"message": "Inserted Successfully", "data": inserted}
    except Exception as e:
        logging.exception("create_card error")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# List all cards
# ------------------------------
@app.get("/all_cards")
def get_all_cards():
    try:
        if collection:
            docs = list(collection.find().sort([("_id", -1)]))
            return {"data": JSONEncoder.encode(docs)}
        else:
            return {"data": _find_all_in_memory()}
    except Exception as e:
        logging.exception("get_all_cards error")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# Update notes
# ------------------------------
@app.put("/update_notes/{card_id}")
def update_notes(card_id: str, payload: dict = Body(...)):
    try:
        ts = now_ist()
        update_payload = {
            "additional_notes": payload.get("additional_notes", ""),
            "edited_at": ts
        }
        if collection and ObjectId:
            collection.update_one({"_id": ObjectId(card_id)}, {"$set": update_payload})
            updated = collection.find_one({"_id": ObjectId(card_id)})
            return {"message": "Updated", "data": JSONEncoder.encode(updated)}
        else:
            existing = _find_one_in_memory_by_id(card_id)
            if not existing:
                raise HTTPException(status_code=404, detail="Card not found.")
            existing.update(update_payload)
            return {"message": "Updated", "data": existing}
    except Exception as e:
        logging.exception("update_notes error")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# Patch update card
# ------------------------------
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

        if collection and ObjectId:
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
        else:
            existing = _find_one_in_memory_by_id(card_id)
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
            merged["edited_at"] = now_ist()
            # update in-memory store
            for i, d in enumerate(_in_memory_store):
                if str(d.get("_id")) == str(card_id):
                    _in_memory_store[i].update(merged)
                    return {"message": "Updated", "data": _in_memory_store[i]}
            raise HTTPException(status_code=404, detail="Card not found.")
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("update_card error")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# Delete card
# ------------------------------
@app.delete("/delete_card/{card_id}", status_code=status.HTTP_200_OK)
def delete_card(card_id: str):
    try:
        if collection and ObjectId:
            result = collection.delete_one({"_id": ObjectId(card_id)})
            if result.deleted_count == 1:
                return {"message": "Deleted"}
            else:
                raise HTTPException(status_code=404, detail="Card not found.")
        else:
            deleted = _delete_in_memory(card_id)
            if deleted == 1:
                return {"message": "Deleted"}
            else:
                raise HTTPException(status_code=404, detail="Card not found.")
    except Exception as e:
        logging.exception("delete_card error")
        raise HTTPException(status_code=500, detail=str(e))
