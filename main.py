# app.py
from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract
import io
import os
import re
from dotenv import load_dotenv
from datetime import datetime
import pytz
import numpy as np

# Optional: use OpenCV if available for deskew (not required)
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# -----------------------------------------------------
# Load env / Mongo
# -----------------------------------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "business_cards")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contacts")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# -----------------------------------------------------
# FastAPI setup
# -----------------------------------------------------
app = FastAPI(title="Business Card OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# Utilities
# -----------------------------------------------------
class JSONEncoder:
    """Encode ObjectId and nested structures to JSON-serializable values."""
    @staticmethod
    def encode(obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, dict):
            return {k: JSONEncoder.encode(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [JSONEncoder.encode(x) for x in obj]
        return obj

def to_ist_now():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------------------------------
# Image preprocessing helpers
# -----------------------------------------------------
def pil_to_cv(img_pil):
    """Convert PIL Image to OpenCV image (BGR)."""
    arr = np.array(img_pil)
    if arr.ndim == 2:
        return arr
    # RGB to BGR
    return arr[:, :, ::-1].copy()

def cv_to_pil(img_cv):
    """Convert OpenCV image (BGR or gray) to PIL."""
    if img_cv.ndim == 2:
        return Image.fromarray(img_cv)
    # BGR to RGB
    return Image.fromarray(img_cv[:, :, ::-1])

def deskew_with_cv(np_img):
    """Attempt to deskew grayscale numpy image using OpenCV. Returns deskewed grayscale array."""
    if not OPENCV_AVAILABLE:
        return np_img
    try:
        img = np_img.copy()
        coords = cv2.findNonZero(cv2.bitwise_not(img))
        if coords is None:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return np_img

def preprocess_for_ocr(pil_img: Image.Image, strong_binarize: bool = False):
    """
    Preprocess PIL image to improve OCR:
      - Convert to grayscale
      - Optionally deskew (if OpenCV present)
      - Enhance contrast and sharpen
      - Binarize (adaptive if requested)
    Returns a PIL image ready for pytesseract.
    """
    img = pil_img.convert("RGB")
    # Resize small images to improve OCR
    max_side = max(img.size)
    if max_side < 1200:
        scale = int(1200 / max_side)
        new_size = (img.width * scale, img.height * scale)
        img = img.resize(new_size, Image.LANCZOS)

    gray = ImageOps.grayscale(img)
    # convert to numpy for optional OpenCV deskew
    np_img = np.array(gray)

    if OPENCV_AVAILABLE:
        try:
            np_img = deskew_with_cv(np_img)
        except Exception:
            pass

    # Back to PIL
    gray = Image.fromarray(np_img)

    # Increase contrast and sharpen
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(1.5)
    gray = gray.filter(ImageFilter.SHARPEN)

    if strong_binarize:
        # Use simple threshold
        bw = gray.point(lambda p: 255 if p > 150 else 0)
        return bw.convert("L")
    else:
        return gray

# -----------------------------------------------------
# Extraction logic (adapted from user's original)
# -----------------------------------------------------
def extract_details(text: str):
    """
    Heuristic extraction from OCR text.
    Returns a dict: name, designation, company, phone_numbers, email, website, address, social_links, additional_notes
    """
    # Normalize whitespace and split into lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
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
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)
    data["email"] = email_match.group(0) if email_match else ""

    # WEBSITE
    website_match = re.search(r"(https?://\S+|www\.\S+|\S+\.(com|in|net|org|co))", raw_text)
    data["website"] = website_match.group(0) if website_match else ""

    # PHONE NUMBERS: capture several formats, including +91, spaces and dashes
    phone_matches = re.findall(r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}", raw_text)
    # Postprocess: remove false matches (like years or 3-digit codes) shorter than 7 digits
    phones = []
    for p in phone_matches:
        digits = re.sub(r"\D", "", p)
        if len(digits) >= 7:   # allow 7+ digit phone numbers
            phones.append(re.sub(r"\s+", " ", p).strip())
    data["phone_numbers"] = list(dict.fromkeys(phones))  # unique preserve order

    # SOCIAL LINKS / LINKEDIN
    for l in lines:
        if "linkedin" in l.lower() or "in/" in l.lower():
            data["social_links"].append(l.strip())

    # DESIGNATION: look for keywords on lines
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager", "general manager",
        "director", "engineer", "consultant", "head", "lead", "marketing",
        "executive", "founder & ceo", "ceo & founder"
    ]
    for line in lines:
        low = line.lower()
        for kw in designation_keywords:
            if kw in low:
                data["designation"] = re.sub(r"[^A-Za-z &]", "", line).strip()
                break
        if data["designation"]:
            break

    # COMPANY: using common corporate tokens or big-capitalized lines
    for line in lines:
        if re.search(r"\b(pvt|private|ltd|llp|inc|corporation|company|works|restaurant|airlines|insurance|technologies|digital|real estate|global|sun|ceiyone)\b", line, re.I):
            data["company"] = line.strip()
            break
    # fallback: very large uppercase line that isn't name
    if not data["company"]:
        for line in lines:
            clean = re.sub(r"[^A-Za-z ]", "", line).strip()
            if len(clean) > 3 and clean.replace(" ", "").isupper() and len(clean.split()) <= 3:
                # if it's not the name (we'll pick name elsewhere), treat it as company possibility
                data["company"] = line.strip()
                break

    # NAME extraction: prefer multi-line uppercase or line above designation
    uppercase_lines = []
    # gather candidate lines (cleaned)
    for l in lines:
        clean = re.sub(r"[^A-Za-z ]", "", l).strip()
        if not clean:
            continue
        # exclude email/website
        if "@" in l or "www" in l.lower():
            continue
        alpha_ratio = len(re.findall(r"[A-Za-z]", clean)) / max(1, len(clean))
        if alpha_ratio < 0.6:
            continue
        if clean.replace(" ", "").isupper():
            uppercase_lines.append(clean)

    if len(uppercase_lines) >= 2:
        # Join up to first two uppercase lines (handles split last/first names)
        data["name"] = " ".join(uppercase_lines[:2])
    elif uppercase_lines:
        data["name"] = uppercase_lines[0]

    # fallback: name is the line immediately above a detected designation
    if not data["name"] and data["designation"]:
        for i, line in enumerate(lines):
            if data["designation"].lower() in line.lower() and i > 0:
                candidate = re.sub(r"[^A-Za-z ]", "", lines[i - 1]).strip()
                if candidate:
                    data["name"] = candidate
                    break

    # final fallback: first reasonably long line that isn't company/website/email
    if not data["name"]:
        for line in lines:
            low = line.lower()
            if data["company"] and data["company"].lower() in low:
                continue
            if "@" in line or "www" in low:
                continue
            candidate = re.sub(r"[^A-Za-z ]", "", line).strip()
            if len(candidate.split()) >= 1 and 2 <= len(candidate) <= 50:
                data["name"] = candidate
                break

    # ADDRESS: look for keywords and/or sequences with numbers and known location words
    address_lines = []
    address_indicators = ["street", "st", "road", "rd", "nagar", "lane", "city", "coimbatore", "tamil", "india", "chennai", "salem", "erode", r"\d{6}"]
    for l in lines:
        if any(re.search(ind, l, re.I) for ind in address_indicators):
            address_lines.append(l.strip())
    if address_lines:
        data["address"] = ", ".join(address_lines)

    return data

# -----------------------------------------------------
# Routes
# -----------------------------------------------------
@app.get("/")
def root():
    return {"message": "OCR Backend Running ✅"}

@app.post("/upload_card")
async def upload_card(file: UploadFile = File(...), strong_binarize: bool = False):
    """
    Upload a single business card image. Returns extracted data and inserts into MongoDB.
    Query param strong_binarize toggles a stronger binarization step (boolean).
    """
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open image: {e}")

    # Preprocess
    proc = preprocess_for_ocr(img, strong_binarize=strong_binarize)

    # Use Tesseract configuration: treat as single uniform block, keep char whitelist? (left default)
    try:
        text = pytesseract.image_to_string(proc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    extracted = extract_details(text)
    extracted["additional_notes"] = extracted.get("additional_notes", "")  # ensure present
    extracted["created_at"] = to_ist_now()

    # Insert into DB
    try:
        res = collection.insert_one(extracted)
        inserted = collection.find_one({"_id": res.inserted_id})
        return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

@app.post("/upload_multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    """
    Upload multiple business card images (multipart form-data, multiple files).
    Returns array of inserted documents.
    """
    results = []
    for file in files:
        try:
            content = await file.read()
            img = Image.open(io.BytesIO(content))
            proc = preprocess_for_ocr(img)
            text = pytesseract.image_to_string(proc)
            extracted = extract_details(text)
            extracted["created_at"] = to_ist_now()
            res = collection.insert_one(extracted)
            inserted = collection.find_one({"_id": res.inserted_id})
            results.append(JSONEncoder.encode(inserted))
        except Exception as e:
            results.append({"file": getattr(file, "filename", "unknown"), "error": str(e)})
    return {"inserted": results}

@app.get("/all_cards")
def get_all_cards(limit: int = 100):
    try:
        docs = list(collection.find().sort("created_at", -1).limit(limit))
        return {"data": JSONEncoder.encode(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=400, detail=str(e))
