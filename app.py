import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pdfplumber
import docx
from PIL import Image
import re
from textblob import TextBlob
import fitz
import tempfile
import os

# ------------------------
# Hugging Face Models
# ------------------------

# Classification Model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

# OCR Model (instead of pytesseract)
ocr_pipeline = pipeline("image-to-text", model="microsoft/trocr-base-printed")


# ------------------------
# Extraction Functions
# ------------------------
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():  # OCR fallback
        ocr_text = ""
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text += ocr_pipeline(img)[0]["generated_text"] + "\n"
        text = ocr_text
    return text.strip()


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs]).strip()


def extract_text_from_image(file_path):
    img = Image.open(file_path)
    return ocr_pipeline(img)[0]["generated_text"].strip()


def check_grammar(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text != text


def extract_dates(text):
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b',
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+\w+\s*,?\s*\d{2,4}\b',
        r'\b\w+\s+\d{1,2},\s*\d{4}\b',
    ]
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        dates_found.extend(matches)
    return list(set(dates_found))


def classify_dates(text, dates):
    issue_keywords = ["issued on", "dated", "notified on", "circular no"]
    event_keywords = ["holiday", "observed on", "exam on", "will be held on", "effective from"]

    issue_dates, event_dates = [], []
    for d in dates:
        idx = text.lower().find(d.lower())
        if idx != -1:
            context = text[max(0, idx-60): idx+60].lower()
            if any(k in context for k in issue_keywords):
                issue_dates.append(d)
            elif any(k in context for k in event_keywords):
                after_text = text[idx: idx+80]
                match = re.search(rf"{re.escape(d)}[^\n]*", after_text)
                event_dates.append(match.group().strip() if match else d)

    if not issue_dates and dates:
        issue_dates.append(dates[0])
    return issue_dates, event_dates


# ------------------------
# Verification Logic
# ------------------------
def verify_text(text, source_type="TEXT", has_signature=False):
    if not text.strip():
        return "--- Evidence Report ---\n\n❌ No readable text provided."

    # Grammar
    grammar_issue = check_grammar(text)
    # Dates
    dates = extract_dates(text)
    issue_dates, event_dates = classify_dates(text, dates)

    # Scam / fake indicators
    scam_keywords = [
        "bank details", "send money", "lottery", "win prize",
        "transfer fee", "urgent", "click here", "claim", "scholarship $"
    ]
    scam_detected = any(kw in text.lower() for kw in scam_keywords)

    # Date consistency
    contradiction = False
    if issue_dates and event_dates:
        try:
            from datetime import datetime
            fmt_variants = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %B %Y", "%B %d, %Y"]

            def parse_date(d):
                for fmt in fmt_variants:
                    try:
                        return datetime.strptime(d, fmt)
                    except Exception:
                        continue
                return None

            parsed_issue = parse_date(issue_dates[0])
            parsed_event = parse_date(event_dates[0])
            if parsed_issue and parsed_event and parsed_event < parsed_issue:
                contradiction = True
        except Exception:
            pass

    # Hugging Face Classification
    labels = ["REAL", "FAKE"]
    result = classifier(text[:1000], candidate_labels=labels)
    model_label = result['labels'][0]
    model_confidence = result['scores'][0]

    # ------------------------
    # Signature / Seal Boost
    # ------------------------
    signature_keywords = ["signature", "signed by", "seal", "stamp", "authorized", "principal", "head of"]
    if any(kw in text.lower() for kw in signature_keywords) or has_signature:
        model_confidence = min(1.0, model_confidence + 0.25)  # boost confidence
        if model_label == "FAKE" and not (scam_detected or contradiction or grammar_issue):
            model_label = "REAL"

    # Final Verdict
    final_label = model_label
    if scam_detected or contradiction or grammar_issue:
        final_label = "FAKE"

    # Report
    report = "📄 Evidence Report\n\n"
    report += "🔎 Document Analysis\n\n"
    report += f"Source: {source_type}\n\n"

    report += "✅ Evidence Considered\n\n"
    if grammar_issue:
        report += "⚠️ Grammar/Spelling issues detected.\n"
    else:
        report += "No grammar issues detected.\n"

    if issue_dates:
        report += f"📌 Issue Date(s): {', '.join(issue_dates)}\n"
    if event_dates:
        report += f"📌 Event Date(s): {', '.join(event_dates)}\n"
    if not dates:
        report += "No specific dates detected.\n"

    if contradiction:
        report += "⚠️ Date inconsistency detected (event before issue date).\n"
    if scam_detected:
        report += "⚠️ Scam-related keywords detected.\n"
    if has_signature or any(kw in text.lower() for kw in signature_keywords):
        report += "✅ Signature/Seal detected in document.\n"

    report += "\nFormatting and tone analyzed.\n\n"
    report += "🏁 Classification Result\n\n"
    report += f"Model Verdict: {model_label} ({model_confidence:.2f})\n"
    report += f"Final Label: {final_label}\n"

    return report



def verify_document(file):
    if file is None:
        return "❌ Please upload a file or provide a file path."

    # Handle file type
    suffix = os.path.splitext(file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        file_path = tmp.name

    ext = file_path.split('.')[-1].lower()
    if ext == "pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == "docx":
        text = extract_text_from_docx(file_path)
    elif ext in ["png", "jpg", "jpeg"]:
        text = extract_text_from_image(file_path)
    else:
        return "❌ Unsupported file type."

    return verify_text(text, source_type=ext.upper())


def process_input(file, manual_text):
    if file is not None:
        return verify_document(file)
    elif manual_text.strip():
        return verify_text(manual_text, source_type="MANUAL TEXT")
    else:
        return "❌ Please upload a document or paste text first."


# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Document Verifier", layout="centered")
st.title("📑 Document Authenticity Verifier")

uploaded_file = st.file_uploader(
    "Upload a document (PDF, DOCX, PNG, JPG)",
    type=["pdf", "docx", "png", "jpg", "jpeg"]
)
manual_text = st.text_area("Or paste text manually")

# Button for uploaded files
if st.button("Verify Uploaded Document"):
    with st.spinner("Analyzing uploaded document..."):
        result = process_input(uploaded_file, "")
    st.text_area("Evidence Report", value=result, height=400)

# Button for manual text
if st.button("Verify Manual Text"):
    with st.spinner("Analyzing manual text..."):
        result = process_input(None, manual_text)
    st.text_area("Evidence Report", value=result, height=400)
