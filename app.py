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
manual_text = st.text_area("Or paste text manually", height=150)
if st.button("Verify Uploaded Document"):
    with st.spinner("Analyzing uploaded document..."):
        result = process_input(uploaded_file, "")
    st.text_area("Evidence Report", value=result, height=400, key="report1")
if st.button("Verify Manual Text"):
    with st.spinner("Analyzing manual text..."):
        result = process_input(None, manual_text)
    st.text_area("Evidence Report", value=result, height=400, key="report2")
