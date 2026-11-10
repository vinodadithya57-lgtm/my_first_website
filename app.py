
import streamlit as st
import re
import io
from collections import Counter
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="AI DocuMind",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI DocuMind")
st.caption("Intelligent Document Analysis and Management System")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Dashboard", "Document Analyzer", "About"]
)

def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(io.BytesIO(file.read()))
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
    return file.getvalue().decode("utf-8", errors="ignore")

def classify_document(text):
    text_lower = text.lower()

    categories = {
        "Education": ["student", "college", "university", "course", "exam"],
        "Finance": ["invoice", "payment", "salary", "bank", "revenue"],
        "Technology": ["software", "python", "computer", "database", "system"],
        "Human Resources": ["employee", "recruitment", "resume", "interview", "attendance"],
        "Legal": ["agreement", "contract", "court", "legal", "clause"]
    }

    scores = {
        category: sum(text_lower.count(word) for word in words)
        for category, words in categories.items()
    }

    best = max(scores, key=scores.get)

    return best if scores[best] > 0 else "General"

def summarize_text(text, sentence_limit=5):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return "Not enough readable text to generate a summary."

    if len(sentences) <= sentence_limit:
        return " ".join(sentences)

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stop_words = {
        "the", "and", "for", "that", "this", "with",
        "from", "are", "was", "were", "have", "has",
        "you", "your", "not", "but", "can", "will"
    }

    frequencies = Counter(
        word for word in words if word not in stop_words
    )

    scored = []
    for index, sentence in enumerate(sentences):
        sentence_words = re.findall(
            r'\b[a-zA-Z]{3,}\b', sentence.lower()
        )
        score = sum(frequencies[w] for w in sentence_words)
        scored.append((score, index, sentence))

    selected = sorted(
        sorted(scored, reverse=True)[:sentence_limit],
        key=lambda item: item[1]
    )

    return " ".join(item[2] for item in selected)

def extract_keywords(text):
    if len(text.split()) < 3:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform([text])
    scores = matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(terms, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return [term for term, score in ranked if score > 0]

if page == "Dashboard":
    st.subheader("Project Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Platform", "AI DocuMind")
    col2.metric("Supported formats", "PDF / TXT")
    col3.metric("Analysis tools", "3")

    st.info(
        "Upload a document in Document Analyzer to extract "
        "text, identify its category, generate a summary "
        "and discover important keywords."
    )

    st.markdown("### Available Features")
    st.write("- Document text extraction")
    st.write("- Automatic category identification")
    st.write("- Extractive text summarization")
    st.write("- TF-IDF keyword extraction")

elif page == "Document Analyzer":
    st.subheader("Upload and Analyze Documents")

    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        try:
            text = extract_text(uploaded_file)

            if not text.strip():
                st.warning(
                    "No readable text found. "
                    "Scanned PDFs may require OCR."
                )
            else:
                st.success("Document successfully processed!")

                st.metric("Extracted words", len(text.split()))

                st.markdown("### Document Category")
                st.info(classify_document(text))

                st.markdown("### Document Summary")
                st.write(summarize_text(text))

                st.markdown("### Important Keywords")
                keywords = extract_keywords(text)

                if keywords:
                    st.write(", ".join(keywords))
                else:
                    st.write("Not enough text for keyword extraction.")

                with st.expander("View Extracted Text"):
                    st.text_area(
                        "Document content",
                        text,
                        height=300
                    )

                st.download_button(
                    "Download Extracted Text",
                    data=text,
                    file_name="extracted_document.txt",
                    mime="text/plain"
                )

        except Exception as error:
            st.error(f"Unable to process document: {error}")

elif page == "About":
    st.subheader("About AI DocuMind")

    st.write("""
    AI DocuMind is a document intelligence application
    developed using Python and Natural Language Processing
    techniques.

    It demonstrates document extraction, keyword analysis,
    extractive summarization and rule-based classification.

    Developer: Adithya Vinod
    """)

    st.markdown("### Technologies")
    st.write("Python | Streamlit | PyPDF | Scikit-learn | NLP")
