# ==============================
# AI CONTENT VALIDATION SYSTEM
# FINAL VERSION (BEST UI + ALL OUTPUTS)
# ==============================

import pandas as pd
import re
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Content Validation", layout="wide")

# -----------------------------
# 🎨 LIGHT PROFESSIONAL UI
# -----------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8fafc, #e0f2fe);
}
html, body {
    color: #0f172a;
    font-size: 18px;
}
h1 {
    color: #0c4a6e;
    font-size: 36px;
}
h2, h3 {
    color: #0369a1;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Content Validation System")

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------
# MEDICAL TERMS
# -----------------------------
medical_terms = [
    "fever","diabetes","bp","blood pressure","heart",
    "infection","pain","cancer","tumor","covid",
    "symptoms","treatment","diagnosis","patient","dose"
]

# -----------------------------
# CLASSIFICATION
# -----------------------------
def classify(score):
    if score >= 0.6:
        return "Reliable"
    elif score >= 0.35:
        return "Moderate"
    else:
        return "Unreliable"

def severity(score):
    if score < 0.2:
        return "🔴 Critical"
    elif score < 0.35:
        return "🟠 High"
    elif score < 0.6:
        return "🟡 Medium"
    else:
        return "🟢 Low"

# -----------------------------
# 🔍 SINGLE TEXT VALIDATION
# -----------------------------
st.header("🔍 Single Text Validation")

col1, col2 = st.columns(2)
with col1:
    ai_text = st.text_area("AI Content")
with col2:
    final_text = st.text_area("Final Content")

if st.button("Check Similarity"):

    ai = clean_text(ai_text)
    final = clean_text(final_text)

    vectorizer = TfidfVectorizer(vocabulary=medical_terms)
    tfidf = vectorizer.fit_transform([ai, final])
    tfidf_score = cosine_similarity(tfidf[0], tfidf[1])[0][0]

    emb = model.encode([ai, final])
    semantic_score = cosine_similarity([emb[0]], [emb[1]])[0][0]

    final_score = (0.15 * tfidf_score) + (0.85 * semantic_score)

    m1, m2, m3 = st.columns(3)
    m1.metric("Final Score", round(final_score,3))
    m2.metric("TF-IDF (Medical)", round(tfidf_score,3))
    m3.metric("Semantic", round(semantic_score,3))

    st.success(f"{classify(final_score)} | {severity(final_score)}")

# -----------------------------
# 📂 DATASET
# -----------------------------
st.header("📂 Upload Dataset")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("Run Full Analysis"):

        ai_clean = df['AI_CONTENT'].astype(str).apply(clean_text).tolist()
        final_clean = df['FINAL_CONTENT'].astype(str).apply(clean_text).tolist()

        # TF-IDF (medical)
        vectorizer = TfidfVectorizer(vocabulary=medical_terms)
        tfidf_matrix = vectorizer.fit_transform(ai_clean + final_clean)

        tfidf_scores = [
            cosine_similarity(tfidf_matrix[i], tfidf_matrix[i + len(ai_clean)])[0][0]
            for i in range(len(ai_clean))
        ]

        # Semantic
        ai_emb = model.encode(ai_clean, batch_size=64)
        final_emb = model.encode(final_clean, batch_size=64)

        semantic_scores = [
            cosine_similarity([a],[b])[0][0]
            for a,b in zip(ai_emb, final_emb)
        ]

        df['SIMILARITY_SCORE'] = [(0.15*t + 0.85*s) for t,s in zip(tfidf_scores, semantic_scores)]
        df['RESULT'] = df['SIMILARITY_SCORE'].apply(classify)
        df['SEVERITY'] = df['SIMILARITY_SCORE'].apply(severity)

        st.success("✅ Analysis Complete")

        # -----------------------------
        # 📊 DASHBOARD
        # -----------------------------
        st.subheader("📊 Dashboard")
        c1,c2,c3 = st.columns(3)
        c1.metric("Avg Score", round(df['SIMILARITY_SCORE'].mean(),3))
        c2.metric("Reliable", len(df[df['RESULT']=="Reliable"]))
        c3.metric("Unreliable", len(df[df['RESULT']=="Unreliable"]))

        # -----------------------------
        # 🎯 ACCURACY
        # -----------------------------
        if 'EXPECTED_RESULT' in df.columns:
            st.subheader("🎯 Model Accuracy")
            acc = accuracy_score(df['EXPECTED_RESULT'], df['RESULT'])
            st.write("Accuracy:", round(acc,3))
            st.text(classification_report(df['EXPECTED_RESULT'], df['RESULT']))
            st.write(confusion_matrix(df['EXPECTED_RESULT'], df['RESULT']))

        # -----------------------------
        # 📊 SIMILARITY RANGE
        # -----------------------------
        st.subheader("📊 Similarity Analysis")

        df['SIMILARITY_RANGE'] = pd.cut(
            df['SIMILARITY_SCORE'],
            bins=[0,0.4,0.7,1],
            labels=['Low','Medium','High'],
            include_lowest=True
        )

        st.write(df['SIMILARITY_RANGE'].value_counts())

        # -----------------------------
        # 📈 GRAPHS
        # -----------------------------
        plt.style.use('default')

        st.subheader("📊 Result Distribution")
        fig, ax = plt.subplots()
        df['RESULT'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

        st.subheader("📈 Score Distribution")
        fig2, ax2 = plt.subplots()
        df['SIMILARITY_SCORE'].hist(bins=20, ax=ax2)
        st.pyplot(fig2)

        # -----------------------------
        # 📉 TEMPORAL
        # -----------------------------
        if 'DATE' in df.columns:
            st.subheader("📉 Temporal Drift")
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            drift = df.groupby(df['DATE'].dt.date)['SIMILARITY_SCORE'].mean()
            st.line_chart(drift)

        # -----------------------------
        # 🏥 DEPARTMENT
        # -----------------------------
        if 'DEPARTMENT' in df.columns:
            st.subheader("🏥 Department Analysis")
            dept = df.groupby('DEPARTMENT')['SIMILARITY_SCORE'].mean()
            st.bar_chart(dept)

        # -----------------------------
        # 🚨 HALLUCINATION
        # -----------------------------
        st.subheader("🚨 Hallucination Detection")
        hallucinated = df[df['SIMILARITY_SCORE'] < 0.4]
        st.write("Total:", len(hallucinated))

        # -----------------------------
        # 🧠 CONFIDENCE
        # -----------------------------
        st.subheader("🧠 Confidence Analysis")
        overconf = df[(df['SIMILARITY_SCORE'] > 0.7) & (df['SIMILARITY_SCORE'] < 0.4)]
        st.write("Overconfident:", len(overconf))

        # -----------------------------
        # 🔝 CASES
        # -----------------------------
        st.subheader("🚨 Worst Cases")
        st.dataframe(df.sort_values(by='SIMILARITY_SCORE').head(10))

        st.subheader("🌟 Best Cases")
        st.dataframe(df.sort_values(by='SIMILARITY_SCORE', ascending=False).head(5))

        st.subheader("⚖️ Medium Cases")
        st.dataframe(df[df['SEVERITY']=="🟡 Medium"].head(5))

        # -----------------------------
        # DOWNLOAD
        # -----------------------------
        st.download_button("⬇ Download Results", df.to_csv(index=False), "results.csv")
