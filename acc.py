# ai_content_validation_app.py

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
st.title("🧠 AI Content Validation System")

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

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
# SINGLE TEXT VALIDATION
# -----------------------------
st.header("🔍 Single Text Validation")

ai_text = st.text_area("Enter AI Content")
final_text = st.text_area("Enter Final Content")

if st.button("Check Similarity"):

    ai = clean_text(ai_text)
    final = clean_text(final_text)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([ai, final])
    tfidf_score = cosine_similarity(tfidf[0], tfidf[1])[0][0]

    # Semantic similarity
    emb = model.encode([ai, final])
    semantic_score = cosine_similarity([emb[0]], [emb[1]])[0][0]

    # Final weighted score
    final_score = (0.4 * tfidf_score) + (0.6 * semantic_score)

    st.success(f"Final Score: {round(final_score,3)}")
    st.write("TF-IDF:", round(tfidf_score,3))
    st.write("Semantic:", round(semantic_score,3))
    st.write("Result:", classify(final_score))
    st.write("Severity:", severity(final_score))

# -----------------------------
# CSV UPLOAD
# -----------------------------
st.header("📂 Upload Dataset")

file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Preview", df.head())

    if st.button("Run Full Analysis"):

        # Clean text
        ai_clean = df['AI_CONTENT'].astype(str).apply(clean_text).tolist()
        final_clean = df['FINAL_CONTENT'].astype(str).apply(clean_text).tolist()

        # TF-IDF scores
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(ai_clean + final_clean)
        tfidf_scores = [
            cosine_similarity(tfidf_matrix[i], tfidf_matrix[i + len(ai_clean)])[0][0]
            for i in range(len(ai_clean))
        ]

        # Semantic similarity
        ai_emb = model.encode(ai_clean, batch_size=64, show_progress_bar=True)
        final_emb = model.encode(final_clean, batch_size=64, show_progress_bar=True)
        semantic_scores = [
            cosine_similarity([a], [b])[0][0]
            for a, b in zip(ai_emb, final_emb)
        ]

        # Final weighted score
        df['SIMILARITY_SCORE'] = [(0.4*t + 0.6*s) for t,s in zip(tfidf_scores, semantic_scores)]
        df['RESULT'] = df['SIMILARITY_SCORE'].apply(classify)
        df['SEVERITY'] = df['SIMILARITY_SCORE'].apply(severity)

        # -----------------------------
        # MODEL ACCURACY
        # -----------------------------
        if 'EXPECTED_RESULT' in df.columns:
            st.subheader("🎯 Model Accuracy")
            accuracy = accuracy_score(df['EXPECTED_RESULT'], df['RESULT'])
            st.write(f"Accuracy: {round(accuracy,3)}")

            st.subheader("📊 Classification Report")
            report = classification_report(df['EXPECTED_RESULT'], df['RESULT'])
            st.text(report)

            st.subheader("📉 Confusion Matrix")
            cm = confusion_matrix(df['EXPECTED_RESULT'], df['RESULT'])
            st.write(cm)

        st.success("✅ Analysis Complete")

        # -----------------------------
        # INSIGHTS
        # -----------------------------
        st.subheader("📊 Insights")
        st.write("Average Similarity:", round(df['SIMILARITY_SCORE'].mean(),3))
        st.write(df['RESULT'].value_counts())

        # -----------------------------
        # DISTRIBUTION GRAPH
        # -----------------------------
        st.subheader("📊 Result Distribution")
        plt.figure()
        df['RESULT'].value_counts().plot(kind='bar')
        st.pyplot(plt)

        st.subheader("📈 Score Distribution")
        plt.figure()
        df['SIMILARITY_SCORE'].hist(bins=20)
        st.pyplot(plt)

        # -----------------------------
        # TEMPORAL DRIFT
        # -----------------------------
        if 'DATE' in df.columns:
            st.subheader("📉 Temporal Drift")
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            drift = df.groupby(df['DATE'].dt.date)['SIMILARITY_SCORE'].mean()
            plt.figure()
            drift.plot()
            plt.xlabel("Date")
            plt.ylabel("Avg Similarity")
            st.pyplot(plt)

        # -----------------------------
        # DEPARTMENT ANALYSIS
        # -----------------------------
        if 'DEPARTMENT' in df.columns:
            st.subheader("🏥 Department Analysis")
            dept_avg = df.groupby('DEPARTMENT')['SIMILARITY_SCORE'].mean()
            plt.figure()
            dept_avg.plot(kind='bar')
            st.pyplot(plt)

            st.write("Hallucination Count by Department:")
            st.write(df[df['RESULT']=="Unreliable"].groupby('DEPARTMENT').size())

        # -----------------------------
        # TOP CASES
        # -----------------------------
        st.subheader("🚨 Top 10 Worst Cases")
        worst = df.sort_values(by='SIMILARITY_SCORE').head(10)
        st.dataframe(worst[['AI_CONTENT','FINAL_CONTENT','SIMILARITY_SCORE','SEVERITY']])

        st.subheader("🌟 Top 5 Best Cases")
        best = df.sort_values(by='SIMILARITY_SCORE', ascending=False).head(5)
        st.dataframe(best[['AI_CONTENT','FINAL_CONTENT','SIMILARITY_SCORE','SEVERITY']])

        st.subheader("⚖️ Top 5 Medium Cases")
        medium = df[df['SEVERITY'] == "🟡 Medium"].sort_values(by='SIMILARITY_SCORE', ascending=False).head(5)
        st.dataframe(medium[['AI_CONTENT','FINAL_CONTENT','SIMILARITY_SCORE','SEVERITY']])

        # -----------------------------
        # DOWNLOAD RESULTS
        # -----------------------------
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Results", csv, "results.csv", "text/csv")