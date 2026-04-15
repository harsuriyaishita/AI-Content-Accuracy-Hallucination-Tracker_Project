# redeploy trigger 

import pandas as pd
import re
import streamlit as st
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore

st.set_page_config(page_title="AI Content Validation", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    color: #1a1a2e !important;
}

html, body, p, span, label {
    color: #1a1a2e !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    line-height: 1.6;
}

h1 {
    font-size: 60px !important;
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #f9ca24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px !important;
}

h2 {
    font-size: 42px !important;
    color: #e74c3c !important;
    font-weight: 900 !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

h3 {
    font-size: 32px !important;
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}

textarea {
    background: linear-gradient(145deg, #ffffff, #f0f2ff) !important;
    color: #1a1a2e !important;
    font-size: 20px !important;
    border: 3px solid transparent;
    border-radius: 20px !important;
    padding: 20px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    font-weight: 500;
    transition: all 0.3s ease;
}

textarea:focus {
    border-color: #4ecdc4 !important;
    box-shadow: 0 0 0 3px rgba(78, 205, 196, 0.2);
}

.stTextArea label {
    color: #2c3e50 !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    margin-bottom: 10px;
}

div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #ffffff, #e6ecf0);
    border-radius: 25px;
    padding: 25px !important;
    border: 3px solid transparent;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

[data-testid="stMetricValue"] {
    font-size: 48px !important;
    font-weight: 900 !important;
    line-height: 1.2;
}

[data-testid="stMetricLabel"] {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #5a6c7d !important;
}

.stButton > button {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4) !important;
    color: white !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    padding: 18px 40px !important;
    border-radius: 25px !important;
    border: none !important;
    box-shadow: 0 15px 35px rgba(255,107,107,0.4);
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 25px 50px rgba(255,107,107,0.6) !important;
}

.stSuccess {
    background: linear-gradient(90deg, #56ab2f, #a8e6cf) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}

.stError {
    background: linear-gradient(90deg, #ff4757, #ff6b7a) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Content Validation System")

medical_terms = [
    "fever","diabetes","bp","blood","pressure","heart",
    "infection","pain","cancer","tumor","covid",
    "symptoms","treatment","diagnosis","patient","dose"
]

@st.cache_resource
def load_models():
    models = {}
    models["medcpt"] = SentenceTransformer("ncbi/MedCPT-Query-Encoder")
    try:
        models["pubmedbert"] = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    except:
        models["pubmedbert"] = None
    try:
        models["clinicalbert"] = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")
    except:
        models["clinicalbert"] = None
    return models

models = load_models()

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 ]', '', str(text).lower())

def compute_bertscore(a, b):
    try:
        with torch.no_grad():
            _, _, F1 = bertscore(
                a, b,
                model_type="distilbert-base-uncased",
                lang="en",
                rescale_with_baseline=True,
                device="cpu"
            )
        return float(F1[0])
    except:
        return 0.0

def medical_match(ai, final):
    ai_words = set(ai.split())
    final_words = set(final.split())
    match = [w for w in medical_terms if w in ai_words and w in final_words]
    return len(match) / (len(medical_terms) + 1)

def detect_contradiction(ai, final):
    negations = ["no","not","never","none","without","negative","dont","doesnt"]
    
    ai_words = set(ai.split())
    final_words = set(final.split())
    
    common_med = [w for w in medical_terms if w in ai_words and w in final_words]
    
    if not common_med:
        return 0
    
    ai_neg = any(n in ai_words for n in negations)
    final_neg = any(n in final_words for n in negations)
    
    return 1 if ai_neg != final_neg else 0

def compute_similarity(model_name, ai, final):
    if model_name not in models or models[model_name] is None:
        return 0.0
    try:
        model = models[model_name]
        emb = model.encode([ai, final])
        return cosine_similarity([emb[0]], [emb[1]])[0][0]
    except:
        return 0.0

def final_score_calc(ai, final):
    medcpt = compute_similarity("medcpt", ai, final)
    bert = compute_bertscore([ai], [final])
    med_match = medical_match(ai, final)
    
    base_score = (0.6 * medcpt) + (0.25 * bert) + (0.15 * med_match)
    
    contradiction = detect_contradiction(ai, final)
    
    if contradiction:
        final_score = base_score * 0.2
    else:
        final_score = base_score
    
    pubmedbert = compute_similarity("pubmedbert", ai, final)
    clinicalbert = compute_similarity("clinicalbert", ai, final)
    
    return final_score, medcpt, bert, med_match, pubmedbert, clinicalbert, contradiction

def smart_result(final_score, contradiction):
    if contradiction:
        return "❌ CONTRADICTION"
    elif final_score > 0.75:
        return "✅ Reliable"
    elif final_score > 0.5:
        return "⚠️ Moderate"
    else:
        return "❌ Unreliable"

st.header("🔍 Single Text Validation")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("🤖 AI Content")
    ai_text = st.text_area("", height=350, placeholder="Paste AI generated content here...")

with col2:
    st.subheader("✅ Final Content")
    final_text = st.text_area("", height=350, placeholder="Paste verified final content here...")

if st.button("🚀 VALIDATE SIMILARITY", use_container_width=True):
    ai = clean_text(ai_text)
    final = clean_text(final_text)

    if len(ai.strip()) < 2 or len(final.strip()) < 2:
        st.error("Please enter both AI and Final content!")
    else:
        with st.spinner("Computing with contradiction detection..."):
            final_score, medcpt, bert, med_match, pubmedbert, clinicalbert, contradiction = final_score_calc(ai, final)

        st.subheader("📊 UPDATED FORMULA RESULTS")
        st.latex(r'''$$BASE\_SCORE = (0.6 \times MedCPT) + (0.25 \times BERTScore) + (0.15 \times MedicalMatch)$$''')
        st.latex(r'''$$FINAL\_SCORE = BASE\_SCORE \times (0.2 \text{ if CONTRADICTION else } 1)$$''')

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🔬 MedCPT", round(medcpt,3))
        c2.metric("🧠 BERTScore", round(bert,3))
        c3.metric("💊 Medical Match", round(med_match,3))
        c4.metric("🚨 Contradiction", contradiction)
        c5.metric("🎯 FINAL_SCORE", round(final_score,3))

        result = smart_result(final_score, contradiction)
        if contradiction:
            st.error(f"🚨 {result}")
        else:
            st.success(f"🎉 {result}")

st.header("📂 Dataset Analysis")

file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    if st.button("🚀 FULL ANALYSIS", use_container_width=True):
        progress = st.progress(0)
        
        final_scores = []
        medcpts = []
        berts = []
        med_matches = []
        pubmedberts = []
        clinicalberts = []
        contradictions = []

        for idx in range(len(df)):
            a = df.iloc[idx]['AI_CONTENT']
            b = df.iloc[idx]['FINAL_CONTENT']
            
            ai = clean_text(a)
            final = clean_text(b)

            final_score, medcpt, bert, med_match, pubmedbert, clinicalbert, contradiction = final_score_calc(ai, final)
            
            final_scores.append(final_score)
            medcpts.append(medcpt)
            berts.append(bert)
            med_matches.append(med_match)
            pubmedberts.append(pubmedbert)
            clinicalberts.append(clinicalbert)
            contradictions.append(contradiction)
            
            progress.progress((idx + 1) / len(df))

        df['FINAL_SCORE'] = final_scores
        df['MEDCPT'] = medcpts
        df['BERTSCORE'] = berts
        df['MEDICAL_MATCH'] = med_matches
        df['PUBMEDBERT'] = pubmedberts
        df['CLINICALBERT'] = clinicalberts
        df['CONTRADICTION'] = contradictions
        df['RESULT'] = df.apply(lambda row: smart_result(row['FINAL_SCORE'], row['CONTRADICTION']), axis=1)

        st.success(f"✅ Analysis Complete! Processed {len(df)} records")
        st.latex(r'''$$BASE\_SCORE = (0.6 \times MedCPT) + (0.25 \times BERTScore) + (0.15 \times MedicalMatch)$$''')
        st.latex(r'''$$FINAL\_SCORE = BASE\_SCORE \times (0.2 \text{ if CONTRADICTION else } 1)$$''')

        # ALL MODEL PERFORMANCE CARDS
        st.subheader("📈 All Model Performance")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric("🔬 MedCPT (60%)", round(df['MEDCPT'].mean(),3))
        col2.metric("🧠 BERTScore (25%)", round(df['BERTSCORE'].mean(),3))
        col3.metric("💊 Medical Match (15%)", round(df['MEDICAL_MATCH'].mean(),3))
        col4.metric("🚨 Contradictions", df['CONTRADICTION'].sum())
        col5.metric("📚 PubMedBERT", round(df['PUBMEDBERT'].mean(),3))
        col6.metric("🏥 ClinicalBERT", round(df['CLINICALBERT'].mean(),3))
        col7.metric("🎯 FINAL_SCORE", round(df['FINAL_SCORE'].mean(),3))

        # DEPARTMENT LOGIC - FIXED SYNTAX ERROR
        if 'DEPARTMENT' in df.columns:
            st.subheader("🏥 Department Analysis")
            dept = df.groupby('DEPARTMENT')

            dept_df = pd.DataFrame({
                "Total Cases": dept.size(),
                "Avg Score": dept['FINAL_SCORE'].mean(),
                "Reliable %": dept.apply(lambda x: (x['RESULT']=="✅ Reliable").mean()*100),
                "Contradiction %": dept['CONTRADICTION'].mean()*100,
                "Hallucination %": dept.apply(lambda x: (x['FINAL_SCORE']<0.4).mean()*100)
            }).round(2)

            st.dataframe(dept_df)
            st.bar_chart(dept_df['Avg Score'])
            st.markdown("""
            **X-axis:** Department  
            **Y-axis:** Average FINAL_SCORE  
            ➡ Compares AI performance across departments.
            """)

        # CONTRADICTION + HALLUCINATION LOGIC
        st.subheader("🚨 Critical Issues Detection")
        contradictions_df = df[df['CONTRADICTION'] == 1]
        hallucinated = df[df['FINAL_SCORE'] < 0.4]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Contradictions", len(contradictions_df))
        col2.metric("💥 Hallucinations", len(hallucinated))
        col3.metric("⚠️ Total Critical", len(contradictions_df) + len(hallucinated))
        
        if len(contradictions_df) > 0:
            st.subheader("🔴 Contradiction Examples")
            st.dataframe(contradictions_df[['AI_CONTENT','FINAL_CONTENT','FINAL_SCORE','CONTRADICTION']].head(5))

        # DASHBOARD
        st.subheader("📊 Dashboard")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Global Avg FINAL_SCORE", round(df['FINAL_SCORE'].mean(),3))
        col2.metric("✅ Reliable", len(df[df['RESULT']=='✅ Reliable']))
        col3.metric("🚨 Contradictions", len(contradictions_df))
        col4.metric("💥 Hallucinations", len(hallucinated))
        col5.metric("📊 Total Cases", len(df))

        st.bar_chart(df['RESULT'].value_counts())

        # WORST CASES & BEST CASES
        st.subheader("🚨 Worst Cases (Prioritized: Contradictions > Lowest Scores)")
        # Prioritize contradictions first, then lowest scores
        worst_priority = df.copy()
        worst_priority['priority_score'] = worst_priority['CONTRADICTION'] * 2 + (1 - worst_priority['FINAL_SCORE'])
        worst_cases = worst_priority.sort_values('priority_score', ascending=False).head(10)[['AI_CONTENT','FINAL_CONTENT','FINAL_SCORE','CONTRADICTION','RESULT']]
        st.dataframe(worst_cases.round(3), use_container_width=True, height=300)

        st.subheader("🌟 Best Cases (Highest Scores)")
        best_cases = df.sort_values('FINAL_SCORE', ascending=False).head(10)[['AI_CONTENT','FINAL_CONTENT','FINAL_SCORE','CONTRADICTION','RESULT']]
        st.dataframe(best_cases.round(3), use_container_width=True, height=300)

        # DOWNLOAD BUTTON
        st.download_button("⬇️ Download Results", df.to_csv(index=False), "results.csv")
