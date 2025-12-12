import streamlit as st
import pandas as pd
import json
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------
# CONFIGURATION
# ------------------------------------------
st.set_page_config(
    page_title="Delta RAG System",
    page_icon="✈️",
    layout="wide"
)

company_name = "delta"

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
chunks = pd.read_csv(f"data/{company_name}/chunks.csv")

with open(f"data/{company_name}/kpi_summary.json") as f:
    verified_kpis = json.load(f)

# ------------------------------------------
# TF-IDF RETRIEVAL
# ------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
chunk_tfidf = vectorizer.fit_transform(chunks["chunk_text"])

def retrieve_chunks(query, top_k=5):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, chunk_tfidf).flatten()
    idx = sims.argsort()[::-1][:top_k]

    results = chunks.iloc[idx].copy()
    results["score"] = sims[idx]
    return results

# ------------------------------------------
# GROQ CLIENT
# ------------------------------------------

client = Groq(api_key="enter_key_api_here")


def build_prompt(question, retrieved):
    ctx = ""
    for _, row in retrieved.iterrows():
        ctx += (
            f"\n\n---\nChunk ID: {row['chunk_id']}\n"
            f"Source: {row['source_file']}\n"
            f"Text:\n{row['chunk_text']}\n"
        )

    return f"""
Use ONLY the information in the retrieved chunks. If the answer is not found, say: "The documents do not contain this information."
Cite your evidence using (chunk: chunk_id).

Question:
{question}
Context:
{ctx}
"""


def answer_question(question, model="llama-3.3-70b-versatile"):
    retrieved = retrieve_chunks(question)
    prompt = build_prompt(question, retrieved)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    answer = response.choices[0].message.content
    return answer, retrieved

def generate_company_overview():
    retrieved = retrieve_chunks("What is Delta Airlines?", top_k=5)
    prompt = build_prompt("Give a 3–4 sentence overview of Delta.\n", retrieved)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    desc = response.choices[0].message.content
    srcs = sorted(set(retrieved["source_file"].tolist()))
    return desc, srcs

# ------------------------------------------
# UI COMPONENTS
# ------------------------------------------
def kpi_card(title, value, unit=""):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #003A70, #68ACE5);
            padding:14px;
            border-radius:12px;
            margin-bottom:20px;
            box-shadow:0 2px 6px rgba(0,0,0,0.12);
        ">
            <h4 style="margin:0; margin-bottom:6px; text-align:center;">
                {title}
            </h4>
            <h2 style="margin:0; font-weight:700; text-align:center;">
                {value} {unit}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------
# HEADER
# ------------------------------------------

st.image("delta_logo.png", width=80)

st.markdown(
    "<h1 style='margin-bottom:0;'>Delta Intelligence RAG System</h1>",unsafe_allow_html=True)

# ------------------------------------------
# TABS
# ------------------------------------------
tab1, tab2 = st.tabs(["Company Snapshot", "Ask a Question"])

# ------------------------------------------
# TAB 1 — COMPANY SNAPSHOT
# ------------------------------------------
with tab1:
    st.markdown("### 📈 2024 KPIs")

    col1, col2, col3 = st.columns(3, gap="large")

    # COLUMN 1
    with col1:
        k = verified_kpis["total_operating_revenue"]
        kpi_card(k["label"], k["value"], k["unit"])

        k = verified_kpis["num_segments"]
        kpi_card(k["label"], k["value"])

    # COLUMN 2
    with col2:
        k = verified_kpis["revenue_growth_yoy"]
        kpi_card(k["label"], k["value"], k["unit"])

        k = verified_kpis["total_customer_served"]
        kpi_card(k["label"], k["value"])

    # COLUMN 3
    with col3:
        k = verified_kpis["total_cargo_revenue"]
        kpi_card(k["label"], k["value"], k["unit"])


        k = verified_kpis["total_fuel_saving"]
        kpi_card(k["label"], k["value"], k["unit"])

    # ESG TARGETS
    st.markdown("### 🌿 2024 Key ESG Targets")
    for t in verified_kpis["esg_targets"]["value"]:
        # esg_badge(t)
        st.write(f"- {t}")

    # COMPANY OVERVIEW
    st.markdown("### 🧠 Company Overview (LLM-generated)")
    if st.button("Generate Overview"):
        desc, srcs = generate_company_overview()

        st.markdown(
            f"""
            <div style="
                padding:20px;
                border-radius:12px;
                background:#151515;
                border:1px solid #333;
            ">
                <p style="font-size:16px; line-height:1.5;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption("Sources: " + ", ".join(srcs))

# ------------------------------------------
# TAB 2 — ASK A QUESTION
# ------------------------------------------
with tab2:

    st.markdown("### ❓ Use quick questions or type your own")

    colA, colB, colC, colD = st.columns(4)

    # ---- PREDEFINED BUTTON ACTIONS (direct answer) ----
    if colA.button("📌 Snapshot"):
        q = "Give a high-level business snapshot of Delta."
        answer, retrieved = answer_question(q)

        st.markdown("### 📘 Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Chunks"):
            st.dataframe(retrieved[["chunk_id","source_file","score"]])

    if colB.button("⚠️ Risks"):
        q = "What are the main risks?"
        answer, retrieved = answer_question(q)

        st.markdown("### 📘 Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Chunks"):
            st.dataframe(retrieved[["chunk_id","source_file","score"]])

    if colC.button("🌍 ESG"):
        q = "What ESG and environmental goals does Delta describe?"
        answer, retrieved = answer_question(q)

        st.markdown("### 📘 Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Chunks"):
            st.dataframe(retrieved[["chunk_id","source_file","score"]])

    if colD.button("🏛️ Stakeholders"):
        q = "Who are the major holders and stakeholders at Delta?"
        answer, retrieved = answer_question(q)

        st.markdown("### 📘 Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Chunks"):
            st.dataframe(retrieved[["chunk_id","source_file","score"]])

    # ---- CUSTOM QUESTION ----
    # st.markdown("### Or type your own question:")
    user_q = st.text_input("Custom question:", "")

    if st.button("Submit", type="primary"):
        if not user_q:
            st.warning("Please enter a question.")
        else:
            answer, retrieved = answer_question(user_q)

            st.markdown("### 📘 Answer")
            st.write(answer)

            with st.expander("🔍 Retrieved Chunks"):
                st.dataframe(retrieved[["chunk_id","source_file","score"]])

