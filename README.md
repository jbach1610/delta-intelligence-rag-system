# DSDA 310 – Case Study 2  
## Delta Intelligence RAG System

This repository contains the code, data pipeline, and deliverables for **Case Study 2** in DSDA 310.  
The project implements a **Company Intelligence Agent** for **Delta Air Lines** using a retrieval-augmented generation (RAG) framework. :contentReference[oaicite:1]{index=1}  

---

## 1. Case Study Overview

**Objectives**

- Build an end-to-end pipeline that:
  - Extracts text from corporate PDF documents (10-K, ESG report, shareholder info).
  - Cleans and chunks the text into a searchable document store.
  - Retrieves relevant chunks using TF-IDF and cosine similarity.
  - Uses an LLM to answer user questions based only on retrieved evidence.
- Expose the system through a **Streamlit dashboard** for non-technical users.

**Methodology**

1. **Document Collection**
   - Delta Air Lines 2024 Form 10-K (financials, operations, risk disclosures).
   - Delta Air Lines 2024 ESG Report (environmental metrics and sustainability goals).
   - Major shareholders summary from Yahoo Finance. :contentReference[oaicite:2]{index=2}  

2. **Data Processing Pipeline**
   - PDF text extraction with `pdfplumber`, plus **OCR-first** extraction for the ESG report due to scan-like formatting.
   - Text cleaning and standardization (remove headers/footers, normalize whitespace, fix PDF artifacts).
   - Sliding-window chunking (~600-word chunks with 100-word overlap) and storage as a unified document store (`chunks.csv`, `chunks.pkl`). :contentReference[oaicite:3]{index=3}  

3. **Retrieval System**
   - TF-IDF vectorization over all chunks.
   - Cosine similarity to rank and return the top-k most relevant chunks for each query. :contentReference[oaicite:4]{index=4}  

4. **LLM Integration**
   - Groq LLaMA-3.3-70B Versatile model.
   - Structured prompt that:
     - Restricts answers to retrieved chunks.
     - Requires chunk-ID citations.
     - Instructs the model to say “not in the documents” when evidence is missing. :contentReference[oaicite:5]{index=5}  

5. **KPI Extraction**
   - Semi-automated extraction of key KPIs from the 10-K and ESG report (e.g., total operating revenue, cargo revenue, YoY growth, number of segments, served customers, fuel savings, and main ESG targets), using regex search plus manual validation. :contentReference[oaicite:6]{index=6}  

6. **Streamlit App**
   - **Company Snapshot tab:** KPI cards, verified ESG targets, and an auto-generated 3–4 sentence overview.
   - **Ask-a-Question tab:** quick-select question groups (Snapshot, Risks, ESG, Stakeholders), user input box, LLM answer with chunk citations, and expandable retrieved-chunk table. :contentReference[oaicite:7]{index=7}  

**Results**

- Delivered a functional RAG system that:
  - Reliably surfaces relevant chunks for company, risk, ESG, and shareholder questions.
  - Provides transparent, cited LLM answers via a user-friendly dashboard.
- Identified key limitations around OCR quality, manual KPI extraction, chunking strategy, and API call limits, with recommendations for future improvements (better extraction methods, alternative chunking, and a more flexible LLM API). :contentReference[oaicite:8]{index=8}  
<img width="1512" height="826" alt="Screenshot 2025-12-13 at 3 16 59 AM" src="https://github.com/user-attachments/assets/49439a38-6b30-43ce-99ce-82c2916e0015" />
<img width="1512" height="826" alt="Screenshot 2025-12-13 at 3 17 30 AM" src="https://github.com/user-attachments/assets/fad6fb7a-e200-41fb-bf30-fae0abc75f9c" />



---

## 2. Repository Structure

```text
notebooks/               # Jupyter notebooks for extraction, cleaning/chunking, KPI extraction
app/                     # Streamlit app and config
data/                    # Raw PDFs (optional) and processed text/chunks
slide deck & report/     # Final report and slides
