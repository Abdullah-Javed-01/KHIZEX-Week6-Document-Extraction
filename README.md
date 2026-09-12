# KHIZEX Week 6 — Automated Document Extraction (OCR + NLP)

An end-to-end legal-document extraction prototype for the **KHIZEX Data Science Internship — Week 6 Build Challenge**.

**Domain:** Legal service agreements  
**Data:** 8 fully synthetic contracts. No real client, patient, confidential, or personally sensitive data is included.

## What the project does

1. Generates reproducible synthetic scanned contract images with rotation, blur, illumination variation, and noise.
2. Pre-processes scans with deskewing, denoising, illumination normalization, and Otsu binarization.
3. Runs **Tesseract 5.5.0** OCR and records word-level confidence.
4. Segments OCR text into legal sections such as parties, effective date, payment, obligations, confidentiality, and governing law.
5. Extracts six critical fields using section context plus typed patterns.
6. Compares a generic extractive summarizer with a field-aware structured summarizer.
7. Evaluates OCR CER/WER, field precision/recall/F1, summary quality and faithfulness, throughput, bottlenecks, and failure modes.

## Quick start

### 1. Install Tesseract

Tesseract must be installed at the operating-system level and available on `PATH`.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the synthetic scans

```bash
python data/generate_scans.py
```

Generated PNG scans are intentionally ignored by Git because they are reproducible from `data/ground_truth/labels.csv`.

### 4. Run the walkthrough

Open:

```text
notebooks/01_document_extraction_walkthrough.ipynb
```

The reusable implementation is in `src/pipeline.py`, while the measured evaluation outputs from the completed run are included in `results/`.

## Key measured findings

| Metric | Result |
|---|---:|
| Raw OCR confidence | 93.49% |
| Processed OCR confidence | 96.00% |
| Raw WER | 0.138 |
| Processed WER | 0.088 |
| Macro extraction precision | 0.979 |
| Macro extraction recall | 0.979 |
| Macro extraction F1 | 0.979 |
| Extractive mean ROUGE-L | 0.260 |
| Field-aware mean ROUGE-L | 0.984 |
| Extractive field completeness | 0.354 |
| Field-aware field completeness | 0.938 |
| Extractive hallucination rate | 0.000 |
| Field-aware hallucination rate | 0.125 |
| Throughput | 17.24 documents/min |
| Main bottleneck | OCR |

The clearest preprocessing example is `contract_06`: raw WER was **0.508** and processed WER was **0.086**.

## OCR and layout

OCR recognizes visual character shapes and maps them to text; it does **not** understand the legal meaning of a document. Columns, tables, stamps, headers, footers, and unusual reading order can break naive top-to-bottom extraction. This prototype uses single-column synthetic agreements and then segments the recognized text into logical sections before critical-field extraction.

## Extraction vs. keyword search

A keyword search only finds literal strings such as `payment` or `date`. This extractor first identifies the section context and then applies typed patterns within the relevant section: a monetary expression inside `PAYMENT`, a date inside `EFFECTIVE DATE`, and party-role context inside `PARTIES`.

This is still a lightweight rule/NLP prototype rather than a trained legal NER or document-layout model. Unusual phrasing, OCR substitutions, unexpected layouts, and ambiguous references remain important failure modes.

## Responsible handling

Only synthetic data is used. A production legal-document deployment would require access control, encryption in transit and at rest, audit logging, retention/deletion policies, secure temporary files, confidentiality controls, jurisdiction-specific privacy review, visible provenance for extracted fields, and human review of low-confidence or high-impact outputs.

## Project structure

```text
.
├── data/
│   ├── generate_scans.py             # reproducible synthetic scan generator
│   └── ground_truth/
│       └── labels.csv                 # source text + six labeled fields
├── models/
│   └── README.md                      # model-artifact note
├── notebooks/
│   └── 01_document_extraction_walkthrough.ipynb
├── results/
│   ├── extracted_fields_vs_ground_truth.csv
│   ├── extraction_metrics.csv
│   ├── metrics_summary.json
│   ├── ocr_before_after_example.md
│   ├── ocr_confidence_plot.svg
│   ├── ocr_metrics.csv
│   ├── summarization_comparison.md
│   └── throughput.csv
├── src/
│   └── pipeline.py                    # OCR, parsing, extraction, summarization
├── ANALYSIS.md                        # written performance/failure analysis
├── SUBMISSION_CHECKLIST.md
├── requirements.txt
└── README.md
```

## Reproducibility note

The included `results/` directory is the completed evaluation run used for the internship submission. OCR confidence and timing can vary slightly by OS, CPU, font availability, and Tesseract build, so exact rerun values may differ while the evaluation procedure remains the same.
