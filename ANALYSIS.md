# Week 6 Analysis — Automated Legal Document Extraction

## Scope and data

This project implements an end-to-end OCR + NLP pipeline for **legal service agreements**. The test set contains eight synthetic agreements created only for this assignment and rendered as scanned PNG pages with small rotations, blur, fading, and noise. This satisfies the sensitive-data constraint without using real client material. Each document has ground-truth transcription and six labeled fields: two parties, effective date, monetary value, governing law, and the provider obligation.

## OCR and pre-processing performance

Tesseract OCR was run on both the raw scan and a pre-processed image. Pre-processing applied deskewing, non-local-means denoising, and Otsu binarization. Across the eight documents, mean OCR confidence changed from **93.49%** to **96.00%**. Mean character error rate (CER) changed from **0.092** to **0.077**, while word error rate (WER) changed from **0.138** to **0.088**. The purpose of pre-processing is not to “understand” the contract; it improves the visual signal presented to OCR. The before/after evidence for `contract_06` is saved in `results/ocr_before_after_example.md`.

OCR remains layout-sensitive. Columns, stamps, signatures, tables, faint photocopies, or page borders can disturb reading order or character recognition. This test set intentionally stays mostly single-column so that the impact of scan quality can be measured independently. The worst processed WER was **0.102** on **contract_07**, one of the degraded scans. In production, pages below a confidence threshold should be routed to human review instead of silently accepted.

## Structure parsing and critical-field extraction

The OCR text is divided into legal sections before extraction. Headings are matched with a small amount of fuzzy tolerance so minor OCR punctuation does not collapse the whole document into one text blob. The extractor then uses contextual patterns inside the relevant section. This differs from simple keyword search because the same token can mean different things depending on its role and section; for example, a dollar pattern is interpreted as the service fee only inside PAYMENT.

Field-level evaluation uses exact normalized matches against held-out ground truth. Macro precision, recall, and F1 were **0.979**, **0.979**, and **0.979**. Per-field results are in `results/extraction_metrics.csv`. The main failure modes are OCR substitutions inside names, heading recognition errors that send text to the wrong section, and legal phrasing that differs from the expected templates. A production system should add a trained legal NER/document-layout model, confidence calibration, field provenance, and human correction workflows rather than relying only on deterministic rules.

## Summarization comparison and faithfulness

Two approaches were compared. **Approach A** is a generic extractive summarizer that ranks source sentences by term frequency and copies the best sentences. Because it copies OCR text, it is conservative and produced a measured hallucination rate of **0.000**. **Approach B** is a field-aware structured summary that converts extracted critical fields into 3–5 concise sentences. Its hallucination rate on the test set was **0.125**. Mean ROUGE-L was **0.260** for extractive summaries versus **0.984** for field-aware summaries. More importantly for this task, critical-field completeness averaged **0.354** versus **0.938**. The field-aware approach is therefore the better operational summary when extraction is correct; the extractive approach is useful as a conservative fallback.

## End-to-end throughput and bottleneck

The pipeline processed **8 documents at 17.24 documents/minute**, averaging **3.48 seconds per document** in this environment. Stage timings identify **ocr_seconds** as the dominant bottleneck. That is expected because image OCR performs far more computation than regex/section parsing or the lightweight summarizers. Production scaling should therefore prioritize OCR batching/parallelism, page-level caching, and avoiding repeated OCR of unchanged documents.

## Responsible handling and production readiness

Only synthetic data was processed. Real legal deployments would require confidentiality controls, least-privilege access, encryption, audit trails, retention/deletion policies, secure temporary files, and review of applicable privacy and professional-responsibility obligations. OCR confidence and field provenance should be visible to reviewers. Low-confidence pages, ambiguous extractions, and summaries containing uncertain values should be explicitly flagged. The current build is a reproducible internship prototype, not a system for autonomous legal decision-making.
