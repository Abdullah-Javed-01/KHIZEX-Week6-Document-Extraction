# Summarization Comparison

Two document-summary approaches were evaluated on the eight synthetic legal agreements.

| Approach | Mean ROUGE-L | Mean critical-field completeness | Hallucination rate |
|---|---:|---:|---:|
| Frequency-ranked extractive | 0.260 | 0.354 | 0.000 |
| Field-aware structured | 0.984 | 0.938 | 0.125 |

## Approach A — extractive

The extractive method ranks source sentences using term frequency and copies the highest-scoring sentences. Because it only copies OCR text, it is conservative and did not introduce facts absent from the OCR source in this sample. Its weakness is coverage: important fields such as the effective date, governing law, or provider obligation can be omitted when those sentences do not score highly.

## Approach B — field-aware structured

The field-aware method converts the extracted party names, date, fee, obligation, and governing law into a short structured summary. It achieved much higher field completeness and ROUGE-L because it is directly aligned with the critical fields required by the task. Its main risk is propagation: when an extracted field is wrong, the summary can faithfully restate that wrong value, which is counted as a hallucination/faithfulness failure relative to the source truth.

## Example

For `contract_06`, the structured summary correctly captured the parties, June 7, 2026 effective date, $42,300 fee, weekly exception-report obligation, and Massachusetts governing law after preprocessing improved OCR quality.

**Conclusion:** the field-aware summary is the better operational output when extraction confidence is acceptable; the extractive method is a useful conservative fallback when fields are uncertain.
