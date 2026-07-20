# Evaluation Rubric: Retrieval Quality & Context Relevance (India GST)

This rubric evaluates the quality, relevance, and source attribution of the retrieved GST context chunks before they are sent to the generation step. This directly measures the accuracy of your document chunking, indexing, and embedding pipeline.

## Grading Scale (1–4)

### Score 4: Highly Relevant & Precise (Excellent)

- **Topic Matching:** All retrieved chunks directly address the core tax question and the specific taxpayer persona (e.g., retrieving the correct composition scheme rules for a small retail store query).
- **Statute & Authority Precision:** The retriever fetched documents from the exact legal authority or statutory body matching the query (e.g., CGST Act for central tax rules, IGST Act for place of supply queries, CBIC circulars for departmental clarifications, or GSTN manuals for technical portal issues). No mismatched legal regimes.
- **Noise Level:** The chunks are compact, highly dense with information, and contain minimal unrelated sections (e.g., they cleanly capture a specific proviso without trailing off into unrelated sections).

### Score 3: Partially Relevant with Some Noise (Good)

- **Topic Matching:** The correct target sections or circulars are retrieved, but the chunking strategy has introduced significant irrelevant surrounding text (e.g., the chunk contains the target sub-clause but also overlaps too heavily into adjacent, unrelated sections).
- **Statute & Authority Precision:** The correct act or circular is present, but irrelevant acts (e.g., retrieving general SGST provisions for an interstate transaction query) are also ranked highly in the top results.

### Score 2: Insufficient or Misaligned Context (Poor)

- **Missing Information:** The retriever missed the specific proviso, notification, or exception required to resolve the query (e.g., retrieved general Section 16 requirements but missed the _Rule 36(4)_ GSTR-2B matching restriction chunk).
- **Cross-Statute/Authority Confusion:** The retriever fetched information from the wrong legal framework (e.g., returning IGST Act chapters for a purely intra-state CGST/SGST query, or pulling Income Tax/Direct Tax rules for a GST query).

### Score 1: Entirely Irrelevant or Empty (Fail)

- **Topic Matching:** The retrieved chunks have no relationship to the query.
- **Missing Targets:** None of the expected gold-standard statutory sections, notification numbers, or document hashes specified in your test set (e.g., `expected_documents.json`) were retrieved in the top K results.

---

## Evaluation Guidance for LLM Judge

```json
{
  "instruction": "Compare the retrieved [Context Chunks] against the [User Query]. Assess whether the chunks provide the specific statutory sections, rules, or circulars required without mixing up different tax acts (e.g. CGST vs. IGST vs. Income Tax).",
  "json_schema": {
    "retrieval_relevance_score": "integer (1-4)",
    "detected_noise_level": "string (None/Low/High)",
    "statute_or_authority_mismatch": "boolean",
    "reasoning": "string"
  }
}
```
