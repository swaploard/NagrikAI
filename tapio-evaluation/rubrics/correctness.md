# Evaluation Rubric: Factual Correctness (India GST)

This rubric evaluates how accurately the assistant's response reflects the provided ground-truth reference answer. Given the legal, high-stakes nature of tax compliance in India, precise numbers, statutory sections, monetary thresholds, rules, tax rates, and filing deadlines are critical.

## Grading Scale (1–5)

### Score 5: Excellent / Completely Correct

- **Factual Alignment:** The generated response is entirely accurate, fully aligns with the reference answer, and contains no contradictions.
- **Precision:** All tax rates (e.g., 5%, 12%, 18%, 28%), monetary thresholds (e.g., "₹5 Crore aggregate turnover for e-invoicing" or "₹20/40 Lakhs for registration"), legal citations (e.g., "Section 16(2)(aa) of the CGST Act" or "Rule 86B"), and filing timelines (e.g., "11th of the following month for GSTR-1") match the reference exactly.
- **Completeness:** No vital conditions, exceptions, blocked credit clauses (Section 17(5)), or provisos mentioned in the reference are missing.

### Score 4: Good / Mostly Correct

- **Factual Alignment:** The core answer is accurate, but it has minor omissions that do not compromise the overall compliance advice.
- **Precision:** Primary tax liabilities and rates are correct, but secondary conditions, specific notification numbers (e.g., Notification No. 12/2017-Central Tax), or minor state-specific SGST exceptions might be slightly understated or generalized.
- **Hallucinations:** No false information, incorrect legal sections, or fictitious tax rates are introduced.

### Score 3: Partially Correct

- **Factual Alignment:** The response captures some correct elements but misses crucial tax caveats, conditions, or secondary requirements.
- **Omissions:** Fails to state an important statutory condition (e.g., states that ITC on food and beverages is blocked, but fails to mention the exception when it is obligatory for an employer to provide it to employees under any law currently in force).
- **Inaccuracies:** Minor factual or interest rate confusion is present (e.g., misstating the interest rate under Section 50(1) as 24% instead of 18% for delayed tax payments) but does not lead to complete compliance failure.

### Score 2: Poor / Minor Factual Errors

- **Factual Alignment:** The response contains at least one significant factual error or major legal misstatement.
- **Contradiction:** Contradicts the reference on essential facts, putting the taxpayer at risk of severe penalties or incorrect filings (e.g., claiming that composition taxpayers can claim Input Tax Credit, or misstating a deadline in a way that would trigger automatic system-generated late fees).

### Score 1: Fail / Highly Incorrect or Hallucinated

- **Factual Alignment:** The response is completely wrong, heavily hallucinated, or irrelevant.
- **Severe Inaccuracy:** Cites incorrect tax rates (e.g., stating a non-existent 40% GST rate for basic items), completely fictitious legal sections (e.g., "Section 215 of the CGST Act"), or non-existent GSTR forms.

---

## Evaluation Guidance for LLM Judge

```json
{
  "instruction": "Compare the [Generated Answer] to the [Reference Answer] for an Indian GST query. Pay absolute attention to legal citations (CGST/SGST/IGST Acts and Rules), monetary thresholds, specific GSTR filing deadlines, tax rates, and blocked credit exceptions. Output your evaluation strictly in the defined JSON format.",
  "json_schema": {
    "factual_correctness_score": "integer (1-5)",
    "reasoning": "string"
  }
}
```
