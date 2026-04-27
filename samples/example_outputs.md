# Example outputs — both spec sample inputs (PDF + JSON)

Captured runs of the bot against the two sample inputs referenced in the challenge doc:

1. **PDF** — the SOC2 Type 2 report at https://productfruits.com/docs/soc2-type2.pdf
2. **JSON** — the Q&A knowledge-base spreadsheet at https://docs.google.com/spreadsheets/d/1u7z18yNKsL8cMLV6OxYI1-8ageRfFG1j/edit, exported and converted to JSON (committed at `samples/spec_kb.json`)

Both runs use the **same 5 questions** from the spec (`samples/spec_questions.json`) and the **same code path**. The contrast is what to look for: the PDF describes a company on AWS in Europe; the JSON describes a different company on GCP in the US. The bot's answers track each document — proving the answers are grounded in the input, not the model's training data.

Local model (`llama3.1:8b` via Ollama) used because the OpenAI key shared in the doc returned `insufficient_quota`; this was confirmed acceptable by Shruti. The same code serves `gpt-4o-mini` when `OPENAI_BASE_URL` is unset.

---

## Run 1 — PDF (SOC2 report)

**Reproduce:**

```bash
curl -L -o /tmp/soc2.pdf https://productfruits.com/docs/soc2-type2.pdf
curl -X POST "http://localhost:8000/qa?verbose=true" \
  -F "document=@/tmp/soc2.pdf" \
  -F "questions=@samples/spec_questions.json"
```

Total time: ~46s.

### Q1. Do you have formally defined criteria for notifying a client during an incident that might impact the security of their data or systems? What are your SLAs for notification?

**A1.** Yes, the entity has documented guidelines on notifying customers and other stakeholders in case of a breach (CC7.3.10). The Data Breach Notification Policy is inspected to determine this. Additionally, the Information Security Policy provides information to employees on how to report failures, incidents, concerns, or other complaints related to the services or systems provided by the entity, which may include procedures for notifying clients during an incident.

The specific SLAs (Service Level Agreements) for notification are not explicitly stated in the provided excerpts.

*retrieval_score: 0.755 · cited pages: 18, 38, 49*

### Q2. Is personal information transmitted, processed, stored, or disclosed to or retained by third parties? If yes, describe.

**A2.** Yes, the context indicates that confidentiality applies to various types of sensitive information, including personal information. According to [4], "Confidential information may include personal information as well as other information, such as trade secrets and intellectual property." However, it does not specify whether this information is transmitted, processed, stored, or disclosed to third parties.

*retrieval_score: 0.715 · cited pages: 23, 24, 33*

### Q3. Which cloud providers do you rely on?

**A3.** Product Fruits s.r.o. uses Amazon Web Services Inc. (AWS) as a subservice organization to provide cloud services.

*retrieval_score: 0.643 · cited pages: 5, 9, 12, 25*

### Q4. Please specify the primary data center location/region of the underlying cloud infrastructure used to host the service(s) as well as the backup location(s).

**A4.** The production infrastructure for the Product Fruits software application is hosted on AWS in their various regions across Europe.

Note that specific region(s) are not specified, though it mentions "various regions" which implies multiple locations within Europe. The backup location(s) are not mentioned.

*retrieval_score: 0.643 · cited pages: 9, 13*

### Q5. Which of the following, if any, are performed as part of your monitoring process for the service: APM / EUM / DEM?

**A5.** The provided context does not contain enough information to answer this question.

*retrieval_score: 0.712 · cited pages: 23, 32, 38, 48*

---

## Run 2 — JSON (Q&A knowledge base)

**Reproduce:**

```bash
curl -X POST "http://localhost:8000/qa?verbose=true" \
  -F "document=@samples/spec_kb.json" \
  -F "questions=@samples/spec_questions.json"
```

Total time: ~34s.

### Q1. Do you have formally defined criteria for notifying a client during an incident that might impact the security of their data or systems? What are your SLAs for notification?

**A1.** The provided context does not contain enough information to answer this question.

*retrieval_score: 0.642*

### Q2. Is personal information transmitted, processed, stored, or disclosed to or retained by third parties? If yes, describe.

**A2.** The knowledge base entries do not explicitly mention the transmission, processing, storage, or disclosure of personal information to third parties. However, it does state that there is an Identity and Access Management System in place that facilitates access to roles, which may involve interactions with external systems or services. The specific details regarding personal information handling are not addressed in this knowledge base.

*retrieval_score: 0.606*

### Q3. Which cloud providers do you rely on?

**A3.** Our data centers are located in the US Central region, specifically hosted within Google Cloud Platform (GCP). All customer data is physically stored in the USA. We leverage GCP for our data hosting, ensuring that all data is managed within their secure data centers.

*retrieval_score: 0.553*

### Q4. Please specify the primary data center location/region of the underlying cloud infrastructure used to host the service(s) as well as the backup location(s).

**A4.** Our data centers are located in the US Central region, specifically hosted within Google Cloud Platform (GCP). All customer data is physically stored in the USA.

Note that specific information about a backup location is not provided in the given context.

*retrieval_score: 0.687*

### Q5. Which of the following, if any, are performed as part of your monitoring process for the service: APM / EUM / DEM?

**A5.** The provided context does not contain enough information to answer this question.

*retrieval_score: 0.602*

---

## Why the two runs differ — and why that's the point

| Question | PDF answer | JSON answer |
| --- | --- | --- |
| Cloud providers | **AWS** | **GCP** |
| Data center region | **Europe** | **US Central** |
| Incident notification SLA | Found CC7.3.10 + Data Breach policy | Refused — KB doesn't cover SLAs |
| PII to third parties | Found "confidential" definition; honestly hedged | Pulled IAM context; honestly hedged |
| APM/EUM/DEM | Refused | Refused |

**Same questions. Same bot. Same prompt. Different documents → different answers.** That's grounded retrieval working as intended. If the bot were hallucinating from training, the answers would converge regardless of input.

Notice also that the bot **refuses identically** on Q5 in both runs — neither document mentions APM/EUM/DEM specifically, so the system prompt's "answer only from context" rule fires the canonical refusal sentence in both cases.
