# Example output — spec sample (SOC2 PDF + 5 spec questions)

Captured by running the bot against the spec-provided sample SOC2 PDF and `samples/spec_questions.json`. Local model (`llama3.1:8b` via Ollama) used because the OpenAI key shared in the challenge doc returned `insufficient_quota`; this was confirmed acceptable by Shruti. The same code path serves `gpt-4o-mini` when `OPENAI_BASE_URL` is unset.

**Reproduce:**

```bash
curl -L -o /tmp/soc2.pdf https://productfruits.com/docs/soc2-type2.pdf
curl -X POST "http://localhost:8000/qa?verbose=true" \
  -F "document=@/tmp/soc2.pdf" \
  -F "questions=@samples/spec_questions.json"
```

Total time: ~46s for 5 questions on Apple Silicon. Stateful re-queries (`POST /documents/{id}/questions`) reuse the index and skip re-embedding.

---

## Q1. Do you have formally defined criteria for notifying a client during an incident that might impact the security of their data or systems? What are your SLAs for notification?

**A1.** Yes, the entity has documented guidelines on notifying customers and other stakeholders in case of a breach (CC7.3.10). The Data Breach Notification Policy is inspected to determine this. Additionally, the Information Security Policy provides information to employees on how to report failures, incidents, concerns, or other complaints related to the services or systems provided by the entity, which may include procedures for notifying clients during an incident.

The specific SLAs (Service Level Agreements) for notification are not explicitly stated in the provided excerpts.

*retrieval_score: 0.755 · cited pages: 18, 38, 49*

---

## Q2. Is personal information transmitted, processed, stored, or disclosed to or retained by third parties? If yes, describe.

**A2.** Yes, the context indicates that confidentiality applies to various types of sensitive information, including personal information. According to [4], "Confidential information may include personal information as well as other information, such as trade secrets and intellectual property." However, it does not specify whether this information is transmitted, processed, stored, or disclosed to third parties.

*retrieval_score: 0.715 · cited pages: 23, 24, 33*

---

## Q3. Which cloud providers do you rely on?

**A3.** Product Fruits s.r.o. uses Amazon Web Services Inc. (AWS) as a subservice organization to provide cloud services.

*retrieval_score: 0.643 · cited pages: 5, 9, 12, 25*

---

## Q4. Please specify the primary data center location/region of the underlying cloud infrastructure used to host the service(s) as well as the backup location(s).

**A4.** The production infrastructure for the Product Fruits software application is hosted on AWS in their various regions across Europe.

Note that specific region(s) are not specified, though it mentions "various regions" which implies multiple locations within Europe. The backup location(s) are not mentioned.

*retrieval_score: 0.643 · cited pages: 9, 13*

---

## Q5. Which of the following, if any, are performed as part of your monitoring process for the service: Application Performance Monitoring (APM), End User Monitoring (EUM), Digital Experience Monitoring (DEM)?

**A5.** The provided context does not contain enough information to answer this question.

*retrieval_score: 0.712 · cited pages: 23, 32, 38, 48*

---

## Notes on the answers

- **Refusals are intentional, not bugs.** Q5 refused even though some monitoring-related chunks scored above the similarity floor (0.712) because none of them mentioned APM/EUM/DEM specifically. The system prompt requires the bot to refuse rather than guess. A larger model (`gpt-4o-mini`) would likely answer Q5 by partially addressing what's there and noting what isn't.
- **Honest hedging on Q2 and Q4.** The bot identifies what's in the context, then explicitly flags what isn't — exactly what a compliance bot should do for an auditor.
- **Page citations** are surfaced via `?verbose=true`. Each cited page is the source of one of the top-k retrieved chunks.
