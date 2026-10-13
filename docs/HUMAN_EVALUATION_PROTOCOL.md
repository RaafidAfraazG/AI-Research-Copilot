# Human Evaluation Protocol for AI Research Copilot

This protocol defines the scientific rubric, methodology, and evaluation templates for manual human evaluation of research answers produced by the AI Research Copilot system.

> [!IMPORTANT]
> Human evaluation is a manual, qualitative verification process. Automated scripts must never fabricate human evaluation scores. Scores must only be recorded when genuine human raters have reviewed the answers according to this protocol.

---

## 1. Evaluation Dimensions & 1–5 Scoring Rubric

Each generated answer is evaluated across 5 distinct dimensions on an integer scale from 1 (poor/unacceptable) to 5 (excellent/exemplary).

**Do NOT compute a single blended average score** across these dimensions, as each measures an independent axis of research quality.

### 1. Relevance
*Does the answer directly address the user's specific research question without extraneous tangents?*

| Score | Rating | Definition |
| :---: | :--- | :--- |
| **1** | Irrelevant | Fails completely to address the core research question. |
| **2** | Low | Tangentially mentions topic keywords but misses the primary query objective. |
| **3** | Moderate | Answers the primary question but includes noticeable irrelevant digressions. |
| **4** | High | Directly and clearly addresses the core research question with minimal fluff. |
| **5** | Exemplary | Laser-focused, directly answering the exact nuances of the research question. |

### 2. Completeness
*Does the answer cover the major technical dimensions and sub-topics expected for this research question?*

| Score | Rating | Definition |
| :---: | :--- | :--- |
| **1** | Incomplete | Completely misses major required technical concepts. |
| **2** | Minimal | Covers only 1 minor aspect, omitting critical background or comparison. |
| **3** | Partial | Covers some primary aspects but leaves noticeable technical gaps. |
| **4** | Substantial | Covers nearly all expected research dimensions with adequate depth. |
| **5** | Comprehensive | Thoroughly addresses all key facets, trade-offs, and necessary context. |

### 3. Groundedness
*Are the factual claims in the answer strictly supported by the provided retrieved evidence passages?*

| Score | Rating | Definition |
| :---: | :--- | :--- |
| **1** | Unsubstantiated | Contains pervasive unverified or hallucinated assertions unsupported by evidence. |
| **2** | Poor Grounding | Majority of factual claims cannot be verified against the provided evidence passages. |
| **3** | Moderate | Most claims are supported, but 1 or 2 unsubstantiated factual statements remain. |
| **4** | High Grounding | All primary factual claims are verified in retrieved evidence; minor neutral transitions. |
| **5** | Fully Grounded | 100% of factual assertions are strictly substantiated by the cited evidence passages. |

### 4. Citation Quality
*Do bibliographic citation markers (`[1]`, `[2]`, etc.) accurately point to the specific passage supporting each claim?*

| Score | Rating | Definition |
| :---: | :--- | :--- |
| **1** | False Citations | Citations link to passages that do not contain or contradict the asserted claim. |
| **2** | Misleading | Citations point to general documents that only vaguely touch upon the assertion. |
| **3** | Plausible | Citations point to roughly relevant documents, but specific passage is weak. |
| **4** | Accurate | Citations point directly to passages containing factual corroboration. |
| **5** | Precise & Traceable | Every claim has exact passage-level attribution with verifiable document provenance. |

### 5. Clarity & Organization
*Is the answer well-structured, mathematically and conceptually coherent, and professionally written?*

| Score | Rating | Definition |
| :---: | :--- | :--- |
| **1** | Incoherent | Disorganized, contradictory, or unreadable prose. |
| **2** | Confusing | Poorly structured with confusing technical explanations or repetitive phrasing. |
| **3** | Acceptable | Understandable structure, but awkward transitions or minor grammatical issues. |
| **4** | Clear | Clean logical flow, well-structured sections, and professional academic tone. |
| **5** | Publication-Grade | Exceptionally well-structured, elegant synthesis, concise and authoritative. |

---

## 2. Evaluation Procedure

1. **Rater Selection**: Evaluators should have foundational domain knowledge in Computer Science, Machine Learning, Information Retrieval, or related technical disciplines.
2. **Blind Assessment**: When comparing system variants, raters should ideally be blind to the variant identity (e.g. evaluating anonymous outputs labeled `Answer A` vs `Answer B`).
3. **Evidence Verification**: Raters must inspect both the generated text and the underlying retrieved passages in the `EvidencePack` before scoring Groundedness and Citation Quality.
4. **Recording**: Scores must be logged into the standard template below.

---

## 3. Human Evaluation Template

```markdown
### Evaluation Entry

- **Evaluation Date:** YYYY-MM-DD
- **Evaluator ID / Initials:** [Rater Name or Anonymized ID]
- **Variant:** [e.g. Full Pipeline / Hybrid Baseline]
- **Query ID:** [e.g. ans_01_factual_rag]
- **Query Text:** "[Exact query string]"

| Dimension | Score (1–5) | Notes & Observations |
| :--- | :---: | :--- |
| **Relevance** | [1-5] | [Specific justification] |
| **Completeness** | [1-5] | [Expected dimensions covered/missed] |
| **Groundedness** | [1-5] | [Identified unsupported claims, if any] |
| **Citation Quality** | [1-5] | [Accuracy of passage references] |
| **Clarity** | [1-5] | [Readability and flow notes] |

- **Overall Qualitative Summary:** [Brief summary of strengths and weaknesses]
```
