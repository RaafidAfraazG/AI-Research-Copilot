# HISTORICAL — NOT A MEASUREMENT OF THE REAL-MODEL PIPELINE

These Phase 7 reports (2026-09-29) were produced with:

* a **mock hash embedder** (dense / hybrid numbers are meaningless),
* the **fallback (no-op) reranker** — "Variant D" is identical to "Variant C" by construction,
* the **MockLLM** for answer metrics,
* an index polluted with test fixtures and ~46 duplicate chunks (56 chunks / 8 documents).

They are kept only for provenance. Do not cite them. Current measurements are in
`../latest.md` (Phase 8.6, real `all-MiniLM-L6-v2` index).
