Legacy (pre-Phase-8.6) stores in this tree are NOT read by the application any more:
  data/opensearch_local/, data/qdrant_local/, backend/data/
They contain mock-embedding vectors and test pollution. Verified, hash-manifested copies are in
data/backups/. The active index lives in data/indexes/ (see docs/ARCHITECTURE.md section 5.5).
They are kept only until you decide to remove them.
