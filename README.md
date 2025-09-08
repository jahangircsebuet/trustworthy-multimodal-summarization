# trustworthy-multimodal-summarization
Trustworthy Multimodal Summarization (RAG + Verification)


flowchart TB
  %% ====== SOURCES ======
  subgraph S[Data Sources]
    T[Social Post Text]
    I[Images / Memes]
    V[Short Videos]
    KB[(External KB\n(e.g., Wikipedia/News))]
  end

  %% ====== PERCEPTION ======
  subgraph P[Perception & Normalization]
    OCR[OCR\n(Tesseract/EasyOCR)]
    CAP[Image Captioning\n(BLIP/BLIP-2)]
    ASR[ASR\n(Faster-Whisper)]
    TRN[Translate → EN\n(optional)]
    PK[Pack Textbag\n(merge text+OCR+ASR+captions)]
  end

  %% ====== RETRIEVAL ======
  subgraph R[Retrieval]
    CHK[Chunk to Passages]
    EMB[Embed\n(all-MiniLM-L6-v2)]
    IDX[(FAISS Index)]
    RET[Retrieve k Snippets\n(who/what/where/when/why)]
  end

  %% ====== GENERATION ======
  subgraph G[Grounded Generation]
    PR[Prompt Builder\n(EVIDENCE + rules + [refN])]
    GEN[Draft Summary\n(Flan-T5 / small seq2seq)]
  end

  %% ====== VERIFICATION ======
  subgraph VFY[Safety Guards & Revision]
    QG[Question Generation\n(T5 QG)]
    QA[QA over Evidence\n(SQuAD2)]
    NLI[NLI\n(MNLI)]
    CLIP[Image–Text Consistency\n(CLIP) (optional)]
    AGR[Guard Aggregation\n(ok / revise / drop)]
    REV[LLM Revision\n(remove/hedge, keep citations)]
  end

  %% ====== EVAL/OUTPUT ======
  subgraph E[Evaluation & Logging]
    MET[Metrics\n(ROUGE, BERTScore, Guard-score)]
    LOG[(Artifacts: prompts,\nindexes, drafts, revisions)]
    OUT[[Final Trusted Summary\n(with citations)]]
  end

  %% WIRING
  T --> TRN
  I --> OCR
  I --> CAP
  V --> ASR
  T --> PK
  TRN --> PK
  OCR --> PK
  ASR --> PK
  CAP --> PK

  KB -. optional .-> EMB

  PK --> CHK --> EMB --> IDX --> RET
  KB --> EMB

  RET --> PR --> GEN
  RET --> QA
  GEN --> QG --> QA
  GEN --> NLI
  I -. optional .-> CLIP
  GEN --> CLIP

  QA --> AGR
  NLI --> AGR
  CLIP --> AGR
  GEN --> AGR

  AGR --> REV --> OUT
  OUT --> MET
  AGR --> MET
  RET --> MET

  %% Artifacts
  IDX --- LOG
  PR --- LOG
  GEN --- LOG
  REV --- LOG
  MET --- LOG
