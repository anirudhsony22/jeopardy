# Jeopardy - Get relevant wikipedia document for your question

**Owner:** Anirudha Soni  
**Full Report:** [Project Report](https://docs.google.com/document/d/1hGXzrfwM0spRU-2zFPhOaqKzWD7p_7kgl-mSVvFfges/edit?usp=sharing)

## Project Details

- **Data**: ~280,000 Wikipedia articles  
- **Tokens**: ~123 million  
- **Questions**: 100 (with answers)  
- **Evaluation Targets**:  
  - Recall@1 ≥ 40%  
  - MRR ≥ 50%

## Pipeline Overview

1. Preprocessing
2. Chunking
3. Embedding
4. Indexing
5. Merging
6. Re-ranking

 ![Pipeline Optimized](https://drive.google.com/file/d/1aY0dGBn_j_slWIhBEPjTS4Zby28CgdZq/view?usp=sharing)
 ![Pipeline Old](https://drive.google.com/file/d/1dr43CGHaHrbiSJdre1fjeeUZeFBF3Nd5/view?usp=sharing)

## ⚠️ Notes

- Index files are **not included** in this repository due to storage constraints.
