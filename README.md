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

### Faster Pipeline  
*Lower latency, slight loss in accuracy (3% Precision@1 and 3% MRR loss)*

 ![Pipeline Optimized](https://private-user-images.githubusercontent.com/44290537/451571219-d20c629b-e2f7-489b-aa3d-a6a5171e8730.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDkwNzA4MzIsIm5iZiI6MTc0OTA3MDUzMiwicGF0aCI6Ii80NDI5MDUzNy80NTE1NzEyMTktZDIwYzYyOWItZTJmNy00ODliLWFhM2QtYTZhNTE3MWU4NzMwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA2MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNjA0VDIwNTUzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWY1MDllOTY2MjY4YjU3NmNhY2UwOWFhZjYxNWY0MGY2OTFkNWM4OGRmYWMzMDUzOTg2YzllNTg2NjMzODRkMmImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.xE7ENdcbpbNc2JLH02ltfSosM6Hlf0izIlSYubMXRaA)


### 🔍 Original Pipeline  
*More accurate (3% Precision@1 and 3% MRR gain), but slightly slower*

 ![Pipeline Old](https://private-user-images.githubusercontent.com/44290537/451571383-fa894419-b0d3-49b1-b80b-7ec86f527ee0.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDkwNzA4MzIsIm5iZiI6MTc0OTA3MDUzMiwicGF0aCI6Ii80NDI5MDUzNy80NTE1NzEzODMtZmE4OTQ0MTktYjBkMy00OWIxLWI4MGItN2VjODZmNTI3ZWUwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA2MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNjA0VDIwNTUzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI5ZDZmNDAwZjYxYzMyMmZhMDBkOWE4MzFjODU0ZDNhMjQ2NDg0NjNiZWM1OTUzNjE0NWMyYzBlOWJiZWMyNmEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.NWh5Xgvv7pYhkkRtYH-S4HAMVVToAjsuXFMbC16R50A)

## ⚠️ Notes

- Index files are **not included** in this repository due to storage constraints.
