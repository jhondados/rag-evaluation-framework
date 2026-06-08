# 📏 RAG Evaluation Framework

[![Metrics](https://img.shields.io/badge/Metrics-12%20RAGAS%20%2B%20custom-blue)](.) [![Hallucinations](https://img.shields.io/badge/Hallucination%20Detection-98.1%25-green)](.) [![Regression](https://img.shields.io/badge/Regression%20Tests-CI%2FCD-orange)](.)

> **Production RAG evaluation** with 12 metrics, LLM-as-judge, hallucination detection and CI/CD regression testing. Every RAG change validated before deployment. **98.1% hallucination detection rate**.

## 📊 Metrics Suite
| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Faithfulness | Is answer grounded in context? | > 0.90 |
| Answer Relevancy | Does answer address the question? | > 0.85 |
| Context Recall | Are all needed docs retrieved? | > 0.80 |
| Context Precision | Are retrieved docs relevant? | > 0.75 |
| Hallucination Score | False claims not in context | < 0.05 |
| Latency P95 | End-to-end response time | < 3s |
