"""RAG evaluation with RAGAS metrics."""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from typing import List, Dict
import pandas as pd

class RAGEvaluator:
    def __init__(self):
        self.llm = ChatVertexAI(model_name="gemini-1.5-pro-002")
        self.embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
        self.metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    def evaluate_rag(self, questions: List[str], answers: List[str],
                     contexts: List[List[str]], ground_truths: List[str]) -> Dict:
        dataset = Dataset.from_dict({"question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths})
        result = evaluate(dataset=dataset, metrics=self.metrics,
            llm=self.llm, embeddings=self.embeddings)
        df = result.to_pandas()
        return {"mean_faithfulness": float(df["faithfulness"].mean()),
                "mean_answer_relevancy": float(df["answer_relevancy"].mean()),
                "mean_context_recall": float(df["context_recall"].mean()),
                "mean_context_precision": float(df["context_precision"].mean()),
                "overall_score": float(df[["faithfulness","answer_relevancy","context_recall","context_precision"]].mean().mean()),
                "low_faith_examples": df[df["faithfulness"] < 0.7][["question","answer","faithfulness"]].to_dict("records")}

    def detect_hallucination(self, answer: str, context: str) -> Dict:
        """LLM-as-judge hallucination detection."""
        prompt = f"""Determine if the answer contains hallucinations (claims not supported by context).
Context: {context[:2000]}
Answer: {answer}
Return JSON: {{"hallucination_detected": bool, "confidence": 0.0-1.0, "unsupported_claims": [list]}}"""
        import json
        resp = self.llm.invoke(prompt).content
        data = json.loads(resp.split("```json")[-1].split("```")[0] if "```" in resp else resp)
        return data
