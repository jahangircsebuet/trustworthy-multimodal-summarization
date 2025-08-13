from transformers import pipeline
_QA = pipeline("question-answering", model="deepset/roberta-base-squad2", device_map="auto")

def answer_with_context(q: str, context: str) -> dict:
    try:
        ans = _QA(question=q, context=context, handle_impossible_answer=True, top_k=1)
        return {"answer": ans.get("answer",""), "score": float(ans.get("score",0.0))}
    except Exception:
        return {"answer": "", "score": 0.0}
