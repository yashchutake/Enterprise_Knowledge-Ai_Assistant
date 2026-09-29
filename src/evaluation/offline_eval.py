"""
Bonus RAGAS metrics: Context Precision and Context Recall.

These metrics need a ground-truth reference answer, so they are run offline
against data/eval_dataset.json rather than on every live user query.

Run with:
    python -m src.evaluation.offline_eval
"""
import asyncio
import json
import os

from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

from .. import config
from ..rag.vectorstore import get_retriever
from ..agents.response_agent import response_node
from .ragas_eval import get_eval_llm

EVAL_DATASET_PATH = os.path.join(config.BASE_DIR, "data", "eval_dataset.json")


async def run_offline_evaluation():
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    retriever = get_retriever()
    llm = get_eval_llm()
    precision_metric = LLMContextPrecisionWithReference(llm=llm)
    recall_metric = LLMContextRecall(llm=llm)

    results = []
    for item in eval_items:
        question = item["question"]
        reference = item["ground_truth"]

        docs = await retriever.ainvoke(question)
        contexts = [d.page_content for d in docs]

        response_state = await response_node({"query": question, "contexts": contexts})
        answer = response_state["answer"]

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        )

        try:
            precision = await precision_metric.single_turn_ascore(sample)
        except Exception as exc:  # noqa: BLE001
            print(f"[RAGAS] context_precision failed for '{question}': {exc}")
            precision = None

        try:
            recall = await recall_metric.single_turn_ascore(sample)
        except Exception as exc:  # noqa: BLE001
            print(f"[RAGAS] context_recall failed for '{question}': {exc}")
            recall = None

        results.append(
            {
                "question": question,
                "answer": answer,
                "context_precision": round(precision, 3) if precision is not None else None,
                "context_recall": round(recall, 3) if recall is not None else None,
            }
        )

    return results


def main():
    results = asyncio.run(run_offline_evaluation())
    print("\n=== Bonus RAGAS Evaluation: Context Precision & Context Recall ===\n")
    for r in results:
        print(json.dumps(r, indent=2))
    if results:
        valid_p = [r["context_precision"] for r in results if r["context_precision"] is not None]
        valid_r = [r["context_recall"] for r in results if r["context_recall"] is not None]
        if valid_p:
            print(f"\nAverage Context Precision: {round(sum(valid_p) / len(valid_p), 3)}")
        if valid_r:
            print(f"Average Context Recall:    {round(sum(valid_r) / len(valid_r), 3)}")


if __name__ == "__main__":
    main()
