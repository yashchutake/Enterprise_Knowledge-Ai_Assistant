"""
RAGAS evaluation for the Response Agent's output.

Mandatory metrics (computed live, per query, no ground truth required):
  - Faithfulness       -> is the answer grounded in the retrieved context?
  - Answer Relevancy    -> does the answer actually address the question?

Bonus metrics (Context Precision / Context Recall) require a ground-truth
reference answer and are computed offline in evaluation/offline_eval.py
against data/eval_dataset.json, not on every live query.
"""
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from .. import config

_eval_embeddings = None


def get_eval_llm() -> LangchainLLMWrapper:
    chat_model = ChatGoogleGenerativeAI(
        model=config.GEMINI_CHAT_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0,
    )
    return LangchainLLMWrapper(chat_model)


def get_eval_embeddings() -> LangchainEmbeddingsWrapper:
    global _eval_embeddings
    if _eval_embeddings is None:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )
        _eval_embeddings = LangchainEmbeddingsWrapper(embeddings_model)
    return _eval_embeddings


def _get_metrics():
    faithfulness_metric = Faithfulness(llm=get_eval_llm())
    answer_relevancy_metric = ResponseRelevancy(
        llm=get_eval_llm(), embeddings=get_eval_embeddings()
    )
    return faithfulness_metric, answer_relevancy_metric


async def evaluate_response(question: str, answer: str, contexts: list[str]) -> dict:
    """Score a single question/answer/context triple with RAGAS."""
    if not answer or not contexts:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}

    faithfulness_metric, relevancy_metric = _get_metrics()

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    try:
        faithfulness_score = await faithfulness_metric.single_turn_ascore(sample)
    except Exception as exc:  # noqa: BLE001
        print(f"[RAGAS] Faithfulness scoring failed: {exc}")
        faithfulness_score = None

    try:
        relevancy_score = await relevancy_metric.single_turn_ascore(sample)
    except Exception as exc:  # noqa: BLE001
        print(f"[RAGAS] Answer relevancy scoring failed: {exc}")
        relevancy_score = None

    return {
        "faithfulness": round(faithfulness_score, 3) if faithfulness_score is not None else None,
        "answer_relevancy": round(relevancy_score, 3) if relevancy_score is not None else None,
    }
