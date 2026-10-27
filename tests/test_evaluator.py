from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from enterprise_rag.evaluator import evaluate_response

CONTEXT = "Employees receive 28 paid vacation days per year."
QUESTION = "How many paid vacation days do employees receive?"


class JsonJudge(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "json-judge"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        payload = (
            '{"faithfulness": 0.91, "answer_relevancy": 0.8, '
            '"context_precision": 0.7, "reasoning": "The answer stays inside the passage."}'
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=payload))])


def test_faithful_answer_scores_higher_than_an_unrelated_one() -> None:
    faithful = evaluate_response(
        question=QUESTION,
        answer="Employees receive 28 paid vacation days per year. [1]",
        contexts=[CONTEXT],
        ground_truth=CONTEXT,
    )
    invented = evaluate_response(
        question=QUESTION,
        answer="The cafeteria serves purple lasagna on Mars every Friday. [9]",
        contexts=[CONTEXT],
    )
    assert faithful.faithfulness > 0.8
    assert faithful.faithfulness > invented.faithfulness
    assert faithful.citation_support == 1.0
    assert invented.citation_support == 0.0
    assert faithful.context_recall == 1.0
    assert faithful.answer_correctness is not None and faithful.answer_correctness > 0.8
    assert faithful.mode == "lexical"


def test_relevancy_prefers_an_answer_that_uses_the_question() -> None:
    relevant = evaluate_response(
        question=QUESTION,
        answer="Employees receive 28 paid vacation days per year. [1]",
        contexts=[CONTEXT],
    )
    aside = evaluate_response(
        question=QUESTION,
        answer="Printers are refilled on Tuesdays. [1]",
        contexts=[CONTEXT],
    )
    assert relevant.answer_relevancy > aside.answer_relevancy


def test_llm_judge_scores_are_parsed_without_a_network_call() -> None:
    report = evaluate_response(
        question=QUESTION,
        answer="Employees receive 28 paid vacation days per year. [1]",
        contexts=[CONTEXT],
        judge=JsonJudge(),
    )
    assert report.mode == "llm"
    assert report.faithfulness == 0.91
    assert "passage" in report.reasoning
    assert report.citation_support == 1.0


def test_invalid_judge_output_falls_back_to_lexical_metrics() -> None:
    class BrokenJudge(JsonJudge):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="not json"))])

    report = evaluate_response(
        question=QUESTION,
        answer="Employees receive 28 paid vacation days per year. [1]",
        contexts=[CONTEXT],
        judge=BrokenJudge(),
    )
    assert report.mode == "lexical"
    assert "not valid JSON" in report.reasoning
