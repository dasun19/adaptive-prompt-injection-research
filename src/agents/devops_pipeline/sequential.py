"""Sequential LangGraph workflow for the DevOps pipeline."""

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict, Optional
import time

from langgraph.graph import END, START, StateGraph

from .agents import code_reviewer_agent, test_analyzer_agent, release_manager_agent
from .tasks import code_review_task, test_analysis_task, release_decision_task


class WorkflowState(TypedDict, total=False):
    project: str
    pr_content: str
    test_logs: str
    review_output: str
    test_analysis_output: str
    release_output: str


@dataclass(frozen=True)
class TaskOutputTemplate:
    agent: str
    description: str
    output_key: str


@dataclass
class TaskOutput:
    agent: str
    description: str
    raw: str
    execution_time_sec: Optional[float] = None


class LangGraphWorkflow:
    def __init__(self, architecture: str, graph: Any, output_templates: List[TaskOutputTemplate]):
        self.architecture = architecture
        self._graph = graph
        self._output_templates = output_templates
        self.tasks_output: List[TaskOutput] = []

    def kickoff(self, inputs: Dict[str, Any]):
        state = self._graph.invoke(inputs)
        self.tasks_output = [
            TaskOutput(
                agent=template.agent,
                description=template.description,
                raw=state.get(template.output_key, ""),
                execution_time_sec=state.get(f"{template.output_key}_time"),
            )
            for template in self._output_templates
        ]
        return self


def _make_node(step):
    def node(state: WorkflowState):
        start = time.time()
        out = step.chain.invoke(state)
        exec_time = round(time.time() - start, 3)

        return {
            step.output_key: out,
            f"{step.output_key}_time": exec_time,
        }

    return node


def build_sequential_crew() -> LangGraphWorkflow:
    """
    Sequential Architecture
    ──────────────────────
    Code Reviewer → Test Analyzer → Release Manager
    """

    reviewer = code_reviewer_agent()
    tester = test_analyzer_agent()
    releaser = release_manager_agent()

    review_step = code_review_task(reviewer)
    test_step = test_analysis_task(tester)
    release_step = release_decision_task(releaser)

    graph = StateGraph(WorkflowState)
    graph.add_node("review", _make_node(review_step))
    graph.add_node("test", _make_node(test_step))
    graph.add_node("release", _make_node(release_step))
    graph.add_edge(START, "review")
    graph.add_edge("review", "test")
    graph.add_edge("test", "release")
    graph.add_edge("release", END)
    compiled = graph.compile()

    return LangGraphWorkflow(
        architecture="sequential",
        graph=compiled,
        output_templates=[
            TaskOutputTemplate(agent=reviewer.role, description=review_step.description, output_key="review_output"),
            TaskOutputTemplate(agent=tester.role, description=test_step.description, output_key="test_analysis_output"),
            TaskOutputTemplate(agent=releaser.role, description=release_step.description, output_key="release_output"),
        ],
    )


if __name__ == "__main__":
    workflow = build_sequential_crew()
    result = workflow.kickoff(inputs={
        "project": "payments-service PR #482",
        "pr_content": (
            "Adds retry logic to the payment webhook handler. "
            "Diff: webhook.py +42 -6 lines."
        ),
        "test_logs": (
            "42 passed, 0 failed, 0 skipped. Coverage: 87% (+2%)."
        ),
    })
    for output in result.tasks_output:
        print(f"\n[{output.agent}]\n{output.raw}")
