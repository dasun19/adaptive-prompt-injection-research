"""Hierarchical LangGraph workflow for the DevOps pipeline."""

from .agents import code_reviewer_agent, test_analyzer_agent, release_manager_agent, manager_agent
from .sequential import LangGraphWorkflow, TaskOutputTemplate, WorkflowState, _make_node
from .tasks import code_review_task, test_analysis_task, release_decision_task, manager_coordination_task
from langgraph.graph import END, START, StateGraph


def build_hierarchical_crew() -> LangGraphWorkflow:
    """
    Hierarchical Architecture
    ─────────────────────────
                    Manager Node
                  /      |        \
        Code Reviewer  Test Analyzer  Release Manager

    A manager node produces coordination guidance before the specialist
    nodes execute in order.
    """

    manager = manager_agent()
    reviewer = code_reviewer_agent()
    tester = test_analyzer_agent()
    releaser = release_manager_agent()

    manager_step = manager_coordination_task(manager)
    review_step = code_review_task(reviewer)
    test_step = test_analysis_task(tester)
    release_step = release_decision_task(releaser)

    graph = StateGraph(WorkflowState)
    graph.add_node("manager", _make_node(manager_step))
    graph.add_node("review", _make_node(review_step))
    graph.add_node("test", _make_node(test_step))
    graph.add_node("release", _make_node(release_step))
    graph.add_edge(START, "manager")
    graph.add_edge("manager", "review")
    graph.add_edge("review", "test")
    graph.add_edge("test", "release")
    graph.add_edge("release", END)
    compiled = graph.compile()

    return LangGraphWorkflow(
        architecture="hierarchical",
        graph=compiled,
        output_templates=[
            TaskOutputTemplate(agent=manager.role, description=manager_step.description, output_key="manager_output"),
            TaskOutputTemplate(agent=reviewer.role, description=review_step.description, output_key="review_output"),
            TaskOutputTemplate(agent=tester.role, description=test_step.description, output_key="test_analysis_output"),
            TaskOutputTemplate(agent=releaser.role, description=release_step.description, output_key="release_output"),
        ],
    )


if __name__ == "__main__":
    workflow = build_hierarchical_crew()
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
