
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from rag import generate_answer, explain_simply, extract_key_findings


class ResearchState(TypedDict):
    question: str
    operation: str
    answer: str
    sources: list


def query_analyzer(state):
    question = state["question"].lower()

    if "explain" in question or "simply" in question:
        operation = "explain"
    elif "key findings" in question or "findings" in question:
        operation = "findings"
    else:
        operation = "qa"

    return {**state, "operation": operation}


def qa_node(state):
    answer, results = generate_answer(state["question"])
    return {**state, "answer": answer, "sources": results}


def explain_node(state):
    answer, results = explain_simply(state["question"])
    return {**state, "answer": answer, "sources": results}


def findings_node(state):
    answer, results = extract_key_findings()
    return {**state, "answer": answer, "sources": results}


def route_operation(state):
    return state["operation"]


graph_builder = StateGraph(ResearchState)

graph_builder.add_node("query_analyzer", query_analyzer)
graph_builder.add_node("qa", qa_node)
graph_builder.add_node("explain", explain_node)
graph_builder.add_node("findings", findings_node)

graph_builder.add_edge(START, "query_analyzer")

graph_builder.add_conditional_edges(
    "query_analyzer",
    route_operation,
    {
        "qa": "qa",
        "explain": "explain",
        "findings": "findings"
    }
)

graph_builder.add_edge("qa", END)
graph_builder.add_edge("explain", END)
graph_builder.add_edge("findings", END)

research_graph = graph_builder.compile()
