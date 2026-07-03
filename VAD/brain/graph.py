from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END

from brain.state import BrainState
from brain.nodes.should_rag import make_should_rag_node
from brain.nodes.retrieve import make_retrieve_node
from brain.nodes.generate import make_generate_node
from brain.nodes.decide import make_decide_node
from brain.nodes.memory import make_memory_node

if TYPE_CHECKING:
    from brain.llm_service import LLMService
    from brain.memory_service import MemoryService
    from brain.rag_service import RAGService

logger = logging.getLogger(__name__)


def _route_after_decide(state: BrainState) -> str:
    return "memory_node" if state.get("should_save_facts") else END


def build_fast_graph(
    llm: "LLMService",
    memory: "MemoryService",
    rag: "RAGService",
    recent_limit: int = 10,
):
    """Compile the fast response graph.

    Flow:
        START → should_rag_node
                    ├─(use_rag=True)  → retrieve_node → generate_node → END
                    └─(use_rag=False) → generate_node → END

    decide/memory nodes run in the background after HTTP response is returned.
    """
    should_rag = make_should_rag_node()
    retrieve = make_retrieve_node(rag)
    generate = make_generate_node(llm, memory, recent_limit)

    builder: StateGraph = StateGraph(BrainState)
    builder.add_node("should_rag_node", should_rag)
    builder.add_node("retrieve_node", retrieve)
    builder.add_node("generate_node", generate)

    builder.add_edge(START, "should_rag_node")
    builder.add_conditional_edges(
        "should_rag_node",
        lambda state: "retrieve_node" if state.get("use_rag") else "generate_node",
        {"retrieve_node": "retrieve_node", "generate_node": "generate_node"},
    )
    builder.add_edge("retrieve_node", "generate_node")
    builder.add_edge("generate_node", END)

    graph = builder.compile()
    logger.info("BrainMaster fast graph compiled successfully")
    return graph


def build_graph(
    llm: "LLMService",
    memory: "MemoryService",
    rag: "RAGService",
    recent_limit: int = 10,
):
    """Compile the full BrainMaster LangGraph workflow (kept for reference).

    Flow: START → retrieve_node → generate_node → decide_node
                                                        ├─(YES)→ memory_node → END
                                                        └─(NO) → END
    """
    retrieve = make_retrieve_node(rag)
    generate = make_generate_node(llm, memory, recent_limit)
    decide = make_decide_node(memory, llm)
    remember = make_memory_node(memory, llm)

    builder: StateGraph = StateGraph(BrainState)
    builder.add_node("retrieve_node", retrieve)
    builder.add_node("generate_node", generate)
    builder.add_node("decide_node", decide)
    builder.add_node("memory_node", remember)

    builder.add_edge(START, "retrieve_node")
    builder.add_edge("retrieve_node", "generate_node")
    builder.add_edge("generate_node", "decide_node")
    builder.add_conditional_edges(
        "decide_node",
        _route_after_decide,
        {"memory_node": "memory_node", END: END},
    )
    builder.add_edge("memory_node", END)

    graph = builder.compile()
    logger.info("BrainMaster graph compiled successfully")
    return graph
