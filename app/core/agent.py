from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.core.agent_tools import make_tools
from app.core.qa_chain import get_llm
from app.core.guardrails import check_input, sanitise_input, format_guardrail_response
import time
from typing import List


# Why LangGraph over old AgentExecutor?
# LangChain 1.x removed AgentExecutor completely.
# LangGraph is the official agent framework.
# create_react_agent builds a prebuilt ReAct graph:
# LLM node decides tool → Tool node executes →
# Result flows back → LLM decides if done
#
# System prompt injection: In LangGraph 1.1.9
# neither 'prompt' nor 'state_modifier' parameters
# are accepted by create_react_agent.
# The correct approach is to prepend a SystemMessage
# to the messages list when invoking — this is the
# standard pattern for injecting system context.
#
# Interview answer for Strides Q18 "What is LangGraph?":
# "LangGraph is LangChain's framework for building stateful
# multi-actor applications as graphs. We used it because
# LangChain 1.x removed AgentExecutor in favour of LangGraph.
# Our agent uses create_react_agent — a prebuilt ReAct graph
# where the LLM node decides which tool to call, the tool
# node executes it, and results flow through the graph state."

AGENT_SYSTEM_PROMPT = """You are a legal contract assistant.

You have three tools that retrieve raw contract text:
1. retrieve_and_answer — for specific clause questions
2. summarise_contract — for overview/summary requests
3. flag_contract_risks — for risk and red flag questions

WORKFLOW:
1. Choose the right tool based on the question
2. The tool returns raw contract text with page numbers
3. Use that text to generate a clear, cited answer
4. Always include [Page X] citations in your answer
5. Only use information from the tool output — never your own knowledge

If the tool returns no relevant content, say:
"This information is not found in the provided contract." """


def create_legal_agent(session_id: str):
    """
    Creates a LangGraph ReAct agent for legal contract analysis.

    No system prompt parameter passed here — LangGraph 1.1.9
    does not accept 'prompt' or 'state_modifier'.
    System prompt is injected as SystemMessage in invoke().
    """
    llm = get_llm()
    tools = make_tools(session_id)

    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    return agent


def get_agent_answer(
    question: str,
    session_id: str,
    chat_history: List = None
) -> dict:
    """
    Main entry point — takes a question, returns an answer.

    System prompt is prepended as SystemMessage in the
    messages list before invoking the agent graph.
    This is the correct pattern for LangGraph 1.1.9.

    How the graph executes:
    1. SystemMessage sets context and tool routing rules
    2. HumanMessage contains the user question
    3. LLM decides which tool to call
    4. Tool retrieves raw contract text
    5. LLM generates cited answer from tool output
    6. We extract the last AIMessage as final answer
    """

    if chat_history is None:
        chat_history = []

    start_time = time.time()

    # Input guardrail
    is_valid, reason = check_input(question)
    if not is_valid:
        return format_guardrail_response(False, reason)

    question = sanitise_input(question)

    try:
        agent = create_legal_agent(session_id)

        # Prepend SystemMessage to inject system prompt
        # This is the correct LangGraph 1.1.9 pattern
        messages = (
            [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
            + chat_history
            + [HumanMessage(content=question)]
        )

        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 10}
        )

        # Extract final answer — last AIMessage with string content
        final_answer = ""
        for message in reversed(result["messages"]):
            if (
                isinstance(message, AIMessage)
                and message.content
                and isinstance(message.content, str)
                and len(message.content.strip()) > 0
            ):
                final_answer = message.content
                break

        if not final_answer:
            final_answer = (
                "The agent could not generate a response. "
                "Please try again or switch to Direct RAG mode."
            )

        return {
            "answer": final_answer,
            "sources": [],
            "latency_seconds": round(time.time() - start_time, 2),
            "status": "success",
            "warnings": [],
            "agent_used": True
        }

    except Exception as e:
        return {
            "answer": f"Agent error: {str(e)}",
            "sources": [],
            "latency_seconds": round(time.time() - start_time, 2),
            "status": "error",
            "warnings": [],
            "agent_used": True
        }