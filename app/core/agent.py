# LangGraph 1.x is the current standard - AgentExecutor removed in LangChain 1.x
# create_react_agent from langgraph.prebuilt is the correct replacement
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.core.agent_tools import make_tools
from app.core.qa_chain import get_llm
from app.core.guardrails import check_input, sanitise_input, format_guardrail_response
import time
from typing import List


# Why LangGraph over old AgentExecutor?
#
# LangChain 1.x removed AgentExecutor completely.
# LangGraph is now the official way to build agents.
# LangGraph models agents as state machines (graphs):
# - Nodes = agent or tool execution steps
# - Edges = routing decisions between steps
# - State = the full conversation history passed between nodes
#
# create_react_agent is a prebuilt graph that handles
# the common ReAct pattern automatically:
# User input → LLM decides tool → Tool executes →
# LLM sees result → Decides if done or needs another tool
#
# Interview answer for Strides Q18 "What is LangGraph?":
# "LangGraph is LangChain's framework for building stateful
# multi-actor applications as graphs. We used it in production
# because LangChain 1.x removed AgentExecutor in favour of
# LangGraph. Our agent uses create_react_agent which builds
# a prebuilt ReAct graph — the LLM node decides which tool
# to call, the tool node executes it, and the result flows
# back to the LLM node. This graph structure makes the agent
# debuggable, observable, and easy to extend with new nodes."

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

    Why create_react_agent?
    It is the simplest correct API in LangGraph 1.x.
    It builds a complete agent graph with:
    - LLM node: decides which tool to call
    - Tool node: executes the chosen tool
    - Conditional edges: loop back if more tools needed
    - END edge: when LLM decides answer is complete

    state_modifier = our system prompt injected into
    the graph's initial state. This is the LangGraph 1.x
    equivalent of the system message in the old prompt template.
    """

    llm = get_llm()
    tools = make_tools(session_id)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=AGENT_SYSTEM_PROMPT
    )

    return agent


def get_agent_answer(
    question: str,
    session_id: str,
    chat_history: List = None
) -> dict:
    """
    Main entry point — takes a question, returns an answer.

    How LangGraph agent execution works:
    1. We call agent.invoke() with a messages list
    2. LangGraph runs the graph:
       - LLM node sees the question + system prompt
       - LLM returns a tool call (structured JSON)
       - Tool node executes the tool
       - Result added to messages state
       - LLM node sees tool result
       - LLM decides if answer is complete
       - If yes → END, if no → call another tool
    3. Final messages list returned
    4. We extract the last AIMessage as the answer

    Why extract the last AIMessage?
    LangGraph returns the full message history including
    tool calls and tool results. The final answer is always
    the last AIMessage in the list.
    """

    if chat_history is None:
        chat_history = []

    start_time = time.time()

    # Input guardrail — before touching the agent
    is_valid, reason = check_input(question)
    if not is_valid:
        return format_guardrail_response(False, reason)

    question = sanitise_input(question)

    try:
        agent = create_legal_agent(session_id)

        # Build messages list for LangGraph
        # LangGraph uses messages state — full conversation history
        messages = chat_history + [HumanMessage(content=question)]

        # Invoke the graph
        result = agent.invoke(
    {"messages": messages},
    config={"recursion_limit": 10})

        # Extract final answer from result messages
        # LangGraph returns all messages including tool calls
        # The last AIMessage with content is the final answer
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
            final_answer = "The agent could not generate a response. Please try again."

        latency = round(time.time() - start_time, 2)

        return {
            "answer": final_answer,
            "sources": [],
            "latency_seconds": latency,
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