from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List

class State(TypedDict):
    messages: List

llm = ChatOpenAI(model="gpt-4o-mini")

def decide_agent(state: State):
    user_input = state["messages"][-1][1].lower()
    if "order" in user_input:
        return "order_agent"
    elif "refund" in user_input:
        return "refund_agent"
    else:
        return "product_agent"

def order_agent(state: State):
    return {"messages": [("assistant", "Looks like your order is still being processed. It should ship soon.")]}

def refund_agent(state: State):
    return {"messages": [("assistant", "You can request a refund within 30 days of purchase as long as the item is unused.")]}

def product_agent(state: State):
    return {"messages": [("assistant", "I’d recommend a MacBook Air or a Dell XPS depending on your budget.")]}

builder = StateGraph(State)
builder.add_node("router", decide_agent)
builder.add_node("order_agent", order_agent)
builder.add_node("refund_agent", refund_agent)
builder.add_node("product_agent", product_agent)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    decide_agent,
    {
        "order_agent": "order_agent",
        "refund_agent": "refund_agent",
        "product_agent": "product_agent",
    },
)

builder.set_finish_point("order_agent")
builder.set_finish_point("refund_agent")
builder.set_finish_point("product_agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = graph.invoke(
        {"messages": [("user", user_input)]},
        config=config
    )

    print("Bot:", response["messages"][-1][1])
