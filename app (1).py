import os
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# A small fake store so the chatbot has something concrete to talk about.
ORDER_DB = {
    "A1001": {"status": "processing", "eta": "2 business days"},
    "A1002": {"status": "shipped", "eta": "arriving tomorrow"},
    "A1003": {"status": "delivered", "eta": "delivered yesterday"},
}

PRODUCTS = {
    "laptop": [
        "MacBook Air for portability and school work",
        "Dell XPS 13 for a Windows option with a premium feel",
    ],
    "headphones": [
        "Sony WH-1000XM5 for noise canceling",
        "AirPods Pro for Apple users and easy everyday use",
    ],
    "phone": [
        "iPhone 15 for a simple and reliable experience",
        "Samsung Galaxy S24 for customization and a strong camera",
    ],
}

ROUTER_PROMPT = """
You are a routing assistant for a customer service chatbot.
Read the conversation and choose exactly one label:
- order_status
- refund_policy
- product_suggestion

Return only the label and nothing else.
""".strip()

ORDER_PROMPT = """
You are an order status support agent for a small online electronics store.
Your job is to help with order tracking questions.
Use a warm, natural customer service tone.

Rules:
- If the customer gives an order number, use the store data below.
- If they do not give an order number, ask for it.
- Keep answers short and helpful.
- If the customer asks a follow-up question, use the earlier conversation.

Store order data:
{order_data}
""".strip()

REFUND_PROMPT = """
You are a refund policy support agent for a small online electronics store.
Use a friendly and clear tone.

Refund policy:
- Returns are allowed within 30 days of delivery.
- The item should be unused and in original condition.
- Opened items can still be reviewed case by case if they are defective.
- Refunds usually take 5 to 7 business days after the return is approved.

Rules:
- Answer only based on this policy.
- If the user asks a follow-up, use the previous messages for context.
- Keep it simple and conversational.
""".strip()

PRODUCT_PROMPT = """
You are a product recommendation agent for a small online electronics store.
Use a natural, helpful tone like a student demo project.

Available product categories and examples:
{product_data}

Rules:
- Ask one quick follow-up only if needed.
- Give 1 or 2 recommendations with a short reason.
- Use previous conversation if the user already mentioned a budget, category, or preference.
""".strip()


if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing. Add your API key before running the chatbot.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)


class RouteDecision(MessagesState):
    route: str


def router_node(state: MessagesState):
    # This node does not change state. It just exists so the graph has a routing step.
    return {}


def route_request(state: MessagesState) -> Literal["order_agent", "refund_agent", "product_agent"]:
    response = llm.invoke([SystemMessage(content=ROUTER_PROMPT)] + state["messages"])
    label = response.content.strip().lower()

    if "order" in label:
        return "order_agent"
    if "refund" in label:
        return "refund_agent"
    return "product_agent"


def order_agent(state: MessagesState):
    response = llm.invoke(
        [SystemMessage(content=ORDER_PROMPT.format(order_data=ORDER_DB))] + state["messages"]
    )
    return {"messages": [response]}


def refund_agent(state: MessagesState):
    response = llm.invoke([SystemMessage(content=REFUND_PROMPT)] + state["messages"])
    return {"messages": [response]}


def product_agent(state: MessagesState):
    response = llm.invoke(
        [SystemMessage(content=PRODUCT_PROMPT.format(product_data=PRODUCTS))] + state["messages"]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("router", router_node)
builder.add_node("order_agent", order_agent)
builder.add_node("refund_agent", refund_agent)
builder.add_node("product_agent", product_agent)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_request,
    {
        "order_agent": "order_agent",
        "refund_agent": "refund_agent",
        "product_agent": "product_agent",
    },
)
builder.add_edge("order_agent", END)
builder.add_edge("refund_agent", END)
builder.add_edge("product_agent", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


def main():
    print("Customer Service Chatbot")
    print("Ask about order status, refund policy, or product suggestions.")
    print("Type 'exit' to stop.\n")

    config = {"configurable": {"thread_id": "customer-demo"}}

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Bot: Thanks for stopping by.")
            break

        result = graph.invoke({"messages": [("user", user_input)]}, config=config)
        print("Bot:", result["messages"][-1].content)
        print()


if __name__ == "__main__":
    main()
