# Agentic AI Customer Service Chatbot

## Project overview
This project is a simple agentic AI chatbot built with OpenAI and LangGraph. I designed it to act like a customer service assistant for a small online electronics store. The chatbot can help with order status questions, refund policy questions, and product suggestions.

## What this project includes
- A clear system prompt for each chatbot role
- Three different agents:
  - Order Status Agent
  - Refund Policy Agent
  - Product Suggestion Agent
- A router that decides which agent should answer
- LangGraph `MemorySaver` so the chatbot remembers earlier messages in the same conversation
- OpenAI API integration using `ChatOpenAI`

## Why it is agentic
Instead of using one single response flow, the chatbot first checks the user request, then routes the request to the most appropriate agent. Each agent has its own role and instructions.

## Files
- `app.py` - main chatbot code
- `requirements.txt` - packages needed to run the project
- `README.md` - project explanation and setup

## How to run it
1. Install the packages:

```bash
pip install -r requirements.txt
```

2. Add your OpenAI API key:

### Mac / Linux
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Windows PowerShell
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

3. Run the chatbot:

```bash
python app.py
```

## Example conversation
This is a simple example of a 3 turn conversation showing memory and routing.

```text
You: Hi, I need help with my order.
Bot: Sure, I can help with that. What is your order number?

You: It is A1002.
Bot: Thanks. Your order A1002 has already shipped and is expected to arrive tomorrow.

You: Also what is your refund policy if I change my mind?
Bot: We accept returns within 30 days of delivery as long as the item is unused and in its original condition. Refunds usually take 5 to 7 business days after approval.
```

## What I learned
This project helped me understand how agentic AI works in a more practical way. The main idea was not just getting one answer from a model, but breaking the chatbot into smaller roles and using memory so the interaction feels more natural.

## Notes for demo
A good live demo is:
1. Ask about an order without giving the order number
2. Give the order number in the next turn
3. Ask a refund or product follow-up question

That shows routing, specialized agents, and memory in one short demo.
