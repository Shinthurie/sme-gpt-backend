from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

chat_history = []
llm = OllamaLLM(model="llama3")

VALID_ACTIONS = [
    "summary",
    "payable_lookup",
    "document_search",
    "corrected_only",
    "histogram",
    "bar_chart",
    "insights"
]


def keyword_route(question: str):
    q = question.lower()

    if any(term in q for term in ["payable", "amount due", "how much payable", "ගෙවිය යුතු"]):
        return "payable_lookup"

    if any(term in q for term in ["corrected", "wrong totals", "fixed totals"]):
        return "corrected_only"

    if any(term in q for term in ["find document", "search", "show invoice", "document"]):
        return "document_search"

    if any(term in q for term in ["histogram", "distribution"]):
        return "histogram"

    if any(term in q for term in ["bar chart", "count chart"]):
        return "bar_chart"

    if any(term in q for term in ["insight", "analysis", "summary of data"]):
        return "insights"

    return None


def normalize_action(response: str):
    response = response.strip().lower()

    for action in VALID_ACTIONS:
        if action in response:
            return action

    return "summary"


def decide_action(summary, question):
    routed = keyword_route(question)
    if routed:
        return routed

    prompt = PromptTemplate(
        input_variables=["summary", "question"],
        template="""
You are a financial data analyst.

Choose ONLY ONE action from:
- summary
- payable_lookup
- document_search
- corrected_only
- histogram
- bar_chart
- insights

Rules:
- Return ONLY the action name
- No explanation
- If unsure, return summary

Dataset:
{summary}

Question:
{question}

Answer:
"""
    )

    chain = prompt | llm
    response = chain.invoke({"summary": summary, "question": question})
    return normalize_action(response)


def ask_ai(data, question):
    global chat_history

    history_text = "\n".join(chat_history[-6:])

    prompt = PromptTemplate(
        input_variables=["data", "question", "history"],
        template="""
You are a professional financial analyst.

Rules:
- Use ONLY the provided result
- Do NOT invent numbers
- Do NOT change the currency
- If currency is not explicitly shown, do not assume one
- If the result is aggregated across multiple records, explicitly say it is an aggregated total
- Explain clearly and simply
- If exact value is missing, say so

Conversation History:
{history}

Available Result:
{data}

Question:
{question}

Answer:
"""
    )

    chain = prompt | llm

    response = chain.invoke({
        "data": data,
        "question": question,
        "history": history_text
    }).strip()

    chat_history.append(f"User: {question}")
    chat_history.append(f"AI: {response}")

    return response