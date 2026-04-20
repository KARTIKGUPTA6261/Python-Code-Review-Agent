import streamlit as st
import uuid
import os
import ast
import re
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

st.set_page_config(page_title="Code Review Agent", page_icon="🔍", layout="centered")
st.title("🔍 Python Code Review Agent")
st.caption("AI-powered reviewer for PEP8, security, complexity, and best practices.")


@st.cache_resource
def load_agent():
    groq_key = os.getenv("GROQ_API_KEY", "")
    llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_key)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    DOCUMENTS = [
        {"id": "doc_001", "topic": "PEP8 Indentation and Line Length",
         "text": "PEP 8 is Python official style guide. Indentation must use 4 spaces per level, never mix tabs and spaces. Maximum line length is 79 characters for code and 72 for docstrings. Long lines should be broken using implied line continuation inside brackets. A linter like flake8 will flag E501 line too long and W191 indentation contains tabs."},
        {"id": "doc_002", "topic": "PEP8 Naming Conventions",
         "text": "Variables and function names use snake_case: user_name, calculate_total. Class names use PascalCase: UserAccount, DataProcessor. Constants use UPPER_SNAKE_CASE: MAX_RETRIES. Private methods use single leading underscore: _private_method. Boolean variables use is_, has_, can_ prefixes: is_valid, has_permission. Avoid vague names like data, temp, info, obj."},
        {"id": "doc_003", "topic": "PEP8 Imports and Whitespace",
         "text": "Imports order: standard library first, then third-party, then local. Each group separated by blank line. Put each import on its own line. Avoid wildcard imports. Use isinstance instead of type. Use is not instead of not is."},
        {"id": "doc_004", "topic": "SOLID Principles SRP and OCP",
         "text": "Single Responsibility Principle: A class should have only one reason to change. Open Closed Principle: Classes should be open for extension but closed for modification. Add new behavior by creating new classes, not modifying existing ones."},
        {"id": "doc_005", "topic": "SOLID Principles LSP ISP DIP",
         "text": "Liskov Substitution: Subclass objects must be substitutable for superclass. Interface Segregation: Split large abstract classes into smaller specific ones. Dependency Inversion: Use dependency injection, pass dependencies through constructor."},
        {"id": "doc_006", "topic": "Security SQL Injection and Input Validation",
         "text": "SQL injection: Never concatenate user input into SQL queries. Always use parameterized queries with cursor.execute and tuple. Validate all inputs for type, length, format. Never expose stack traces to end users."},
        {"id": "doc_007", "topic": "Error Handling Best Practices",
         "text": "Always catch specific exceptions. Bare except catches SystemExit and KeyboardInterrupt which is wrong. Use exception chaining: raise DatabaseError from original. Define custom exceptions inheriting from Exception. Tools must return error strings, never raise exceptions."},
        {"id": "doc_008", "topic": "Cyclomatic Complexity and Code Metrics",
         "text": "Cyclomatic complexity: start at 1, add 1 for each if, for, while, except, and, or. Complexity 1 to 5 is simple, 6 to 10 is acceptable, 11 to 20 is risky, 21 and above must be refactored. Functions under 30 lines, avoid nesting deeper than 4 levels."},
        {"id": "doc_009", "topic": "Python Best Practices Context Managers and Type Hints",
         "text": "Always use with statement for files. Never use mutable default arguments like def func with lst equals list. Use type hints on all functions. Use dataclass for data containers. Use f-strings. Use enumerate instead of range len."},
        {"id": "doc_010", "topic": "Code Smells and Refactoring",
         "text": "Code smells: Long Method, God Class, Primitive Obsession, Feature Envy, Dead Code, Magic Numbers, Duplicate Code. Refactoring: Extract Method, Rename, Guard Clause for early returns, Replace Magic Number with Constant."},
        {"id": "doc_011", "topic": "Testing Best Practices and Docstrings",
         "text": "Testing AAA pattern: Arrange, Act, Assert. Each test tests one thing. Use pytest. Mock external dependencies. Target 80 percent coverage. Docstrings in Google style: summary, Args, Returns, Raises."},
    ]

    client = chromadb.Client()
    try:
        client.delete_collection("capstone_kb")
    except Exception:
        pass
    collection = client.create_collection("capstone_kb")
    texts = [d["text"] for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
    )

    class CapstoneState(TypedDict):
        question: str; messages: List[dict]; route: str; retrieved: str
        sources: List[str]; tool_result: str; answer: str
        faithfulness: float; eval_retries: int; user_name: str; code_detected: bool

    FAITHFULNESS_THRESHOLD = 0.7
    MAX_EVAL_RETRIES = 2
    SLIDING_WINDOW = 6

    def analyze_complexity(code):
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return "SYNTAX ERROR: " + str(e)
        lines = code.splitlines()
        report = ["CODE ANALYSIS REPORT", "=" * 35, "Total lines: " + str(len(lines))]
        fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        report.append("Functions: " + str(len(fns)))
        long_lines = ["Line " + str(i+1) for i, l in enumerate(lines) if len(l) > 79]
        if long_lines:
            report.append("Long lines >79 chars: " + str(long_lines[:3]))
        else:
            report.append("All lines within 79-char PEP8 limit")
        for fn in fns:
            cc = 1
            for node in ast.walk(fn):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                                     ast.With, ast.Assert, ast.comprehension)):
                    cc += 1
                elif isinstance(node, ast.BoolOp):
                    cc += len(node.values) - 1
            flag = "Simple" if cc <= 5 else "Moderate" if cc <= 10 else "HIGH-refactor"
            has_doc = (fn.body and isinstance(fn.body[0], ast.Expr)
                       and isinstance(fn.body[0].value, ast.Constant))
            report.append(fn.name + "(): CC=" + str(cc) + " " + flag
                          + ", docstring=" + ("YES" if has_doc else "MISSING")
                          + ", params=" + str(len(fn.args.args)))
        return "\n".join(report)

    def memory_node(state):
        msgs = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
        if len(msgs) > SLIDING_WINDOW:
            msgs = msgs[-SLIDING_WINDOW:]
        name = state.get("user_name", "")
        m = re.search(r"my name is ([A-Za-z]+)", state["question"], re.IGNORECASE)
        if m:
            name = m.group(1).capitalize()
        code_det = ("```" in state["question"]
                    or ("def " in state["question"] and ":" in state["question"]))
        return {"messages": msgs, "user_name": name, "code_detected": code_det}

    def router_node(state):
        prompt = ("Router for Python Code Review Agent. Routes: "
                  "retrieve (PEP8/SOLID/security questions), "
                  "memory_only (simple follow-up), "
                  "tool (Python code submitted for review). "
                  "Question: " + state["question"] + " Reply ONE word only:")
        resp = llm.invoke(prompt).content.strip().lower()
        if "memory" in resp:
            route = "memory_only"
        elif "tool" in resp:
            route = "tool"
        else:
            route = "retrieve"
        if state.get("code_detected"):
            route = "tool"
        return {"route": route}

    def retrieval_node(state):
        qe = embedder.encode([state["question"]]).tolist()
        res = collection.query(query_embeddings=qe, n_results=3,
                               include=["documents", "metadatas"])
        chunks = res["documents"][0]
        topics = [m["topic"] for m in res["metadatas"][0]]
        ctx = "\n\n---\n\n".join("[" + topics[i] + "]\n" + chunks[i]
                                    for i in range(len(chunks)))
        return {"retrieved": ctx, "sources": topics}

    def skip_retrieval_node(state):
        return {"retrieved": "", "sources": []}

    def tool_node(state):
        q = state["question"]
        m = re.search(r"```python\n(.*?)```", q, re.DOTALL)
        if not m:
            m = re.search(r"```\n?(.*?)```", q, re.DOTALL)
        code = m.group(1).strip() if m else (q if ("def " in q or "class " in q) else None)
        if code:
            tr = analyze_complexity(code)
            qe = embedder.encode(["PEP8 naming complexity best practices"]).tolist()
            res = collection.query(query_embeddings=qe, n_results=3,
                                   include=["documents", "metadatas"])
            chunks = res["documents"][0]
            topics = [mt["topic"] for mt in res["metadatas"][0]]
            ctx = "\n\n---\n\n".join("[" + topics[i] + "]\n" + chunks[i]
                                        for i in range(len(chunks)))
            return {"tool_result": tr, "retrieved": ctx, "sources": topics}
        return {"tool_result": "No code block found. Use ```python ... ``` blocks.",
                "retrieved": "", "sources": []}

    def answer_node(state):
        route = state.get("route", "retrieve")
        ctx_parts = []
        if state.get("retrieved"):
            ctx_parts.append("KNOWLEDGE BASE:\n" + state["retrieved"])
        if state.get("tool_result"):
            ctx_parts.append("CODE ANALYSIS:\n" + state["tool_result"])
        context = "\n\n".join(ctx_parts)
        if route == "tool":
            sp = ("Expert Python Code Review Agent. "
                  "Review: Assessment, Issues, Strengths, Recommendations. "
                  "Base only on context.")
        else:
            sp = ("Expert Python coding mentor. "
                  "Answer ONLY from context. "
                  "If not found say: I do not have that in my knowledge base.")
        if context:
            sp = sp + "\n\n" + context
        if state.get("eval_retries", 0) > 0:
            sp = sp + "\n\nIMPROVE: Be more specific and grounded in context."
        msgs = [SystemMessage(content=sp)]
        for msg in state.get("messages", [])[:-1]:
            if msg["role"] == "user":
                msgs.append(HumanMessage(content=msg["content"]))
            else:
                msgs.append(AIMessage(content=msg["content"]))
        msgs.append(HumanMessage(content=state["question"]))
        return {"answer": llm.invoke(msgs).content}

    def eval_node(state):
        retries = state.get("eval_retries", 0)
        context = state.get("retrieved", "")[:500]
        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}
        prompt = ("Rate faithfulness 0.0-1.0. Reply only a number.\n"
                  "Context: " + context + "\n"
                  "Answer: " + state.get("answer", "")[:300])
        try:
            score = float(llm.invoke(prompt).content.strip().split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.75
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state):
        msgs = state.get("messages", []) + [{"role": "assistant", "content": state["answer"]}]
        if len(msgs) > SLIDING_WINDOW:
            msgs = msgs[-SLIDING_WINDOW:]
        return {"messages": msgs}

    def route_decision(state):
        r = state.get("route", "retrieve")
        if r == "tool": return "tool"
        if r == "memory_only": return "skip"
        return "retrieve"

    def eval_decision(state):
        if (state.get("faithfulness", 1.0) >= FAITHFULNESS_THRESHOLD
                or state.get("eval_retries", 0) >= MAX_EVAL_RETRIES):
            return "save"
        return "answer"

    g = StateGraph(CapstoneState)
    for name, fn in [("memory", memory_node), ("router", router_node),
                     ("retrieve", retrieval_node), ("skip", skip_retrieval_node),
                     ("tool", tool_node), ("answer", answer_node),
                     ("eval", eval_node), ("save", save_node)]:
        g.add_node(name, fn)
    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_edge("retrieve", "answer")
    g.add_edge("skip", "answer")
    g.add_edge("tool", "answer")
    g.add_edge("answer", "eval")
    g.add_edge("save", END)
    g.add_conditional_edges("router", route_decision,
                            {"retrieve": "retrieve", "tool": "tool", "skip": "skip"})
    g.add_conditional_edges("eval", eval_decision,
                            {"answer": "answer", "save": "save"})
    agent_app = g.compile(checkpointer=MemorySaver())
    return agent_app, embedder, collection


try:
    agent_app, embedder, collection = load_agent()
    st.success("Knowledge base loaded — " + str(collection.count()) + " documents")
except Exception as e:
    st.error("Failed to load agent: " + str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

with st.sidebar:
    st.header("About")
    st.write("Python Code Review Agent")
    st.write("Session: " + st.session_state.thread_id)
    st.divider()
    st.write("**Topics covered:**")
    for t in ["PEP8 Indentation and Naming", "SOLID Principles",
               "Security and SQL Injection", "Error Handling",
               "Cyclomatic Complexity", "Code Smells and Refactoring",
               "Testing and Docstrings"]:
        st.write("• " + t)
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about Python best practices or paste code for review..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Reviewing..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, could not generate an answer.")
        st.write(answer)
        faith = result.get("faithfulness", 0.0)
        if faith > 0:
            st.caption("Faithfulness: " + str(round(faith, 2))
                       + "  |  Route: " + result.get("route", "")
                       + "  |  Sources: " + str(len(result.get("sources", []))))
    st.session_state.messages.append({"role": "assistant", "content": answer})
