"""
agent.py — Code Review Agent (Standalone Module)
=================================================

This file is the standalone importable module for the Code Review Agent.
It contains everything needed to run the agent independently.

Usage:
    from agent import ask, retrieve, analyze_complexity

    result = ask("What is PEP8?", thread_id="session_001")
    print(result["answer"])
    print(result["route"])
    print(result["faithfulness"])

Submit this file along with:
    - day13_capstone.ipynb
    - capstone_streamlit.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import ast
import re
from typing import TypedDict, List
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2
SLIDING_WINDOW         = 6   # keep last 6 messages (3 turns)

OUT_OF_SCOPE_RESPONSE = (
    "I'm a specialized Python Code Review Agent. I can help with:\n"
    "- Reviewing Python code for PEP8, security, complexity issues\n"
    "- Questions about Python best practices and design principles\n\n"
    "Please paste Python code or ask a coding question!"
)

# ── Knowledge Base Documents (11 docs, one topic each) ────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "PEP8 Indentation and Line Length",
        "text": """PEP 8 is Python's official style guide. Indentation must use 4 spaces per level — never mix tabs and spaces. Maximum line length is 79 characters for code and 72 for docstrings and comments. Long lines should be broken using Python's implied line continuation inside brackets, parentheses, or braces. Continuation lines should align with the opening delimiter or use a hanging indent of 4 spaces. Avoid backslash line continuation when brackets can be used instead. A linter like flake8 or pycodestyle will flag E501 (line too long) and W191 (indentation contains tabs). Always configure your editor to show a ruler at column 79 and to insert spaces instead of tabs automatically."""
    },
    {
        "id": "doc_002",
        "topic": "PEP8 Naming Conventions",
        "text": """Python naming conventions from PEP 8: Variables and function names use snake_case (all lowercase, words separated by underscores): user_name, calculate_total. Class names use PascalCase (CapWords): UserAccount, DataProcessor. Constants use UPPER_SNAKE_CASE: MAX_RETRIES, DEFAULT_TIMEOUT. Private attributes and methods use a single leading underscore: _private_method. Name-mangled attributes use double leading underscore: __very_private. Module names should be short and lowercase: utils, models. Boolean variables should use is_, has_, can_, should_ prefixes: is_valid, has_permission, can_retry. Never use single-letter variable names except for loop counters (i, j, k). Avoid vague names like data, temp, info, obj, thing, value."""
    },
    {
        "id": "doc_003",
        "topic": "PEP8 Imports and Whitespace",
        "text": """Imports in Python must follow this order: (1) Standard library imports, (2) Related third-party imports, (3) Local application imports. Each group is separated by a blank line. Put each import on its own line. Avoid wildcard imports (from module import *) as they pollute the namespace. Whitespace rules: no space before a colon in slices or dicts, no trailing whitespace at end of lines, surround top-level function and class definitions with two blank lines, surround method definitions with one blank line. Use spaces around binary operators (=, +, -, *, /), but no spaces inside brackets. Use is not instead of not ... is. Use isinstance() instead of comparing types directly with type()."""
    },
    {
        "id": "doc_004",
        "topic": "SOLID Principles — SRP and OCP",
        "text": """SOLID is an acronym for five object-oriented design principles. Single Responsibility Principle (SRP): A class should have only one reason to change. If a class handles user authentication AND sends emails AND logs to a database, it violates SRP. Fix: split into UserAuthenticator, EmailSender, AuditLogger. Open/Closed Principle (OCP): Classes should be open for extension but closed for modification. Instead of adding if/elif chains to existing code to handle new types, use inheritance or interfaces so new behavior is added by creating new classes without modifying existing ones. Example violation: adding elif type == 'pdf' to an existing document processor. Fix: create a PdfProcessor subclass."""
    },
    {
        "id": "doc_005",
        "topic": "SOLID Principles — LSP, ISP, and DIP",
        "text": """Liskov Substitution Principle (LSP): Objects of a subclass must be substitutable for their superclass without breaking the program. A subclass must not restrict or change behavior expected from the parent class contract. Interface Segregation Principle (ISP): Clients should not be forced to depend on interfaces they do not use. Split large abstract base classes into smaller, specific ones so classes only implement what they actually need. Dependency Inversion Principle (DIP): High-level modules should not depend on low-level modules. Both should depend on abstractions. Use dependency injection — pass the database object into the class constructor rather than instantiating it inside the class. This makes code testable because you can inject a mock database during testing."""
    },
    {
        "id": "doc_006",
        "topic": "Security — SQL Injection and Input Validation",
        "text": """SQL injection is the most critical security vulnerability in Python web applications. It occurs when user input is concatenated directly into SQL queries. Never do this: query = 'SELECT * FROM users WHERE name = ' + name. Always use parameterized queries: cursor.execute('SELECT * FROM users WHERE name = ?', (name,)). With SQLAlchemy ORM, use filter_by() or filter() methods which are safe by default. Input validation: validate all user inputs for data type, length, format, and allowed characters. Use an allowlist approach — define exactly what IS allowed and reject everything else. Never trust inputs from forms, query parameters, headers, or cookies without validation. Sanitize file paths using os.path.basename() to prevent directory traversal attacks."""
    },
    {
        "id": "doc_007",
        "topic": "Error Handling Best Practices",
        "text": """Always catch the most specific exception possible. Bare except: catches everything including SystemExit and KeyboardInterrupt — almost always wrong. Bad: except: pass. Good: except ValueError as e: logger.error(e). Use exception chaining to preserve the original traceback: raise DatabaseError('Query failed') from original_exception. Define custom exception classes for your domain — inherit from Exception, not BaseException. Use try/finally or context managers (with statements) to ensure cleanup always runs even when exceptions occur. Log exceptions with full traceback using logger.exception('Error message'). Never swallow exceptions silently. Tools in the agent must return error strings rather than raising exceptions so the agent graph does not crash."""
    },
    {
        "id": "doc_008",
        "topic": "Cyclomatic Complexity and Code Metrics",
        "text": """Cyclomatic complexity measures the number of independent paths through code. Start at 1 and add 1 for each: if, elif, for, while, except, with, assert, and, or, ternary expression. Complexity 1-5 is simple and easy to test, 6-10 is acceptable, 11-20 is risky and hard to test, 21 and above must be refactored urgently. Functions should ideally be under 20-30 lines. Classes should not exceed 200-300 lines. Avoid nesting deeper than 3-4 levels — flatten using early returns (guard clauses) or by extracting helper functions. Functions with more than 4 parameters are a warning sign — group related parameters into a dataclass or dictionary. Use pylint, radon, or flake8-cognitive-complexity to measure these metrics automatically."""
    },
    {
        "id": "doc_009",
        "topic": "Python Best Practices — Context Managers and Type Hints",
        "text": """Context managers: always use the with statement for file operations, database connections, and locks. Bad: f = open('file.txt'); f.close(). Good: with open('file.txt', encoding='utf-8') as f:. Mutable default arguments are a classic Python trap — never use def func(lst=[]). The list is created once and shared across all calls. Fix: def func(lst=None): if lst is None: lst = []. Type hints: add type hints to all function signatures for readability and IDE support. Use Optional[T] for nullable parameters, Union[T1,T2] for multiple types, List[T] and Dict[K,V] for collections. Use @dataclass for data containers. Use f-strings for string formatting in Python 3.6+. Use enumerate() instead of range(len())."""
    },
    {
        "id": "doc_010",
        "topic": "Code Smells and Refactoring Techniques",
        "text": """Code smells are patterns that indicate deeper design problems. Long Method: functions over 30 lines likely do too much — extract into well-named helpers. God Class: a class that knows and does too much — split by responsibility. Primitive Obsession: using raw dicts and strings for domain concepts — create dataclasses. Feature Envy: a method that uses another class's data more than its own — move the method there. Dead Code: unused variables or imports — remove them. Magic Numbers: replace unexplained literals with named constants — STATUS_ACTIVE = 3 instead of if status == 3. Duplicate Code (DRY violation): extract repeated logic into a reusable function. Refactoring techniques: Extract Method, Rename, Introduce Guard Clause (return early to reduce nesting), Replace Magic Number with Constant."""
    },
    {
        "id": "doc_011",
        "topic": "Testing Best Practices and Docstrings",
        "text": """Unit testing follows the AAA pattern: Arrange (set up data), Act (call the function), Assert (verify result). Each test should test exactly one thing. Tests must be independent — never rely on execution order. Test names must be descriptive: test_login_with_invalid_password_returns_401. Use pytest for its powerful fixtures and readable output. Mock external dependencies (database, APIs) using unittest.mock so tests run fast without side effects. Target 80% code coverage on critical business logic. Docstrings: every public function, class, and module needs a docstring in Google style format with a summary line, Args section listing parameters with types and descriptions, Returns section, and Raises section for exceptions. Outdated docstrings that no longer match the implementation are worse than no docstring at all."""
    },
]


# ── Part 1: Build embedder and ChromaDB ───────────────────────────────────────
print("Loading SentenceTransformer embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

_client = chromadb.Client()   # in-memory (no disk persistence)
try:
    _client.delete_collection("capstone_kb")
except Exception:
    pass

collection = _client.create_collection("capstone_kb")

_texts      = [d["text"]             for d in DOCUMENTS]
_ids        = [d["id"]               for d in DOCUMENTS]
_metadatas  = [{"topic": d["topic"]} for d in DOCUMENTS]
_embeddings = embedder.encode(_texts).tolist()

collection.add(
    documents=_texts,
    embeddings=_embeddings,
    ids=_ids,
    metadatas=_metadatas
)

print(f"Knowledge base ready: {collection.count()} documents")
for d in DOCUMENTS:
    print(f"  • {d['topic']}")


# ── Retrieval helper ───────────────────────────────────────────────────────────
def retrieve(query: str, n_results: int = 3) -> dict:
    """Query ChromaDB and return formatted context string + source topics.

    Args:
        query:     The search query string.
        n_results: Number of chunks to retrieve (default 3).

    Returns:
        Dict with keys:
            context (str)       — formatted text with [Topic] labels
            sources (List[str]) — list of topic names retrieved
    """
    q_emb   = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    chunks = results["documents"][0]
    topics = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(
        f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
    )
    return {"context": context, "sources": topics}


# ── Part 2: State TypedDict ────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    """State TypedDict for the Code Review Agent.

    Designed FIRST — every field a node reads or writes must appear here.
    """
    # Input
    question:      str          # user's current question or code snippet

    # Memory
    messages:      List[dict]   # conversation history (sliding window)

    # Routing
    route:         str          # "retrieve" / "memory_only" / "tool"

    # RAG
    retrieved:     str          # ChromaDB context chunks (formatted)
    sources:       List[str]    # source topic names

    # Tool
    tool_result:   str          # AST complexity analysis output

    # Answer
    answer:        str          # final LLM response

    # Quality control
    faithfulness:  float        # eval score 0.0-1.0
    eval_retries:  int          # safety valve counter (max 2)

    # Domain-specific
    user_name:     str          # extracted if user says "my name is X"
    code_detected: bool         # True if Python code block found in input


# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)
print("LLM ready: llama-3.3-70b-versatile via Groq")


# ── Part 3: AST Tool ──────────────────────────────────────────────────────────
def analyze_complexity(code: str) -> str:
    """AST-based Python code complexity analysis tool.

    Analyzes cyclomatic complexity, function length, docstring presence,
    naming convention violations, and PEP8 line length.

    Never raises exceptions — always returns a string result.
    This is required: a crashing tool crashes the entire graph run.

    Args:
        code: Python source code string to analyze.

    Returns:
        Formatted analysis report string.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SYNTAX ERROR — cannot analyze: {e}"
    except Exception as e:
        return f"Parse error: {e}"

    lines  = code.splitlines()
    report = [
        "CODE ANALYSIS REPORT",
        "=" * 38,
        f"Total lines  : {len(lines)}"
    ]

    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    cls = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    report.append(f"Functions    : {len(fns)}")
    report.append(f"Classes      : {len(cls)}")

    # PEP8 line length check (max 79)
    long_lines = [
        f"Line {i+1} ({len(l)} chars)"
        for i, l in enumerate(lines) if len(l) > 79
    ]
    if long_lines:
        report.append("\nPEP8 line length violations (>79 chars):")
        for ll in long_lines[:5]:
            report.append(f"  - {ll}")
    else:
        report.append("All lines within 79-char PEP8 limit")

    # Per-function analysis
    for fn in fns:
        report.append(f"\nFunction: {fn.name}()")

        # Line count
        end = getattr(fn, "end_lineno", fn.lineno)
        report.append(f"  Lines: {end - fn.lineno + 1}")

        # Cyclomatic complexity
        cc = 1
        for node in ast.walk(fn):
            if isinstance(node, (
                ast.If, ast.For, ast.While, ast.ExceptHandler,
                ast.With, ast.Assert, ast.comprehension
            )):
                cc += 1
            elif isinstance(node, ast.BoolOp):
                cc += len(node.values) - 1

        if cc <= 5:
            flag = "Simple (good)"
        elif cc <= 10:
            flag = "Moderate (acceptable)"
        else:
            flag = "HIGH — refactor recommended"

        report.append(f"  Cyclomatic Complexity: {cc} — {flag}")

        # Docstring check
        has_doc = (
            fn.body
            and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
        )
        report.append(
            "  Docstring: present"
            if has_doc else
            "  Docstring: MISSING — add a Google-style docstring"
        )

        # Naming convention check (must be snake_case)
        if not re.match(r"^[a-z][a-z0-9_]*$", fn.name):
            report.append(
                f"  Naming: '{fn.name}' violates snake_case convention"
            )

        # Parameter count
        param_count = len(fn.args.args)
        if param_count > 4:
            report.append(
                f"  Parameters: {param_count} — consider grouping into a dataclass"
            )

    return "\n".join(report)


# ── Part 3: Node Functions ─────────────────────────────────────────────────────

def memory_node(state: CapstoneState) -> dict:
    """Add user question to conversation history with sliding window.

    Also extracts user name if present and detects Python code blocks.

    Args:
        state: Current graph state.

    Returns:
        Dict with updated messages, user_name, code_detected.
    """
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > SLIDING_WINDOW:
        msgs = msgs[-SLIDING_WINDOW:]

    # Extract user name if mentioned
    user_name = state.get("user_name", "")
    m = re.search(r"my name is ([A-Za-z]+)", state["question"], re.IGNORECASE)
    if m:
        user_name = m.group(1).capitalize()

    # Detect Python code in input
    code_detected = (
        "```python" in state["question"]
        or "```"    in state["question"]
        or ("def "  in state["question"] and ":" in state["question"])
        or ("class " in state["question"] and ":" in state["question"])
    )

    return {
        "messages":      msgs,
        "user_name":     user_name,
        "code_detected": code_detected
    }


def router_node(state: CapstoneState) -> dict:
    """Classify user input and decide the retrieval route.

    Routes:
        retrieve    — PEP8/SOLID/security/best-practice questions
        memory_only — simple follow-ups needing no KB retrieval
        tool        — Python code submitted for review/analysis

    Args:
        state: Current graph state.

    Returns:
        Dict with route field set.
    """
    question = state["question"]
    messages = state.get("messages", [])
    recent   = "; ".join(
        f"{m['role']}: {m['content'][:60]}"
        for m in messages[-3:-1]
    ) or "none"

    prompt = f"""You are a router for a Python Code Review Agent.

Available options:
- retrieve: user asked about PEP8, SOLID principles, security, naming \
conventions, error handling, testing, or Python best practices
- memory_only: simple follow-up like 'what did you just say?', 'tell me more', 'thanks'
- tool: user submitted Python code (contains def, class, or code blocks \
with ```) for complexity analysis and review

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()

    if "memory"   in decision: decision = "memory_only"
    elif "tool"   in decision: decision = "tool"
    else:                      decision = "retrieve"

    # Override: if code block is detected, always route to tool
    if state.get("code_detected"):
        decision = "tool"

    return {"route": decision}


def retrieval_node(state: CapstoneState) -> dict:
    """Retrieve relevant context from ChromaDB knowledge base.

    Args:
        state: Current graph state.

    Returns:
        Dict with retrieved context string and sources list.
    """
    result = retrieve(state["question"], n_results=3)
    return {"retrieved": result["context"], "sources": result["sources"]}


def skip_retrieval_node(state: CapstoneState) -> dict:
    """Skip retrieval for memory-only queries.

    Args:
        state: Current graph state.

    Returns:
        Dict with empty retrieved and sources fields.
    """
    return {"retrieved": "", "sources": []}


def tool_node(state: CapstoneState) -> dict:
    """Run AST complexity analysis tool on submitted Python code.

    Extracts code from ```python blocks or raw def/class statements.
    Also retrieves KB context so the answer node has best-practice info.

    Args:
        state: Current graph state.

    Returns:
        Dict with tool_result, retrieved context, and sources.
    """
    question = state["question"]

    # Extract code block
    code_match = re.search(r"```python\n(.*?)```", question, re.DOTALL)
    if not code_match:
        code_match = re.search(r"```\n?(.*?)```", question, re.DOTALL)

    if code_match:
        code = code_match.group(1).strip()
    elif "def " in question or "class " in question:
        code = question
    else:
        code = None

    if code:
        tool_result = analyze_complexity(code)
        # Also retrieve KB context to guide the LLM review
        result    = retrieve("PEP8 naming complexity best practices", n_results=3)
        retrieved = result["context"]
        sources   = result["sources"]
    else:
        tool_result = (
            "No Python code block found. "
            "Please wrap your code in ```python ... ``` blocks."
        )
        retrieved = ""
        sources   = []

    return {
        "tool_result": tool_result,
        "retrieved":   retrieved,
        "sources":     sources
    }


def answer_node(state: CapstoneState) -> dict:
    """Generate final answer using LLM with RAG context and tool output.

    Builds a route-specific system prompt. Adds escalation instruction
    when eval_retries > 0 (answer quality was low on previous attempt).

    Args:
        state: Current graph state.

    Returns:
        Dict with answer field set.
    """
    question     = state["question"]
    retrieved    = state.get("retrieved", "")
    tool_result  = state.get("tool_result", "")
    messages     = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)
    route        = state.get("route", "retrieve")

    # Build context section
    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"CODE ANALYSIS TOOL OUTPUT:\n{tool_result}")
    context = "\n\n".join(context_parts)

    # System prompt varies by route
    if route == "tool":
        system_content = """You are an expert Python Code Review Agent.
Structure your review as:
1. Overall Assessment
2. Issues Found (cite from tool output and knowledge base)
3. Strengths
4. Specific Recommendations with code examples
Base your review ONLY on the Knowledge Base and Tool Output provided.
Do NOT add information not present in the context."""
    else:
        system_content = """You are an expert Python Code Review Agent and coding mentor.
Answer using ONLY the information provided in the context below.
If the answer is not in the context, say: I don't have that information in my knowledge base.
Do NOT add information from your training data."""

    if context:
        system_content += f"\n\n{context}"
    else:
        system_content = (
            "You are a helpful Python Code Review Agent. "
            "Answer based on the conversation history."
        )

    # Escalation instruction after a failed eval
    if eval_retries > 0:
        system_content += (
            "\n\nIMPORTANT: Your previous answer did not meet quality standards. "
            "Be MORE specific and grounded in the context above. "
            "Provide concrete code examples."
        )

    # Build LangChain message list
    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        if msg["role"] == "user":
            lc_msgs.append(HumanMessage(content=msg["content"]))
        else:
            lc_msgs.append(AIMessage(content=msg["content"]))
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}


def eval_node(state: CapstoneState) -> dict:
    """Evaluate answer faithfulness on a 0.0-1.0 scale.

    Skips evaluation when there is no retrieved context
    (tool-only or memory-only answers).

    Args:
        state: Current graph state.

    Returns:
        Dict with faithfulness score and incremented eval_retries.
    """
    answer  = state.get("answer", "")
    context = state.get("retrieved", "")[:500]
    retries = state.get("eval_retries", 0)

    if not context:
        # No KB retrieval — skip faithfulness check
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:300]}"""

    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.5

    gate = "PASS" if score >= FAITHFULNESS_THRESHOLD else "RETRY"
    print(f"  [eval] Faithfulness: {score:.2f} — {gate}")
    return {"faithfulness": score, "eval_retries": retries + 1}


def save_node(state: CapstoneState) -> dict:
    """Append final assistant answer to conversation history.

    Args:
        state: Current graph state.

    Returns:
        Dict with updated messages list.
    """
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    if len(messages) > SLIDING_WINDOW:
        messages = messages[-SLIDING_WINDOW:]
    return {"messages": messages}


# ── Part 4: Conditional edge functions ────────────────────────────────────────

def route_decision(state: CapstoneState) -> str:
    """After router_node: decide which retrieval path to take.

    Returns:
        'retrieve' | 'skip' | 'tool'
    """
    route = state.get("route", "retrieve")
    if route == "tool":        return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    """After eval_node: retry answer or save and finish.

    Returns:
        'answer' (retry) | 'save' (proceed to end)
    """
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"


# ── Part 4: Build and compile the graph ───────────────────────────────────────

def _build_graph() -> object:
    """Build and compile the LangGraph for the Code Review Agent.

    Returns:
        Compiled LangGraph application with MemorySaver checkpointer.
    """
    graph = StateGraph(CapstoneState)

    # Add all 8 nodes
    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    # Entry point
    graph.set_entry_point("memory")

    # Fixed edges
    graph.add_edge("memory",   "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_edge("save",     END)

    # Conditional edges
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )
    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {"answer": "answer", "save": "save"}
    )

    # Compile with MemorySaver for persistent conversation memory
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    print("Graph compiled successfully!")
    return app


# Build the graph when module is imported
app = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def ask(question: str, thread_id: str = "default") -> dict:
    """Submit a question or code to the Code Review Agent.

    This is the main public function. The same thread_id preserves
    conversation memory across multiple calls.

    Args:
        question:  User's question or Python code snippet to review.
                   Wrap code in ```python ... ``` blocks for best results.
        thread_id: Session identifier. Same ID = same conversation memory.
                   Use a different ID to start a fresh conversation.

    Returns:
        Dict with keys:
            answer      (str)   — agent's response
            route       (str)   — routing decision: retrieve/tool/memory_only
            faithfulness(float) — quality score 0.0-1.0 from eval_node
            sources     (list)  — KB topic names used in the answer

    Example:
        result = ask("What is PEP8?", thread_id="student_001")
        print(result["answer"])

        result = ask("Review this:\\n```python\\ndef foo(x,y,z,a,b): pass\\n```")
        print(result["answer"])
        print(result["faithfulness"])
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config=config)
    return {
        "answer":       result.get("answer", ""),
        "route":        result.get("route", ""),
        "faithfulness": result.get("faithfulness", 0.0),
        "sources":      result.get("sources", [])
    }


# ── Quick self-test when run directly ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("agent.py self-test")
    print("=" * 50)

    tests = [
        ("What is PEP8?",              "default"),
        ("How do I prevent SQL injection?", "default"),
        ("My name is Arjun. What are naming conventions?", "default"),
        ("Can you remind me what we discussed?",           "default"),  # memory test
    ]

    for q, tid in tests:
        print(f"\nQ: {q}")
        r = ask(q, thread_id=tid)
        print(f"Route       : {r['route']}")
        print(f"Faithfulness: {r['faithfulness']:.2f}")
        print(f"Sources     : {r['sources']}")
        print(f"Answer      : {r['answer'][:150]}...")
        print("-" * 40)

    print("\nagent.py self-test complete.")
