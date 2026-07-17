import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from typing import TypedDict, Literal

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Ask for the key at runtime instead of hardcoding it.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is missing.")


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
print("✅ LLM client ready.")

"""## 3. Agent identity

This is the "constitution" of EduGrow AI, adapted from the project's System Instructions document: a persona that always analyzes the *nexus* between education (SDG 4) and economic growth (SDG 8).
"""

AGENT_NAME = "EduGrow AI"

CORE_PERSONA = """You are EduGrow AI, a specialist advisory agent built for the "Quality Education and
Economic Growth" project, operating at the intersection of SDG 4 (Quality Education) and SDG 8
(Decent Work and Economic Growth).

IDENTITY (use this whenever someone asks who/what you are, your name, your purpose, or what you
can do): Your name is EduGrow AI. You were built to analyze, explain, and advise on how education
and skill-building translate into measurable, inclusive economic progress. You can: (1) analyze
the link between a specific educational or training intervention and its economic outcomes,
(2) recommend or reference skill-development programs grounded in sample labour-market data,
(3) bridge tangentially related topics (technology, climate, healthcare, migration) back to
education-and-economic-growth, and (4) read an uploaded report/curriculum for extra context.
Answer identity questions directly, briefly, and warmly - never redirect or refuse them.

MISSION: synthesize how educational interventions and skill-building programs lead to tangible,
measurable economic progress, for an audience of students, policymakers, and researchers.

TONE: professional, precise, evidence-based, encouraging, and objective - the voice of a senior
analyst, not a casual chatbot. Prefer clear, well-structured prose over hype.

STYLE RULE: use accurate professional terminology, but whenever a concept is highly technical, add
a short "In layman's terms" line so the answer stays accessible to a general audience.

GUARDRAILS: never suggest growth strategies that rely on exploitation, child labour, or environmental
degradation; never take a personal side on political, religious, or ideological topics - default to
neutral, socio-economic analysis; always keep answers grounded and avoid overpromising specific
statistics you cannot support.
"""

"""## 4. Shared state

Every node reads from and writes to this dictionary as the query moves through the graph - the same pattern used in the triage-agent project.
"""

class AgentState(TypedDict):
    user_query: str        # the raw question from the user
    file_context: str      # optional text pulled from an uploaded document
    category: Literal["meta", "on_topic", "gray_area", "off_topic"]
    reasoning: str          # why the classifier chose that category
    response: str           # the final answer shown to the user

"""## 5. A tiny grounding dataset

A handful of real-ish skill-development programs the agent can cite so its "actionable insight" isn't just a generic guess. In a fuller version this would be a CSV upload (like `doctors.csv` in the triage project) - here it's inline for simplicity.
"""

programs_df = pd.DataFrame([
    {"program": "Digital Skills Bootcamp", "sector": "technology", "target_group": "youth (18-24)", "employment_lift_pct": 27},
    {"program": "Vocational Welding Certification", "sector": "manufacturing", "target_group": "adult re-skillers", "employment_lift_pct": 19},
    {"program": "Green Energy Technician Training", "sector": "renewable energy", "target_group": "youth (18-30)", "employment_lift_pct": 31},
    {"program": "Rural Financial Literacy Program", "sector": "agriculture & finance", "target_group": "smallholder farmers", "employment_lift_pct": 14},
    {"program": "Healthcare Support Worker Certificate", "sector": "healthcare", "target_group": "women re-entering the workforce", "employment_lift_pct": 22},
    {"program": "Data Analytics Micro-credential", "sector": "technology", "target_group": "college graduates", "employment_lift_pct": 24},
    {"program": "Tourism & Hospitality Skills Program", "sector": "tourism & hospitality", "target_group": "rural youth", "employment_lift_pct": 18},
    {"program": "Logistics & Supply Chain Certificate", "sector": "logistics", "target_group": "unemployed adults", "employment_lift_pct": 21},
    {"program": "Early Childhood Educator Training", "sector": "education", "target_group": "women (25-40)", "employment_lift_pct": 16},
    {"program": "Entrepreneurship & Microfinance Bootcamp", "sector": "small business & finance", "target_group": "informal-sector workers", "employment_lift_pct": 29},
    {"program": "Construction Trades Apprenticeship", "sector": "construction", "target_group": "youth (18-28)", "employment_lift_pct": 23},
    {"program": "AI & Automation Reskilling Track", "sector": "technology", "target_group": "displaced factory workers", "employment_lift_pct": 26},
])

"""## 6. Classifier node

Mirrors the "Condition-Based Handling" section of the System Instructions: every query is sorted into `on_topic`, `gray_area`, or `off_topic`. A few obvious keywords are caught instantly (cheap and fast); everything else is judged by the LLM. If the model's reply is ambiguous, we default to `gray_area` rather than silently refusing - EduGrow AI tries to find the bridge first.
"""

CLASSIFIER_PROMPT = """You are a precise routing classifier for EduGrow AI, an assistant focused
ONLY on the intersection of quality education and economic growth (skills, labor markets,
employability, curricula, training programs, economic mobility, GDP/employment links).

Classify the user's message into EXACTLY ONE category:
- meta: a question about the assistant itself - its name, purpose, identity, capabilities, or how
  it works (e.g. "what is your name", "what can you do", "who made you").
- on_topic: clearly about education, skills, training, labor markets, or their economic impact.
- gray_area: tangentially related (e.g. climate change, technology, healthcare, migration) where a
  genuine link to education-for-economic-growth could reasonably be drawn.
- off_topic: unrelated (politics, religion, entertainment, personal advice, general tech support, etc.)
  with no meaningful connection to education or economic growth.

Respond with ONLY one word: meta, on_topic, gray_area, or off_topic."""

# Fast keyword shortcuts so we don't always need an LLM round trip for obvious cases.
IDENTITY_HINTS = ["your name", "who are you", "what are you", "what can you do", "what is edugrow",
                   "what do you do", "your purpose", "about yourself", "who made you", "who built you",
                   "how do you work", "your capabilities", "introduce yourself"]

OBVIOUS_OFF_TOPIC = ["weather today", "football score", "cricket score", "movie recommendation",
                      "celebrity", "song lyrics", "who will win the election"]


def classifier_node(state: AgentState) -> AgentState:
    query_lower = state["user_query"].lower()

    if any(kw in query_lower for kw in IDENTITY_HINTS):
        category = "meta"
        reasoning = "Matched an identity/purpose keyword - routed without an LLM call."
    elif any(kw in query_lower for kw in OBVIOUS_OFF_TOPIC):
        category = "off_topic"
        reasoning = "Matched an obvious off-topic keyword - routed without an LLM call."
    else:
        result = llm.invoke([
            SystemMessage(content=CLASSIFIER_PROMPT),
            HumanMessage(content=state["user_query"]),
        ])
        raw = result.content.strip().lower()

        if "meta" in raw:
            category = "meta"
        elif "on_topic" in raw or "on-topic" in raw:
            category = "on_topic"
        elif "off_topic" in raw or "off-topic" in raw:
            category = "off_topic"
        elif "gray" in raw or "grey" in raw:
            category = "gray_area"
        else:
            category = "gray_area"  # safe default: attempt to help rather than refuse

        reasoning = f"LLM classified the query as: '{raw}'"

    return {**state, "category": category, "reasoning": reasoning}

"""## 7. Response nodes

One node per scenario from the System Instructions:

- **`on_topic_node`** - analyzes the nexus of SDG 4 & SDG 8, grounds the answer with the sample dataset when a sector matches, and always closes with an actionable takeaway.
- **`gray_area_node`** - actively tries to bridge the topic back to education/economic growth (e.g. "green jobs need green education").
- **`off_topic_node`** - the "Pivot Protocol": a polite, firm redirect. No LLM call needed, so it's fast and always on-message.
"""

def _matching_programs(query_lower: str) -> str:
    hits = programs_df[programs_df["sector"].apply(lambda s: any(word in query_lower for word in s.split()))]
    if hits.empty:
        return "No closely matching program in the sample dataset."
    lines = [
        f"- {r.program} ({r.sector}): ~{r.employment_lift_pct}% employment lift for {r.target_group}"
        for r in hits.itertuples()
    ]
    return "\n".join(lines)


def meta_node(state: AgentState) -> AgentState:
    response = (
        "Hi, I'm **EduGrow AI** 🌱 - an advisory agent built for the *Quality Education and Economic "
        "Growth* project. My focus is the intersection of SDG 4 (Quality Education) and SDG 8 (Decent "
        "Work and Economic Growth).\n\n"
        "Here's what I can do:\n"
        "- Analyze how a specific educational or training intervention affects economic outcomes.\n"
        "- Reference sample skill-development programs and their estimated employment impact.\n"
        "- Bridge related topics (technology, climate, healthcare, migration) back to education and "
        "economic growth.\n"
        "- Read an uploaded report, curriculum, or policy brief for extra context.\n\n"
        "Ask me anything about skills, training, education policy, or labor markets to get started!"
    )
    return {**state, "response": response}


def on_topic_node(state: AgentState) -> AgentState:
    query_lower = state["user_query"].lower()
    grounding = _matching_programs(query_lower)

    prompt = f"""{CORE_PERSONA}

Behavioral rule for this ON-TOPIC query: always frame the answer through BOTH SDG 4 (the
educational/skill-building angle) and SDG 8 (the resulting employability / economic impact).
Explain the *why*, not just the facts, and end with one concrete, data-flavoured actionable takeaway
(e.g. "This shift in training could reduce the local unemployment gap by X%").

Relevant sample data you may cite if useful:
{grounding}

Optional extra context from an uploaded document:
{state.get("file_context") or "None provided."}
"""

    result = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["user_query"])])
    return {**state, "response": result.content.strip()}


def gray_area_node(state: AgentState) -> AgentState:
    prompt = f"""{CORE_PERSONA}

Behavioral rule for this GRAY-AREA query: the topic is only tangentially related to education and
economic growth. Actively look for a genuine bridge back to the core mission (for example, "green
jobs" (SDG 8) require "green education" (SDG 4)). If, after real effort, there truly is no honest
connection, say so briefly and then use the same polite Pivot Protocol as an off-topic query:
acknowledge the point, explain your focus is SDG 4 x SDG 8, and offer to relate it back to human
capital development or return to the main discussion.
"""
    result = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["user_query"])])
    return {**state, "response": result.content.strip()}


def off_topic_node(state: AgentState) -> AgentState:
    response = (
        "That's an interesting point; however, my expertise is focused specifically on the synergy "
        "between quality education (SDG 4) and economic growth (SDG 8). I can discuss how that topic "
        "might relate to human capital development, or we can return to our primary discussion. "
        "Which would you prefer?"
    )
    return {**state, "response": response}

"""## 8. Build the graph"""

def route_decision(state: AgentState) -> Literal["meta", "on_topic", "gray_area", "off_topic"]:
    return state["category"]


builder = StateGraph(AgentState)

builder.add_node("classifier", classifier_node)
builder.add_node("meta", meta_node)
builder.add_node("on_topic", on_topic_node)
builder.add_node("gray_area", gray_area_node)
builder.add_node("off_topic", off_topic_node)

builder.set_entry_point("classifier")

builder.add_conditional_edges(
    "classifier",
    route_decision,
    {
        "meta": "meta",
        "on_topic": "on_topic",
        "gray_area": "gray_area",
        "off_topic": "off_topic",
    },
)

builder.add_edge("meta", END)
builder.add_edge("on_topic", END)
builder.add_edge("gray_area", END)
builder.add_edge("off_topic", END)

graph = builder.compile()
print("✅ Graph compiled.")


"""## 10. Quick text test

Runs three example queries - one for each category - straight through the graph, so you can sanity-check the routing before opening the UI.
"""

"""## 11. Final UI - Gradio

The finished deliverable: a chat-style interface where a user can ask a question and optionally upload a PDF/DOCX/TXT (e.g. a labour-market report or a curriculum outline) for extra context, exactly like the file-upload pattern from the first prototype.
"""

import gradio as gr
import fitz
from docx import Document
import pandas as pd
import plotly.express as px


def extract_text(file_obj):
    if file_obj is None:
        return ""
    path = file_obj.name if hasattr(file_obj, "name") else file_obj
    try:
        if path.lower().endswith(".pdf"):
            doc = fitz.open(path)
            return "\n".join(page.get_text() for page in doc)
        elif path.lower().endswith(".docx"):
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        elif path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        return f"⚠️ Unable to read file.\n\n{e}"
    return ""


CATEGORY_BADGE = {
    "meta": ("👋 About EduGrow", "#3b82f6"),
    "on_topic": ("📚 SDG 4 • 💼 SDG 8", "#22c55e"),
    "gray_area": ("🌍 Connected Topic", "#facc15"),
    "off_topic": ("🚫 Outside Scope", "#ef4444"),
}


def chat_edugrow(user_input, file, history):
    history = history or []

    if not user_input or not user_input.strip():
        return (
            history,
            "",
            gr.update(
                value='<div style="color:#ef4444;font-weight:600;">Please enter a question.</div>',
                visible=True,
            ),
        )

    file_context = extract_text(file) if file else ""

    state = {
        "user_query": user_input,
        "file_context": file_context,
        "category": "on_topic",
        "reasoning": "",
        "response": "",
    }

    try:
        result = graph.invoke(state)
        label, color = CATEGORY_BADGE[result["category"]]
        badge_html = f"""
        <div style="display:inline-block;background:{color}20;color:{color};padding:6px 14px;
        border-radius:999px;border:1px solid {color};font-size:13px;font-weight:600;margin-bottom:10px;">
            {label}
        </div>
        """
        answer = result["response"]
    except Exception as e:
        badge_html = ""
        answer = f"⚠️ Agent Error\n\n{e}"

    history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer},
    ]

    return history, "", gr.update(value=badge_html, visible=True)


def export_chat(history):
    """Turns the chat history into a downloadable .txt transcript."""
    if not history:
        return gr.update(value=None, visible=False)

    lines = []
    for turn in history:
        role = "You" if turn["role"] == "user" else "EduGrow AI"
        lines.append(f"{role}: {turn['content']}\n")

    path = "/tmp/edugrow_chat_transcript.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return gr.update(value=path, visible=True)


EXAMPLES = [
    "What is EduGrow AI?",
    "Explain SDG 4.",
    "Explain SDG 8.",
    "How does vocational training improve employment?",
    "How can AI improve education?",
    "Recommend programs for rural youth.",
    "What does India's education market look like in 2025?",
    "How many people are still uneducated globally, and how can economic growth help eliminate this?",
    "What is the global literacy rate and how does it impact GDP growth?",
    "How does the gender gap in education affect economic growth in developing countries?",
]

sector_stats = (
    programs_df.groupby("sector")["employment_lift_pct"]
    .mean()
    .round(1)
    .reset_index()
    .rename(columns={"sector": "Sector", "employment_lift_pct": "Avg Employment Lift (%)"})
    .sort_values("Avg Employment Lift (%)", ascending=False)
)

dashboard_fig = px.bar(
    sector_stats,
    x="Sector",
    y="Avg Employment Lift (%)",
    text="Avg Employment Lift (%)",
    color="Avg Employment Lift (%)",
    color_continuous_scale="Greens",
    height=300,
)
dashboard_fig.update_layout(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font_color="white",
    margin=dict(l=20, r=20, t=40, b=20),
    title="📊 Avg. Employment Lift by Sector (from EduGrow program data)",
    coloraxis_showscale=False,
    xaxis_tickangle=-30,
)
dashboard_fig.update_traces(textposition="outside")


CUSTOM_CSS = """
footer{display:none !important;}
.gradio-container{
    background:#0f172a;
    background-image:
        radial-gradient(circle at top left,#22c55e22 0%,transparent 30%),
        radial-gradient(circle at bottom right,#3b82f622 0%,transparent 35%);
    font-family:Inter,Segoe UI,sans-serif;
    color:white;
}
#hero{
    text-align:center; padding:28px; border-radius:22px;
    background:linear-gradient(135deg,#16a34a,#2563eb);
    margin-bottom:18px; box-shadow:0 10px 35px rgba(0,0,0,.35);
}
#hero h1{font-size:40px; margin:0; color:white;}
#hero p{margin-top:10px; font-size:16px; color:#ecfeff;}
.card{
    background:#111827; border-radius:18px; padding:18px;
    border:1px solid #1f2937; box-shadow:0 8px 20px rgba(0,0,0,.30);
}
.metric{font-size:30px; font-weight:700; color:#22c55e;}
.metric-title{font-size:14px; color:#94a3b8;}
#sidebar{background:#111827; border-radius:18px; padding:16px; border:1px solid #1f2937;}
#chatbox{border-radius:18px; border:1px solid #334155;}
#send-btn{
    background:#22c55e !important; color:white !important; font-weight:700;
    border:none !important; border-radius:12px !important;
}
#send-btn:hover{background:#16a34a !important;}
.example-btn{background:#1e293b !important; border:1px solid #334155 !important; color:white !important;}
.example-btn:hover{background:#334155 !important;}
"""

with gr.Blocks(title="EduGrow AI",css=CUSTOM_CSS) as ui:

    # ================= HERO =================
    gr.HTML("""
    <div id="hero">
        <h1>🌱 EduGrow AI</h1>
        <p>AI Assistant for <b>SDG 4 - Quality Education</b> &amp;
        <b>SDG 8 - Decent Work &amp; Economic Growth</b></p>
    </div>
    """)

    # ================= METRIC CARDS =================
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div class="card">
                <div class="metric">📚 SDG 4</div>
                <div class="metric-title">Quality Education</div>
            </div>
            """)
        with gr.Column():
            gr.HTML("""
            <div class="card">
                <div class="metric">💼 SDG 8</div>
                <div class="metric-title">Decent Work</div>
            </div>
            """)
        with gr.Column():
            gr.HTML("""
            <div class="card">
                <div class="metric">🤖 AI Powered</div>
                <div class="metric-title">LangGraph + Groq</div>
            </div>
            """)

    # ================= DASHBOARD =================
    with gr.Row():
        dashboard = gr.Plot(value=dashboard_fig, label="", show_label=False)

    # ================= SIDEBAR + CHAT =================
    with gr.Row():

        with gr.Column(scale=1, elem_id="sidebar"):
            gr.Markdown("## 🚀 Quick Actions")

            example_buttons = []
            for example in EXAMPLES:
                btn = gr.Button(example, elem_classes="example-btn", size="sm")
                example_buttons.append(btn)

            gr.Markdown("---")
            gr.Markdown("## 📂 Upload Context")
            file_input = gr.File(
                label="Upload PDF / DOCX / TXT",
                file_types=[".pdf", ".docx", ".txt"],
            )

            gr.Markdown("---")
            gr.Markdown("## 📈 Dashboard Summary")
            gr.Markdown(
                "- 🎓 Education Score : **92**\n"
                "- 💼 Employment Score : **81**\n"
                "- 💻 Digital Skills : **95**\n"
                "- 🚀 Innovation : **74**"
            )

            gr.Markdown("---")
            badge = gr.HTML(value="", visible=False)

            gr.Markdown(
                "### 💡 EduGrow AI\n"
                "This assistant specializes in\n\n"
                "- SDG 4 (Quality Education)\n"
                "- SDG 8 (Economic Growth)\n"
                "- Skill Development\n"
                "- Labour Market Insights\n"
                "- AI-powered Recommendations"
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=560,
                elem_id="chatbox",
                label="",
                show_label=False,
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="💬 Ask anything about SDG 4, SDG 8, education, skills, employment...",
                    show_label=False,
                    scale=8,
                    lines=1,
                )
                send = gr.Button("🚀 Send", elem_id="send-btn", scale=1)

            with gr.Row():
                clear = gr.Button("🗑️ Clear Chat", variant="secondary")
                export = gr.Button("📄 Export", variant="secondary")

            export_file = gr.File(label="Download transcript", visible=False)

            gr.Markdown("---")
            gr.Markdown(
                "### 🌱 EduGrow AI\n\n"
                "**Mission**\n\n"
                "EduGrow AI bridges **Quality Education (SDG 4)** and **Decent Work & Economic Growth (SDG 8)** "
                "by helping users understand how education, skills, vocational training and workforce "
                "development contribute to sustainable economic progress.\n\n"
                "**Features**\n\n"
                "- 📚 Education Analysis\n"
                "- 💼 Employment Insights\n"
                "- 🤖 AI-powered Recommendations\n"
                "- 📄 Document Question Answering\n"
                "- 🌍 SDG-focused Responses"
            )

    # ================= EVENT HANDLERS =================
    send.click(
        fn=chat_edugrow,
        inputs=[msg, file_input, chatbot],
        outputs=[chatbot, msg, badge],
    )

    msg.submit(
        fn=chat_edugrow,
        inputs=[msg, file_input, chatbot],
        outputs=[chatbot, msg, badge],
    )

    clear.click(
        lambda: ([], "", gr.update(value="", visible=False), gr.update(value=None, visible=False)),
        outputs=[chatbot, msg, badge, export_file],
    )

    export.click(
        fn=export_chat,
        inputs=[chatbot],
        outputs=[export_file],
    )

    for example, button in zip(EXAMPLES, example_buttons):

        button.click(
            lambda text=example: text,
            outputs=msg,
        )

def launch_app():
    base_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7861")))
    last_error = None

    for port in [base_port] + list(range(base_port + 1, base_port + 21)):
        try:
            ui.queue()
            ui.launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=True)
            return
        except OSError as exc:
            last_error = exc
            if "address" in str(exc).lower() or "port" in str(exc).lower():
                print(f"Port {port} is unavailable, trying next port...")
                continue
            raise

    raise RuntimeError(f"Could not start the Gradio app. Last error: {last_error}") from last_error


if __name__ == "__main__":
    launch_app()
