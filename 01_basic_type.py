from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import operator
import os
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# ---- Models ----

class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(..., description="what the task is about")


class PLAN(BaseModel):
    blog_title: str
    tasks: List[Task]


# ---- State ----

class State(TypedDict):
    topic: str
    plan: PLAN
    sections: Annotated[List[str], operator.add]
    final: str


# ---- LLM ----

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7)


# ---- Nodes ----

def orchestrator(state: State) -> dict:
    plan = llm.with_structured_output(PLAN).invoke([
        SystemMessage(content="Create a blog plan with 1-2 sections on the given topic."),
        HumanMessage(content=f"Topic: {state['topic']}")
    ])
    return {"plan": plan}


def fanout(state: State):
    return [
        Send("worker", {
            "task": task,
            "blog_title": state["plan"].blog_title,
            "topic": state["topic"]
        })
        for task in state["plan"].tasks
    ]


def worker(payload: dict) -> dict:
    task       = payload["task"]
    topic      = payload["topic"]
    blog_title = payload["blog_title"]
    # blog_title = plan.blog_title

    response = llm.invoke([
        SystemMessage(content="Write one clean Markdown section."),
        HumanMessage(content=(
            f"Blog: {blog_title}\n"
            f"Topic: {topic}\n\n"
            f"Section: {task.title}\n"
            f"Brief: {task.brief}\n\n"
            "Return only the section content in Markdown."
        )),
    ])

    # ✅ Handle both string and list content
    content = response.content
    if isinstance(content, list):
        section_md = content[0].get("text", "").strip()
    else:
        section_md = content.strip()

    return {"sections": [section_md]}


def reducer(state: State) -> dict:
    title = state["plan"].blog_title
    body  = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}\n"

    # ---- save to file ----
    filename    = title.lower().replace(" ", "_") + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    print(f"\n✅ Blog saved to: {filename}")

    return {"final": final_md}


# ---- Build Graph ----

g = StateGraph(State)

g.add_node("orchestrator", orchestrator)
g.add_node("worker",       worker)
g.add_node("reducer",      reducer)

g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker",  "reducer")
g.add_edge("reducer", END)

app = g.compile()


# ---- Run ----

if __name__ == "__main__":
    out = app.invoke({"topic": "Write a blog on Self Attention"})
    print("\n---- FINAL BLOG ----\n")
    print(out["final"])