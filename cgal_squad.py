"""
CGAL Research Squad — a multi-agent assistant for working on the
Consensus-Gated Associative Learning framework and its Monty implementation.

This is a sketch. Treat it as a starting point: the agent descriptions need
iteration based on actual routing behavior, the system prompts need to be
filled in with real context (the CGAL draft, the Monty repo structure),
and you'll likely want to add a few custom tools.

Setup:
    pip install "agent-squad[anthropic]"

Usage (after filling in your API key):
    python cgal_squad.py

Architecture:
    - 5 specialist agents + 1 generalist fallback
    - Anthropic-based classifier (since we're using Claude agents)
    - In-memory storage (swap to DynamoDB or SQLite for persistence)
"""

import asyncio
import os
from pathlib import Path

from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from agent_squad.agents import AnthropicAgent, AnthropicAgentOptions
from agent_squad.classifiers import AnthropicClassifier, AnthropicClassifierOptions
from agent_squad.storage import InMemoryChatStorage
from agent_squad.types import ConversationMessage


# ---------------------------------------------------------------------------
# Shared context loading
# ---------------------------------------------------------------------------
# Each agent benefits from knowing the current state of the CGAL framework.
# Load the draft once and inject it into each agent's system prompt.
# Adjust the path to wherever you keep cgal_draft.md.

CGAL_DRAFT_PATH = Path(os.environ.get("CGAL_DRAFT_PATH", "cgal_draft.md"))
CGAL_DRAFT = (
    CGAL_DRAFT_PATH.read_text() if CGAL_DRAFT_PATH.exists()
    else "[CGAL draft not found at CGAL_DRAFT_PATH — operating without it]"
)

SHARED_CONTEXT = f"""You are part of a research squad helping the user work on
the CGAL framework (Consensus-Gated Associative Learning), a synthesis built
from the Thousand Brains Theory, predictive coding, binding-by-correlation,
and local consensus learning. The framework's current state is the document
below. When relevant, refer to it by section number.

## CGAL DRAFT (current state)

{CGAL_DRAFT}

## YOUR ROLE

[Filled in per-agent below]

## GENERAL CONVENTIONS

- Be honest about uncertainty. Distinguish what is settled in the framework
  from what is open or speculative.
- Push back when you disagree. Don't pile on validation.
- If a question is outside your specialty, say so and suggest which specialist
  would be better. The user can re-route.
- Cite section numbers (e.g., "see 3.16d") when referencing the framework.
- Keep responses focused. Long answers when needed; short answers when not.
"""


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

# A note on descriptions:
# The classifier routes by reading these descriptions. Vague or overlapping
# descriptions produce flaky routing. Each description below is deliberately
# specific about what its agent handles AND what it does NOT handle. Iterate
# these as you observe routing failures.

THEORIST_DESCRIPTION = """Handles questions about the CGAL framework's
theoretical structure: extending it, finding internal inconsistencies,
working through implications of new ideas, checking whether proposed
mechanisms compose with existing ones. Owns architectural decisions about
the framework itself.

Specifically handles:
- "What does CGAL say about X?"
- "How would Y fit into the framework?"
- "Does Z conflict with the existing machinery?"
- "What are the implications of this design choice?"
- Working through new stepping stones in the reasoning chain.

Does NOT handle:
- Empirical neuroscience questions (route to Neuroscientist).
- Specific Monty implementation details (route to Monty-engineer).
- Experiment design or statistical analysis (route to Experiment-designer).
- Document drafting and revision (route to Writer).
"""

NEUROSCIENTIST_DESCRIPTION = """Handles questions about what is empirically
known in neuroscience and computational cognitive science. Sources claims,
flags speculation, distinguishes well-established findings from contested
or speculative ones, identifies relevant literature.

Specifically handles:
- "Is it actually true that the brain does X?"
- "What's the evidence for/against Y?"
- "Who has studied Z, what did they find?"
- "Is this claim well-established or speculative?"
- Background on specific brain regions, mechanisms, or cognitive phenomena.

Does NOT handle:
- Theoretical extension of CGAL (route to Theorist).
- Code or implementation questions (route to Monty-engineer).
- Whether a proposed experiment would test a claim (route to Experiment-designer).
"""

MONTY_ENGINEER_DESCRIPTION = """Handles questions about Monty's codebase
(github.com/thousandbrainsproject/tbp.monty) and how to implement CGAL
modifications in it. Knows the code structure, identifies where changes
would go, thinks through implementation tradeoffs.

Specifically handles:
- "Where in Monty does X happen?"
- "How would I implement Y in the codebase?"
- "What's the cleanest way to add Z without breaking existing tests?"
- Code review of proposed CGAL modifications.
- GitHub issue drafting for specific code changes.

Does NOT handle:
- Whether a CGAL claim is theoretically right (route to Theorist).
- Whether the experiment design will produce valid results (route to Experiment-designer).
- General Python questions unrelated to Monty.
"""

EXPERIMENT_DESIGNER_DESCRIPTION = """Handles questions about empirical
validation of CGAL claims: experiment design, baselines, metrics, statistical
analysis, what would falsify a claim vs. confirm it.

Specifically handles:
- "How should I test claim X?"
- "What's the right baseline to compare against?"
- "What metrics would meaningfully distinguish CGAL from baseline Monty?"
- "How many seeds/runs do I need for this comparison?"
- "What would null/positive results actually mean?"

Does NOT handle:
- Implementing the experiment in code (route to Monty-engineer).
- Whether the experiment matches a theoretical claim (Theorist may also help).
- Writing up results (route to Writer).
"""

WRITER_DESCRIPTION = """Handles drafting and revision of CGAL-related
documents: the framework draft, GitHub issues, paper sections, blog posts.
Focuses on clarity, structure, tone, and audience.

Specifically handles:
- "Draft a section about X."
- "Revise this paragraph to be clearer."
- "How should I structure this writeup?"
- "What's a good way to explain Y to audience Z?"
- Writing conventions, citation style, document organization.

Does NOT handle:
- Whether the substance is correct (route to relevant specialist).
- Code or implementation details (route to Monty-engineer).
"""

GENERALIST_DESCRIPTION = """Handles questions that don't cleanly fit one
specialist, that span multiple specialties, or that are conversational
(thinking out loud, clarifying questions, meta-questions about the project).
The fallback when no specialist clearly fits.

Specifically handles:
- "What should I work on next?"
- "I'm not sure what's most important here..."
- "Can you summarize where we are?"
- Questions that touch multiple specialties simultaneously.
- Casual conversation about the project's direction.

Does NOT handle anything that clearly fits a specialist — defer to them.
"""


def make_agent(name: str, description: str, role_addendum: str) -> AnthropicAgent:
    """Construct one agent with shared context plus a role-specific addendum."""
    system_prompt = SHARED_CONTEXT.replace(
        "[Filled in per-agent below]",
        role_addendum,
    )
    return AnthropicAgent(
        AnthropicAgentOptions(
            name=name,
            description=description,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_id="claude-opus-4-7",  # adjust if needed
            streaming=True,
            custom_system_prompt={"template": system_prompt},
        )
    )


def build_squad() -> AgentSquad:
    """Build the CGAL research squad with all six agents."""
    classifier = AnthropicClassifier(AnthropicClassifierOptions(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model_id="claude-haiku-4-5-20251001",  # cheap, fast classifier
    ))

    orchestrator = AgentSquad(
        options=AgentSquadConfig(
            LOG_AGENT_CHAT=True,
            LOG_CLASSIFIER_CHAT=True,
            LOG_CLASSIFIER_RAW_OUTPUT=False,
            LOG_CLASSIFIER_OUTPUT=True,
            LOG_EXECUTION_TIMES=True,
        ),
        classifier=classifier,
        storage=InMemoryChatStorage(),  # swap for DynamoDBChatStorage in prod
    )

    orchestrator.add_agent(make_agent(
        name="Theorist",
        description=THEORIST_DESCRIPTION,
        role_addendum=(
            "You are the Theorist. You own the CGAL framework's theoretical "
            "coherence. When the user proposes new ideas, work through whether "
            "they fit, what they imply, and where they break. Be willing to "
            "say 'this conflicts with section X' or 'this is underspecified.' "
            "The pattern of CGAL's development has been simplification through "
            "questioning — preserve that. If a proposed addition can be "
            "absorbed into existing machinery rather than added as a new "
            "mechanism, prefer the absorption."
        ),
    ))

    orchestrator.add_agent(make_agent(
        name="Neuroscientist",
        description=NEUROSCIENTIST_DESCRIPTION,
        role_addendum=(
            "You are the Neuroscientist. Ground claims in actual evidence. "
            "Distinguish well-established findings (replicated, mechanistic) "
            "from suggestive ones (single studies, contested) from speculative "
            "ones (theoretical, not yet tested). Cite researchers and "
            "approximate dates when relevant. If asked about something outside "
            "your knowledge, say so — don't fabricate citations."
        ),
    ))

    orchestrator.add_agent(make_agent(
        name="Monty-engineer",
        description=MONTY_ENGINEER_DESCRIPTION,
        role_addendum=(
            "You are the Monty-engineer. The repo is at "
            "github.com/thousandbrainsproject/tbp.monty (MIT-licensed, Python). "
            "Key paths: src/tbp/monty/frameworks/models/ for models, "
            "src/tbp/monty/conf/experiment/ for configs. When suggesting "
            "code changes, be specific about file paths and methods. Prefer "
            "minimal, reversible diffs over refactors. When uncertain about "
            "the actual codebase structure, say so and suggest the user verify."
        ),
    ))

    orchestrator.add_agent(make_agent(
        name="Experiment-designer",
        description=EXPERIMENT_DESIGNER_DESCRIPTION,
        role_addendum=(
            "You are the Experiment-designer. The four hypotheses we're "
            "testing in the Monty fork are H1 (continual learning), H2 (noise "
            "robustness), H3 (sample efficiency), H4 (no regression on "
            "baseline). Push for falsifiability — what specific result would "
            "refute a claim? Be honest about statistical power and what small "
            "experiments can/can't establish."
        ),
    ))

    orchestrator.add_agent(make_agent(
        name="Writer",
        description=WRITER_DESCRIPTION,
        role_addendum=(
            "You are the Writer. The CGAL draft has a particular voice — "
            "honest about uncertainty, clear about what's settled vs. open, "
            "minimal hype. Preserve that. When drafting, mark speculative "
            "claims explicitly. When revising, prefer concrete prose over "
            "abstract claims. The audience is technical but not necessarily "
            "expert in either neuroscience or ML — assume bridging is needed."
        ),
    ))

    orchestrator.add_agent(make_agent(
        name="Generalist",
        description=GENERALIST_DESCRIPTION,
        role_addendum=(
            "You are the Generalist. You handle the questions that don't "
            "cleanly fit a specialist. If a question would benefit from a "
            "specialist, say so explicitly so the user can re-ask the right "
            "one. Otherwise, engage with the question directly."
        ),
    ))

    return orchestrator


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

async def repl():
    """Simple terminal REPL for the squad."""
    orchestrator = build_squad()
    user_id = os.environ.get("USER", "researcher")
    session_id = "cgal-session-001"  # bump for new conversations

    print("CGAL Research Squad ready. Type 'exit' to quit.\n")
    print("Specialists: Theorist, Neuroscientist, Monty-engineer, "
          "Experiment-designer, Writer, Generalist\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        response = await orchestrator.route_request(
            user_input,
            user_id=user_id,
            session_id=session_id,
        )

        # Surface routing decision so you can see who handled it.
        print(f"\n[routed to: {response.metadata.agent_name}]")

        if response.streaming:
            async for chunk in response.output:
                print(chunk, end="", flush=True)
            print()
        else:
            content = response.output
            if isinstance(content, ConversationMessage):
                content = content.content[0].get("text", "")
            print(content)
        print()


if __name__ == "__main__":
    asyncio.run(repl())
