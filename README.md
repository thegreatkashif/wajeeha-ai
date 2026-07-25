# Wajeeha AI

First working slice of an autonomous AI assistant: **Brain** (LLM router +
planner) + **Memory** (short-term, long-term, semantic) + **Coding Agent**
+ **Home Agent** (controls a Google Home Mini via Home Assistant).

More modules (voice, vision, desktop control, networking, security, UI,
mobile app, etc.) are being added on top of this incrementally.

## Requirements

- Python 3.11+
- An Anthropic (or OpenAI/Gemini/Ollama) API key
- A running [Home Assistant](https://www.home-assistant.io/) instance if
  you want the Home agent active

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then fill in your API key(s)
```

Edit `config/config.yaml` if you want to change the default LLM provider,
model, memory paths, or agent settings.

## Run

```bash
python cli.py chat                # interactive REPL
python cli.py run "turn off the living room lights"   # one-shot
```

## Test

```bash
pytest
```

## Layout

wajeeha-ai/
brain/ LLM router, planner, orchestrator
memory/ short-term, long-term (SQLite), semantic (Chroma) memory
agents/ base agent + coding agent + home agent
config/ config.yaml (behavior) + settings.py (typed loader)
tests/ pytest suite
cli.py entry point

## Notes on the Home agent

Google doesn't expose a general local-control API for Google Home Mini.
This agent talks to Home Assistant instead — see the comment block at the
top of `agents/home_agent.py` for the full explanation.