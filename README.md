# Wajeeha AI

First working slice of an autonomous AI assistant: **Brain** (LLM router +
planner) + **Memory** (short-term, long-term, semantic) + **Coding Agent**
+ **Home Agent** (controls a Google Home Mini via Home Assistant).

More modules (voice, vision, desktop control, networking, security, UI,
mobile app, etc.) are being added on top of this incrementally.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running locally
  (free, runs on your own machine — no API key needed)
- A running [Home Assistant](https://www.home-assistant.io/) instance if
  you want the Home agent active (optional)

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1
cp .env.example .env
```

The default LLM provider is set to `ollama` in `config/config.yaml`, so no
API key is required out of the box. If you'd rather use a paid cloud
provider (Anthropic, OpenAI, Gemini) for better response quality, add the
relevant key to `.env` and change `default_provider` in
`config/config.yaml` to match.

## Run

```bash
python cli.py chat                # interactive REPL
python cli.py run "list the files in my workspace"   # one-shot
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


## Notes

- **Home agent / Google Home Mini:** Google doesn't expose a general
  local-control API for Google Home Mini. This agent talks to Home
  Assistant instead — see the comment block at the top of
  `agents/home_agent.py` for the full explanation. Until
  `HOME_ASSISTANT_TOKEN` is set in `.env`, the home agent is disabled and
  only the coding agent runs.
- **Local model quality:** `llama3.1` via Ollama is free but less reliable
  at following strict instructions than cloud models — short/casual
  messages ("hii") sometimes produce no response. Fuller sentences work
  better.