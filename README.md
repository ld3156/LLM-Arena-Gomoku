# 🧠 LLM Arena: Gomoku (OpenAI vs Gemini)

This project pits different Large Language Models against each other in a constrained Gomoku (Five-in-a-Row) environment to trace and evaluate their reasoning and behavior. The primary goal of this framework is to observe model performance regarding **rule adherence, long-term memory, and strategic stability.**

![Game Preview](screenshot/gomoku-3.png)

An extensible LLM-vs-LLM competitive framework for **Gomoku (Five-in-a-Row)**, focused on:

- reasoning behavior
- instruction-following
- legality compliance
- memory under constrained context

This project is built for reproducible model benchmarking in a controlled game environment.

---

## 🏗️ Core Architecture

### 1) Game Structure

- **Game Engine (Referee)**  
  Maintains board state, computes legal moves, and determines game-over conditions (including five-in-a-row win detection).

- **Orchestrator (Arena)**  
  Drives turn-based gameplay, applies opening rules (including **SWAP / Pie Rule**), calls LLM APIs, validates move legality, and records events.

- **Live UI**  
  Shows the board at the top and real-time logs below, including:
  - Agent input to the LLM
  - LLM raw output
  - Parsed move
  - Referee judgments

- **Persistent Logs**  
  Automatically stores full match records (`CSV`, `JSONL`, `summary`, `final board`) for reproducibility and post-game analysis.

---

### 2) Role of the Agent

The Agent acts as an **adapter + strategy layer** between model APIs and game rules.

Its standardized responsibilities:

1. Initialize model context with a unified opening prompt.
2. Generate turn actions and parse `row/col` from model output.
3. Handle exceptions (format failures, API errors, illegal coordinates).

The project uses strict referee enforcement:

- Every turn is legality-checked.
- If parsing/legality fails, the system retries.
- If failures exceed the threshold (**default: 3 attempts**), the model **forfeits**.

To study information asymmetry and prompt leakage, a `hint` switch is provided:

- `hint=True` → send opponent's previous **action + reason**
- `hint=False` → send only opponent's previous **action** (reason hidden)

---

### 3) Prompting Scheme (Two-Stage Prompting)

#### A. Opening Unified Rule Prompt
Sent identically to both players, including:

- board size
- win condition
- coordinate convention (**0-based**)
- output constraints
- opening **SWAP / Pie Rule**:
  - Black places the first stone
  - White decides to `swap` or `keep`

#### B. Turn-by-Turn Prompt

- The full board is **not resent every turn**.
- Only opponent last move info is provided (controlled by `hint`).
- Models must output in fixed format:

```text
[action] place at row X, column Y; [reason] <less than 100 words>
```

For format-compliance evaluation, logs include:

- `strict_format_matched=True` → exact target format matched
- `strict_format_matched=False` → parsed via loose fallback (semantically valid, format not exact)

---

## ✅ Current Status

- Stable matches between **OpenAI** and **Gemini** models
- Modular architecture for easy extension
- New providers can be added by implementing a new Agent Adapter

---

## 📁 Project Structure

```text
.
├─ arena_engine.py         # Referee/game engine
├─ arena_agents.py         # Agent interface + OpenAI/Gemini adapters
├─ arena_orchestrator.py   # Match loop, legality checks, retries, logging
├─ arena_ui.py             # Live window UI (board + logs)
├─ LLM_Arena_Gomoku.ipynb  # Main notebook runner
├─ requirements.txt
└─ arena_logs/             # Auto-generated per-match logs
```

---

## 🚀 Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Open `LLM_Arena_Gomoku.ipynb`
3. Configure API keys
4. Select model names
5. Run the main match cell

---

## 🧪 Output Artifacts (per match)

Each run is saved to:

```text
arena_logs/<timestamp>/
```

Containing:

- `turn_logs.csv`
- `turn_logs.jsonl`
- `event_logs.jsonl`
- `summary.json`
- `final_board.json`

---

## 🔭 Extensibility

To test different LLM performance:
Find the following code in notebook, and change to other model names, such as gemini-2.5-flash
```text
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-pro"
```

Notice that as of Feb22 2026, OpenAI supports gpt-5.2 and Gemini supports gemini-2.5-pro for API usage.
Gemini-3 series is not available.

---

## 📌 Notes

- If you see model `404`/`not found`, verify model IDs against your account-accessible model list.
- For strict tournament behavior, keep `illegal_retry_limit=3` and forfeit on repeated invalid outputs.
- Use `hint=False` for stronger information-isolation experiments.
