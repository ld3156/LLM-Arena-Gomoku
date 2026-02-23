import json
import random
import re
from abc import ABC, abstractmethod
from typing import Optional

from arena_engine import GomokuEngine, Move


class BaseLLMAgent(ABC):
    def __init__(self, name: str, marker: int, model: str):
        self.name = name
        self.marker = marker  # 1 or -1
        self.model = model
        self.game_rules_prompt = ""

    def initialize_game(self, engine: GomokuEngine, initial_prompt: str) -> None:
        # Default implementation: cache prompt for turn-level generation.
        self.game_rules_prompt = initial_prompt

    @abstractmethod
    def generate_move(self, engine: GomokuEngine, opponent_last_action: Optional[str] = None) -> Move:
        pass

    def decide_swap(self, first_action_text: str) -> bool:
        # Default: do not swap.
        return False


class RandomAgent(BaseLLMAgent):
    """Fallback agent for local flow testing without API keys."""

    def __init__(self, name: str = "Random", marker: int = 1):
        super().__init__(name=name, marker=marker, model="random")

    def generate_move(self, engine: GomokuEngine, opponent_last_action: Optional[str] = None) -> Move:
        row, col = random.choice(engine.legal_moves())
        raw = f"[action] place at row {row}, column {col}; [reason] random fallback move"
        return Move(
            row=row,
            col=col,
            reasoning="random fallback move",
            raw_response=raw,
            strict_format_matched=True,
        )

    def decide_swap(self, first_action_text: str) -> bool:
        return random.random() < 0.5


def build_initial_rules_prompt(engine: GomokuEngine) -> str:
    return (
        "You are joining a Gomoku match.\n"
        "Global rules shared to both players:\n"
        f"- Board size: {engine.size} x {engine.size}\n"
        f"- Win condition: {engine.win_len} in a row\n"
        "- Players alternate one legal move each turn.\n"
        "- You must never place on occupied cells or outside the board.\n"
        "- Row/column indices are 0-based.\n"
        "For every turn, your output format MUST be exactly:\n"
        '"[action] place at row X, column Y; [reason] <less than 100 words>"'
    )


def build_turn_prompt(agent_name: str, opponent_last_action: Optional[str]) -> str:
    opponent_feedback = opponent_last_action or "No opponent move yet (you move first)."
    return (
        f"You are {agent_name}. It is your turn.\n"
        "Opponent previous action:\n"
        f"{opponent_feedback}\n\n"
        "Return exactly one line in this format:\n"
        '"[action] place at row X, column Y; [reason] <less than 100 words>"\n\n'
        "Do not output markdown or extra lines.\n\n"
        "Important: infer board state from full action history in the conversation."
    )


def build_swap_decision_prompt(agent_name: str, first_action_text: str) -> str:
    return (
        f"You are {agent_name}. Opening swap decision phase.\n"
        f"Black first action was:\n{first_action_text}\n\n"
        "Reply exactly one line in this format:\n"
        '"[action] swap; [reason] <less than 100 words>"\n'
        "or\n"
        '"[action] keep; [reason] <less than 100 words>"\n\n'
        "Do not output markdown or extra lines."
    )


def parse_move_from_text(text: str) -> Optional[Move]:
    if not text:
        return None

    text = text.strip()

    fmt_match = re.search(
        r"\[action\]\s*place\s+at\s+row\s*(-?\d+)\s*,\s*column\s*(-?\d+)\s*;\s*\[reason\]\s*(.+)",
        text,
        flags=re.I | re.S,
    )
    if fmt_match:
        return Move(
            row=int(fmt_match.group(1)),
            col=int(fmt_match.group(2)),
            reasoning=fmt_match.group(3).strip(),
            raw_response=text,
            strict_format_matched=True,
        )

    # 兼容宽松格式，例如 [place] ...，会被接受但标记为非 strict。
    loose_match = re.search(
        r"\[(?:action|place)\]\s*place\s+at\s+row\s*(-?\d+)\s*,\s*column\s*(-?\d+)\s*;\s*\[reason\]\s*(.+)",
        text,
        flags=re.I | re.S,
    )
    if loose_match:
        return Move(
            row=int(loose_match.group(1)),
            col=int(loose_match.group(2)),
            reasoning=loose_match.group(3).strip(),
            raw_response=text,
            strict_format_matched=False,
        )

    try:
        data = json.loads(text)
        return Move(
            row=int(data["row"]),
            col=int(data["col"]),
            reasoning=str(data.get("reasoning", "")),
            raw_response=text,
            strict_format_matched=False,
        )
    except Exception:
        pass

    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return Move(
                row=int(data["row"]),
                col=int(data["col"]),
                reasoning=str(data.get("reasoning", "")),
                raw_response=text,
                strict_format_matched=False,
            )
        except Exception:
            return None

    return None


def parse_swap_decision(text: str) -> Optional[bool]:
    if not text:
        return None
    match = re.search(r"\[action\]\s*(swap|keep)\s*;\s*\[reason\]\s*(.+)", text.strip(), flags=re.I | re.S)
    if not match:
        return None
    action = match.group(1).strip().lower()
    return action == "swap"


class OpenAIAgent(BaseLLMAgent):
    def __init__(self, api_key: str, marker: int, model: str = "gpt-4o-mini", name: str = "OpenAI"):
        super().__init__(name=name, marker=marker, model=model)
        if not api_key:
            raise ValueError("OpenAI API key is required")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self._messages = []

    def initialize_game(self, engine: GomokuEngine, initial_prompt: str) -> None:
        super().initialize_game(engine, initial_prompt)
        self._messages = [
            {"role": "system", "content": "You are preparing for a Gomoku match."},
            {"role": "user", "content": initial_prompt + "\n\nReply READY."},
        ]
        resp = self.client.chat.completions.create(model=self.model, temperature=0, messages=self._messages)
        ready_text = resp.choices[0].message.content or "READY"
        self._messages.append({"role": "assistant", "content": ready_text})

    def generate_move(self, engine: GomokuEngine, opponent_last_action: Optional[str] = None) -> Move:
        prompt = build_turn_prompt(self.name, opponent_last_action)
        turn_messages = self._messages + [{"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=turn_messages,
        )
        text = resp.choices[0].message.content or ""
        self._messages.append({"role": "user", "content": prompt})
        self._messages.append({"role": "assistant", "content": text})
        move = parse_move_from_text(text)
        if move is None:
            raise ValueError(f"OpenAI invalid response: {text[:200]}")
        return move

    def decide_swap(self, first_action_text: str) -> bool:
        prompt = build_swap_decision_prompt(self.name, first_action_text)
        turn_messages = self._messages + [{"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=turn_messages,
        )
        text = resp.choices[0].message.content or ""
        self._messages.append({"role": "user", "content": prompt})
        self._messages.append({"role": "assistant", "content": text})
        decision = parse_swap_decision(text)
        if decision is None:
            raise ValueError(f"OpenAI invalid swap decision: {text[:200]}")
        return decision


class GeminiAgent(BaseLLMAgent):
    def __init__(self, api_key: str, marker: int, model: str = "gemini-1.5-flash", name: str = "Gemini"):
        super().__init__(name=name, marker=marker, model=model)
        if not api_key:
            raise ValueError("Gemini API key is required")
        try:
            import google.genai as genai
        except Exception:
            try:
                from google import genai  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Cannot import google.genai. Please run: pip install -U google-genai, "
                    "then restart the notebook kernel."
                ) from e

        self.client = genai.Client(api_key=api_key)
        self.chat = None

    def initialize_game(self, engine: GomokuEngine, initial_prompt: str) -> None:
        super().initialize_game(engine, initial_prompt)
        self.chat = self.client.chats.create(model=self.model)
        _ = self.chat.send_message(initial_prompt + "\n\nReply READY.")

    def generate_move(self, engine: GomokuEngine, opponent_last_action: Optional[str] = None) -> Move:
        if self.chat is None:
            self.chat = self.client.chats.create(model=self.model)
            if self.game_rules_prompt:
                _ = self.chat.send_message(self.game_rules_prompt + "\n\nReply READY.")
        prompt = build_turn_prompt(self.name, opponent_last_action)
        resp = self.chat.send_message(prompt)
        text = getattr(resp, "text", "") or ""
        if not text and hasattr(resp, "candidates") and resp.candidates:
            # Compatibility fallback for SDK response variants.
            try:
                parts = resp.candidates[0].content.parts
                text = "".join(getattr(p, "text", "") for p in parts).strip()
            except Exception:
                text = ""
        move = parse_move_from_text(text)
        if move is None:
            raise ValueError(f"Gemini invalid response: {text[:200]}")
        return move

    def decide_swap(self, first_action_text: str) -> bool:
        if self.chat is None:
            self.chat = self.client.chats.create(model=self.model)
            if self.game_rules_prompt:
                _ = self.chat.send_message(self.game_rules_prompt + "\n\nReply READY.")
        prompt = build_swap_decision_prompt(self.name, first_action_text)
        resp = self.chat.send_message(prompt)
        text = getattr(resp, "text", "") or ""
        if not text and hasattr(resp, "candidates") and resp.candidates:
            try:
                parts = resp.candidates[0].content.parts
                text = "".join(getattr(p, "text", "") for p in parts).strip()
            except Exception:
                text = ""
        decision = parse_swap_decision(text)
        if decision is None:
            raise ValueError(f"Gemini invalid swap decision: {text[:200]}")
        return decision
