import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from arena_agents import BaseLLMAgent, build_initial_rules_prompt
from arena_engine import GomokuEngine, Move


class ArenaOrchestrator:
    def __init__(
        self,
        engine: GomokuEngine,
        black_agent: BaseLLMAgent,
        white_agent: BaseLLMAgent,
        max_retries: int = 2,
        render_each_turn: bool = True,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        turn_delay_sec: float = 0.0,
        opening_rule: str = "swap",
        illegal_retry_limit: int = 3,
        log_root_dir: str = "arena_logs",
        hint: bool = True,
    ):
        self.engine = engine
        self.agents = {1: black_agent, -1: white_agent}
        self.max_retries = max_retries
        self.render_each_turn = render_each_turn
        self.event_callback = event_callback
        self.turn_delay_sec = turn_delay_sec
        self.opening_rule = opening_rule.lower().strip()
        self.illegal_retry_limit = illegal_retry_limit
        self.log_root_dir = log_root_dir
        self.hint = hint
        self.logs: List[Dict[str, Any]] = []
        self.event_logs: List[Dict[str, Any]] = []
        self._initialized = False
        self._last_action_by_player: Dict[int, str] = {}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.log_root_dir, self.run_id)

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.event_logs.append(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "event_type": event_type,
                "payload": self._to_jsonable(payload),
            }
        )
        if self.event_callback:
            self.event_callback(event_type, payload)

    def _to_jsonable(self, obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                return str(obj)
        return str(obj)

    def _persist_logs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        logs_df = pd.DataFrame(self.logs)
        logs_df.to_csv(os.path.join(self.output_dir, "turn_logs.csv"), index=False, encoding="utf-8")
        with open(os.path.join(self.output_dir, "turn_logs.jsonl"), "w", encoding="utf-8") as f:
            for item in self.logs:
                f.write(json.dumps(self._to_jsonable(item), ensure_ascii=False) + "\n")

        with open(os.path.join(self.output_dir, "event_logs.jsonl"), "w", encoding="utf-8") as f:
            for item in self.event_logs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        summary = {
            "run_id": self.run_id,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "winner": self.winner_text(),
            "board_size": self.engine.size,
            "win_len": self.engine.win_len,
            "opening_rule": self.opening_rule,
            "illegal_retry_limit": self.illegal_retry_limit,
            "hint": self.hint,
            "move_count": self.engine.move_count,
            "black_agent": self.agents[1].name,
            "white_agent": self.agents[-1].name,
        }
        with open(os.path.join(self.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.output_dir, "final_board.json"), "w", encoding="utf-8") as f:
            json.dump(self._to_jsonable(self.engine.board), f, ensure_ascii=False)

    def _format_action_feedback(self, row: int, col: int, reason: str) -> str:
        if self.hint:
            return f"[action] place at row {row}, column {col}; [reason] {reason}"
        return f"[action] place at row {row}, column {col}"

    def _broadcast_initial_prompt(self) -> None:
        if self._initialized:
            return
        initial_prompt = build_initial_rules_prompt(self.engine)
        if self.opening_rule == "swap":
            initial_prompt += (
                "\n\nOpening rule: SWAP (Pie Rule).\n"
                "1) Black makes the first move.\n"
                "2) White then decides whether to swap colors.\n"
                "3) If white swaps, white becomes black and original black becomes white."
            )

        for marker, agent in self.agents.items():
            player = "BLACK" if marker == 1 else "WHITE"
            try:
                agent.initialize_game(self.engine, initial_prompt)
                self.logs.append(
                    {
                        "turn_index": 0,
                        "player": player,
                        "agent": agent.name,
                        "model": agent.model,
                        "row": None,
                        "col": None,
                        "applied": True,
                        "reasoning": "Initial shared rules prompt sent.",
                        "raw_response": "[init] READY",
                        "strict_format_matched": None,
                        "agent_message": initial_prompt,
                        "opponent_last_action": None,
                    }
                )
            except Exception as e:
                self.logs.append(
                    {
                        "turn_index": 0,
                        "player": player,
                        "agent": agent.name,
                        "model": agent.model,
                        "row": None,
                        "col": None,
                        "applied": False,
                        "reasoning": f"Failed to send initial prompt: {e}",
                        "raw_response": "",
                        "strict_format_matched": None,
                        "agent_message": initial_prompt,
                        "opponent_last_action": None,
                    }
                )
        self._initialized = True
        self._emit(
            "init",
            {
                "board": self.engine.board.copy(),
                "size": self.engine.size,
                "logs": self.logs[-2:],
            },
        )

    def _run_swap_opening_phase(self) -> None:
        if self.opening_rule != "swap" or self.engine.move_count != 0 or self.engine.winner != 0:
            return

        # Step 1: black makes first move.
        black_agent = self.agents[1]
        first_move, fail_reason = self._safe_get_move_with_context(black_agent, opponent_last_action=None)
        if first_move is None:
            self.engine.winner = -1
            forfeit_log = {
                "turn_index": self.engine.move_count,
                "player": "BLACK",
                "agent": black_agent.name,
                "model": black_agent.model,
                "row": None,
                "col": None,
                "applied": False,
                "reasoning": (
                    f"Forfeit in opening phase: exceeded invalid-action retry limit ({self.illegal_retry_limit}). "
                    f"Last error: {fail_reason}"
                ),
                "raw_response": "",
                "strict_format_matched": None,
                "agent_message": "Opening phase: black first move under SWAP rule.",
                "opponent_last_action": None,
            }
            self.logs.append(forfeit_log)
            self._emit(
                "meta",
                {
                    "board": self.engine.board.copy(),
                    "history": list(self.engine.history),
                    "log": forfeit_log,
                    "winner": self.engine.winner,
                },
            )
            return
        ok = self.engine.apply_move(first_move.row, first_move.col)
        first_action_text = self._format_action_feedback(
            row=first_move.row,
            col=first_move.col,
            reason=first_move.reasoning or "no reason provided",
        )
        self._last_action_by_player[1] = first_action_text

        first_log = {
            "turn_index": self.engine.move_count,
            "player": "BLACK",
            "agent": black_agent.name,
            "model": black_agent.model,
            "row": first_move.row,
            "col": first_move.col,
            "applied": ok,
            "reasoning": first_move.reasoning,
            "raw_response": first_move.raw_response,
            "strict_format_matched": first_move.strict_format_matched,
            "agent_message": "Opening phase: black first move under SWAP rule.",
            "opponent_last_action": None,
        }
        self.logs.append(first_log)
        self._emit(
            "turn",
            {
                "board": self.engine.board.copy(),
                "history": list(self.engine.history),
                "log": first_log,
                "winner": self.engine.winner,
            },
        )
        if self.turn_delay_sec > 0:
            time.sleep(self.turn_delay_sec)

        # Step 2: white decides swap or keep.
        white_agent = self.agents[-1]
        swap_error = None
        do_swap = False
        try:
            do_swap = bool(white_agent.decide_swap(first_action_text))
        except Exception as e:
            swap_error = str(e)

        if do_swap:
            self.agents[1], self.agents[-1] = self.agents[-1], self.agents[1]

        decision_action = "[action] swap" if do_swap else "[action] keep"
        decision_reason = "White chose to swap colors." if do_swap else "White kept original colors."
        if swap_error:
            decision_reason = f"Swap decision fallback to keep due to error: {swap_error}"

        swap_log = {
            "turn_index": self.engine.move_count,
            "player": "WHITE",
            "agent": white_agent.name,
            "model": white_agent.model,
            "row": None,
            "col": None,
            "applied": True,
            "reasoning": decision_reason,
            "raw_response": f"{decision_action}; [reason] {decision_reason}",
            "strict_format_matched": None,
            "agent_message": f"Opening phase: decide swap based on black first move: {first_action_text}",
            "opponent_last_action": first_action_text,
        }
        self.logs.append(swap_log)
        self._emit(
            "meta",
            {
                "board": self.engine.board.copy(),
                "history": list(self.engine.history),
                "log": swap_log,
                "winner": self.engine.winner,
            },
        )
        if self.turn_delay_sec > 0:
            time.sleep(self.turn_delay_sec)

    def _safe_get_move_with_context(
        self, agent: BaseLLMAgent, opponent_last_action: str = None
    ) -> Tuple[Optional[Move], Optional[str]]:
        last_error = None
        for attempt in range(1, self.illegal_retry_limit + 1):
            try:
                move = agent.generate_move(self.engine, opponent_last_action=opponent_last_action)
                if self.engine.is_legal(move.row, move.col):
                    return move, None
                last_error = (
                    f"illegal move ({move.row}, {move.col}), "
                    f"attempt {attempt}/{self.illegal_retry_limit}"
                )
            except Exception as e:
                last_error = f"{e} (attempt {attempt}/{self.illegal_retry_limit})"
        return None, last_error

    def run(self, max_turns: int = 200) -> pd.DataFrame:
        self._broadcast_initial_prompt()
        self._run_swap_opening_phase()

        if self.render_each_turn:
            self.engine.render("Initial Board")

        while self.engine.winner == 0 and self.engine.move_count < max_turns:
            agent = self.agents[self.engine.turn]
            player = self.engine.turn
            opponent = -player
            opponent_last_action = self._last_action_by_player.get(opponent)

            move, fail_reason = self._safe_get_move_with_context(agent, opponent_last_action=opponent_last_action)
            if move is None:
                self.engine.winner = opponent
                forfeit_log = {
                    "turn_index": self.engine.move_count,
                    "player": "BLACK" if player == 1 else "WHITE",
                    "agent": agent.name,
                    "model": agent.model,
                    "row": None,
                    "col": None,
                    "applied": False,
                    "reasoning": (
                        f"Forfeit: exceeded invalid-action retry limit ({self.illegal_retry_limit}). "
                        f"Last error: {fail_reason}"
                    ),
                    "raw_response": "",
                    "strict_format_matched": None,
                    "agent_message": (
                        "Opponent previous action: "
                        + (opponent_last_action if opponent_last_action else "No opponent move yet (you move first).")
                    ),
                    "opponent_last_action": opponent_last_action,
                }
                self.logs.append(forfeit_log)
                self._emit(
                    "turn",
                    {
                        "board": self.engine.board.copy(),
                        "history": list(self.engine.history),
                        "log": forfeit_log,
                        "winner": self.engine.winner,
                    },
                )
                break
            ok = self.engine.apply_move(move.row, move.col)

            action_text = self._format_action_feedback(
                row=move.row,
                col=move.col,
                reason=move.reasoning or "no reason provided",
            )
            self._last_action_by_player[player] = action_text

            self.logs.append(
                {
                    "turn_index": self.engine.move_count,
                    "player": "BLACK" if player == 1 else "WHITE",
                    "agent": agent.name,
                    "model": agent.model,
                    "row": move.row,
                    "col": move.col,
                    "applied": ok,
                    "reasoning": move.reasoning,
                    "raw_response": move.raw_response,
                    "strict_format_matched": move.strict_format_matched,
                    "agent_message": (
                        "Opponent previous action: "
                        + (opponent_last_action if opponent_last_action else "No opponent move yet (you move first).")
                    ),
                    "opponent_last_action": opponent_last_action,
                }
            )
            latest_log = self.logs[-1]
            self._emit(
                "turn",
                {
                    "board": self.engine.board.copy(),
                    "history": list(self.engine.history),
                    "log": latest_log,
                    "winner": self.engine.winner,
                },
            )

            if self.render_each_turn:
                status = "Running"
                if self.engine.winner == 1:
                    status = "BLACK wins"
                elif self.engine.winner == -1:
                    status = "WHITE wins"
                elif self.engine.winner == 2:
                    status = "Draw"
                self.engine.render(f"Turn {self.engine.move_count} | {status}")

            if self.turn_delay_sec > 0:
                time.sleep(self.turn_delay_sec)

        self._emit(
            "end",
            {
                "winner": self.winner_text(),
                "board": self.engine.board.copy(),
                "history": list(self.engine.history),
                "logs": self.logs,
            },
        )
        self._persist_logs()
        return pd.DataFrame(self.logs)

    def winner_text(self) -> str:
        if self.engine.winner == 1:
            return "BLACK wins"
        if self.engine.winner == -1:
            return "WHITE wins"
        if self.engine.winner == 2:
            return "Draw"
        return "No result"
