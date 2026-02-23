from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Move:
    row: int
    col: int
    reasoning: str = ""
    raw_response: str = ""
    strict_format_matched: bool = False


class GomokuEngine:
    def __init__(self, size: int = 9, win_len: int = 5):
        self.size = size
        self.win_len = win_len
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)  # 0 empty, 1 black, -1 white
        self.turn = 1
        self.winner = 0
        self.move_count = 0
        self.history: List[Tuple[int, int, int]] = []  # (player, row, col)

    def legal_moves(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]

    def is_legal(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row, col] == 0

    def apply_move(self, row: int, col: int) -> bool:
        if not self.is_legal(row, col) or self.winner != 0:
            return False
        self.board[row, col] = self.turn
        self.history.append((self.turn, row, col))
        self.move_count += 1

        if self._check_win_from(row, col):
            self.winner = self.turn
        elif self.move_count == self.size * self.size:
            self.winner = 2  # draw
        else:
            self.turn *= -1
        return True

    def _count_dir(self, row: int, col: int, dr: int, dc: int) -> int:
        player = self.board[row, col]
        cnt = 0
        r, c = row + dr, col + dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
            cnt += 1
            r += dr
            c += dc
        return cnt

    def _check_win_from(self, row: int, col: int) -> bool:
        # Four directions: horizontal, vertical, diag, anti-diag
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            total = 1 + self._count_dir(row, col, dr, dc) + self._count_dir(row, col, -dr, -dc)
            if total >= self.win_len:
                return True
        return False

    def state_for_prompt(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "win_len": self.win_len,
            "turn": "BLACK(1)" if self.turn == 1 else "WHITE(-1)",
            "board": self.board.tolist(),
            "legal_moves": self.legal_moves(),
            "history": [{"player": p, "row": r, "col": c} for (p, r, c) in self.history],
        }

    def render(self, title: str = "Gomoku Arena"):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(title)
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(self.size - 0.5, -0.5)
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid(True)

        for idx, (player, r, c) in enumerate(self.history, start=1):
            color = "black" if player == 1 else "white"
            stone = plt.Circle((c, r), 0.38, facecolor=color, edgecolor="black", linewidth=1.2)
            ax.add_patch(stone)
            text_color = "white" if player == 1 else "black"
            ax.text(c, r, str(idx), ha="center", va="center", fontsize=8, color=text_color)

        plt.show()
