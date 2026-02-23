import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext
from typing import Any, Dict

from arena_engine import GomokuEngine
from arena_orchestrator import ArenaOrchestrator


class ArenaLiveWindow:
    def __init__(self, engine: GomokuEngine, title: str = "LLM Arena - Gomoku"):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title(title)
        self._closed = False
        self._worker_thread = None
        self.root.protocol("WM_DELETE_WINDOW", self._on_manual_close)

        self.cell = 42
        self.margin = 20
        canvas_size = self.margin * 2 + self.cell * (self.engine.size - 1)

        self.status_var = tk.StringVar(value="Game initializing...")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="#f2d7a3")
        self.canvas.pack(padx=8, pady=4)

        self.log_box = scrolledtext.ScrolledText(self.root, height=16, width=100, state="disabled")
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        self._draw_grid()

    def _coords(self, row: int, col: int):
        x = self.margin + col * self.cell
        y = self.margin + row * self.cell
        return x, y

    def _draw_grid(self):
        self.canvas.delete("all")
        n = self.engine.size
        for i in range(n):
            x1, y1 = self._coords(0, i)
            x2, y2 = self._coords(n - 1, i)
            self.canvas.create_line(x1, y1, x2, y2, fill="#333333")

            x3, y3 = self._coords(i, 0)
            x4, y4 = self._coords(i, n - 1)
            self.canvas.create_line(x3, y3, x4, y4, fill="#333333")

    def _draw_stones(self):
        self._draw_grid()
        for idx, (player, row, col) in enumerate(self.engine.history, start=1):
            x, y = self._coords(row, col)
            r = 14
            fill = "black" if player == 1 else "white"
            text_color = "white" if player == 1 else "black"
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline="black", width=1)
            self.canvas.create_text(x, y, text=str(idx), fill=text_color, font=("Arial", 8, "bold"))

    def _append_log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.configure(state="disabled")
        self.log_box.see(tk.END)

    def _on_manual_close(self):
        # Mark closed first so background callbacks stop touching widgets.
        self._closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def handle_event(self, event_type: str, payload: Dict[str, Any]):
        # Ensure UI updates execute in Tk main thread.
        if self._closed:
            return
        try:
            self.root.after(0, lambda: self._handle_event_on_ui_thread(event_type, payload))
        except tk.TclError:
            self._closed = True

    def _handle_event_on_ui_thread(self, event_type: str, payload: Dict[str, Any]):
        if self._closed:
            return
        if event_type == "init":
            self.status_var.set("Game started. Initial rules sent to both models.")
            self._append_log("=== Game Init ===")
            for item in payload.get("logs", []):
                self._append_log(f"[INIT][{item.get('player')}] agent={item.get('agent')} | {item.get('reasoning')}")
                self._append_log(f"Agent->LLM: {item.get('agent_message')}")
                self._append_log(f"LLM said: {item.get('raw_response')}")
            return

        if event_type == "turn":
            log = payload.get("log", {})
            self._draw_stones()
            self.status_var.set(
                f"Turn {log.get('turn_index')} | {log.get('player')} ({log.get('agent')}) "
                f"placed at ({log.get('row')}, {log.get('col')})"
            )
            self._append_log(
                f"[TURN {log.get('turn_index')}] {log.get('player')} / {log.get('agent')} -> "
                f"({log.get('row')}, {log.get('col')}) | applied={log.get('applied')}"
            )
            self._append_log(f"Agent->LLM: {log.get('agent_message')}")
            self._append_log(f"LLM said: {log.get('raw_response')}")
            return

        if event_type == "meta":
            log = payload.get("log", {})
            self._draw_stones()
            self.status_var.set(f"Opening decision | {log.get('player')} ({log.get('agent')})")
            self._append_log(
                f"[OPENING] {log.get('player')} / {log.get('agent')} | {log.get('raw_response')}"
            )
            self._append_log(f"Agent->LLM: {log.get('agent_message')}")
            return

        if event_type == "end":
            winner = payload.get("winner", "No result")
            self._draw_stones()
            self.status_var.set(f"Game ended: {winner}")
            self._append_log(f"=== Game Ended: {winner} ===")
            try:
                messagebox.showinfo("Game Finished", f"Result: {winner}")
            finally:
                # Use destroy (not quit) so window does not remain frozen.
                self._on_manual_close()

    def run_with_orchestrator(self, orchestrator: ArenaOrchestrator, max_turns: int = 200):
        def _worker():
            orchestrator.run(max_turns=max_turns)

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()
        self.root.mainloop()
