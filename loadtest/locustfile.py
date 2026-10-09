"""Locust load test (Phase 6): ≥50 concurrent chat sessions.

Measures two custom metrics per session:
  - chat            : full streaming request duration
  - chat_first_token: time to the first streamed token (acceptance: p95 < 3s)

Run against a running API (set API_KEY if the API is keyed):

    API_KEY=... locust -f loadtest/locustfile.py --host http://localhost:8000 \
        -u 50 -r 5 -t 120s --headless

or headful: `locust -f loadtest/locustfile.py --host http://localhost:8000`
then open http://localhost:8089 and start 50 users.
"""

from __future__ import annotations

import json
import os
import time
from itertools import cycle

import requests
from locust import User, constant, task

API_KEY = os.environ.get("API_KEY", "")
QUERIES = cycle(
    [
        "What is the exit load of the fund?",
        "What are the key person clauses?",
        "How is the management fee calculated?",
        "What is the benchmark for the fund?",
        "Summarize the investment objective",
        "What are the redemption terms?",
    ]
)


def _fire(environment, name: str, response_time: float, exception=None) -> None:
    environment.events.request.fire(
        request_type="POST",
        name=name,
        response_time=response_time,
        response_length=0,
        exception=exception,
        context={},
        url=name,
    )


class ChatUser(User):
    wait_time = constant(1)
    abstract = False

    def on_start(self) -> None:
        self.headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
        self.conversation_id: str | None = None

    @task
    def ask_question(self) -> None:
        query = next(QUERIES)
        started = time.perf_counter()
        first_token_ms: float | None = None
        try:
            with self.client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": self.conversation_id,
                    "message": query,
                    "history": [],
                },
                headers=self.headers,
                stream=True,
                catch_response=True,
                name="chat",
            ) as response:
                if response.status_code != 200:
                    response.failure(f"status {response.status_code}")
                    return
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: "):])
                    if first_token_ms is None and event.get("type") in {"token", "error"}:
                        first_token_ms = (time.perf_counter() - started) * 1000
                        _fire(self.environment, "chat_first_token", first_token_ms)
                        if event.get("type") == "error":
                            response.failure(event.get("message", "stream error"))
                            return
                    if event.get("type") == "done":
                        self.conversation_id = event.get(
                            "conversation_id", self.conversation_id
                        )
                response.success()
        except requests.RequestException as exc:
            _fire(
                self.environment,
                "chat_first_token",
                (time.perf_counter() - started) * 1000,
                exception=exc,
            )
