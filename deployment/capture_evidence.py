"""Capture API evidence for the repository.

Hits the running service and writes a human-readable transcript to
``evidence/predict-curl.txt``. Usage:

    python deployment/capture_evidence.py --base-url http://127.0.0.1:8010
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone


def call(base_url: str, path: str, payload: dict | None = None) -> str:
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.dumps(json.loads(response.read()), indent=2, sort_keys=True)
            return f"HTTP {response.status}\n{body}"
    except urllib.error.HTTPError as error:
        body = json.dumps(json.loads(error.read()), indent=2, sort_keys=True)
        return f"HTTP {error.code}\n{body}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--out", default="evidence/predict-curl.txt")
    args = parser.parse_args()

    normal = {
        "type": "L",
        "air_temperature": 300.5,
        "process_temperature": 310.2,
        "rotational_speed": 1500,
        "torque": 40.1,
        "tool_wear": 12,
    }
    # A row that genuinely failed in the source dataset (Machine failure == 1).
    real_failure = {
        "type": "L",
        "air_temperature": 298.9,
        "process_temperature": 309.1,
        "rotational_speed": 2861,
        "torque": 4.6,
        "tool_wear": 143,
    }

    sections = [
        ("GET /health", call(args.base_url, "/health")),
        ("POST /predict  (a normal, in-spec product)", call(args.base_url, "/predict", normal)),
        (
            "POST /predict  (a real FAILURE row from the dataset)",
            call(args.base_url, "/predict", real_failure),
        ),
        (
            "POST /predict/batch",
            call(args.base_url, "/predict/batch", {"products": [normal, real_failure]}),
        ),
        (
            "POST /predict  (missing field -> 400)",
            call(args.base_url, "/predict", {"type": "L", "air_temperature": 300.5}),
        ),
        (
            "POST /predict  (non-numeric field -> 400)",
            call(
                args.base_url,
                "/predict",
                {**normal, "air_temperature": "hot"},
            ),
        ),
    ]

    lines = [
        "=== Laboratory Quality Prediction System - API evidence ===",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Base URL: {args.base_url}",
        "",
    ]
    for title, body in sections:
        lines.append(f"--- {title}")
        lines.append(body)
        lines.append("")

    text = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
