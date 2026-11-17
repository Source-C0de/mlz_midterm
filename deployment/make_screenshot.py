"""Render evidence/screenshot.png: the deployed service answering live requests.

Draws a terminal-style transcript of the running container plus the JSON it
returned, so the deployed-service evidence is visible at a glance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (24, 26, 33)
FG = (222, 226, 235)
DIM = (140, 148, 162)
GREEN = (126, 211, 135)
RED = (240, 113, 120)
BLUE = (122, 162, 247)
YELLOW = (229, 192, 123)


def font(size: int, bold: bool = False):
    candidates = [
        ("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf"),
        ("C:/Windows/Fonts/segoeui.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout + out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<{exc}>"


def request(base_url: str, path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", default="evidence/screenshot.png")
    args = parser.parse_args()

    normal = {
        "type": "L",
        "air_temperature": 300.5,
        "process_temperature": 310.2,
        "rotational_speed": 1500,
        "torque": 40.1,
        "tool_wear": 12,
    }
    failure = {
        "type": "L",
        "air_temperature": 298.9,
        "process_temperature": 309.1,
        "rotational_speed": 2861,
        "torque": 4.6,
        "tool_wear": 143,
    }

    ps_status = run(["docker", "ps", "--filter", "name=quality-api", "--format",
                     "{{.Names}}  {{.Status}}  {{.Ports}}"])
    health = request(args.base_url, "/health")
    pred_ok = request(args.base_url, "/predict", normal)
    pred_fail = request(args.base_url, "/predict", failure)

    lines: list[tuple[str, tuple[int, int, int]]] = []
    lines.append(("$ docker ps --filter name=quality-api", DIM))
    lines.append((ps_status, GREEN))
    lines.append(("", FG))
    lines.append(("$ curl -s localhost:8080/health", DIM))
    lines.append((json.dumps(health[1], indent=2), BLUE))
    lines.append(("", FG))
    lines.append(("$ curl -X POST localhost:8080/predict   # normal, in-spec product", DIM))
    lines.append((json.dumps(pred_ok[1], indent=2), GREEN))
    lines.append(("", FG))
    lines.append(("$ curl -X POST localhost:8080/predict   # a real FAILURE row from the dataset", DIM))
    lines.append((json.dumps(pred_fail[1], indent=2), RED))

    body_font = font(15)
    title_font = font(20, bold=True)

    margin = 28
    title = "Laboratory Quality Prediction System - deployed service"
    subtitle = f"Containerised Flask API   |   {args.base_url}"

    widths = [margin * 2]
    heights = [margin * 2 + 76]
    for text, _ in lines:
        for line in text.split("\n") or [""]:
            widths.append(margin * 2 + int(body_font.getlength(line)))
            heights.append(heights[-1] + 21)

    width = max(widths)
    height = heights[-1] + margin

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((margin, margin), title, font=title_font, fill=FG)
    draw.text((margin, margin + 30), subtitle, font=body_font, fill=DIM)
    draw.line([(margin, margin + 60), (width - margin, margin + 60)], fill=(60, 64, 76))

    y = margin + 76
    for text, colour in lines:
        for line in text.split("\n") or [""]:
            draw.text((margin, y), line, font=body_font, fill=colour)
            y += 21

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(f"Saved {args.out} ({width}x{height})")


if __name__ == "__main__":
    main()
