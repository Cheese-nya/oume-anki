"""Command line for the local OUME study engine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from oume.persistence import JsonStore
from oume.session import StudyEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Study one OUME-Pragmatic queue without Anki.")
    parser.add_argument("--profile", default=str(Path.home() / ".local/share/oume/profile.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    enroll = sub.add_parser("enroll")
    enroll.add_argument("items", help="JSON list of {card_id, front, back, context}")
    sub.add_parser("next")
    answer = sub.add_parser("answer")
    answer.add_argument("--token", required=True)
    answer.add_argument("--correct", choices=("yes", "no"), required=True)
    answer.add_argument("--rt-ms", type=int, default=4000)
    answer.add_argument("--typed", default="")
    answer.add_argument("--choice", default="")
    answer.add_argument("--hint", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args()
    engine = StudyEngine(JsonStore(args.profile))
    if args.command == "enroll":
        print(json.dumps({"added": engine.enroll(json.loads(args.items))}, ensure_ascii=False))
    elif args.command == "next":
        print(json.dumps(engine.next_item(unix_sec=time.time()), ensure_ascii=False, indent=2))
    elif args.command == "answer":
        print(json.dumps(engine.submit(
            args.token, args.correct == "yes", args.rt_ms, args.typed, args.choice, args.hint
        ), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(engine.status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
