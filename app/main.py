"""BadDragon entrypoint (scaffold)."""

from __future__ import annotations

import argparse

from app.interfaces.cli import (
    run_hello_and_capture,
    run_terminal_chat,
    run_terminal_chat_with_debug,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hello",
        action="store_true",
        help="Send one hello message and save full sent/returned payloads into logs/",
    )
    parser.add_argument(
        "--debug-io",
        action="store_true",
        help="In terminal chat mode, print full payload sent to LLM and full returned payload.",
    )
    parser.add_argument(
        "--debug-lite",
        action="store_true",
        help="In terminal chat mode, print only concise per-turn LLM call counts.",
    )
    args = parser.parse_args()

    if args.hello:
        result = run_hello_and_capture()
        print("Hello request finished.")
        print(f"URL: {result['url']}")
        print(f"Sent payload: {result['sent_path']}")
        print(f"Returned payload: {result['recv_path']}")
        print(f"Returned text: {result['text_path']}")
        return

    if args.debug_io:
        run_terminal_chat_with_debug(debug_io=True, debug_lite=False)
        return

    if args.debug_lite:
        run_terminal_chat_with_debug(debug_io=False, debug_lite=True)
        return

    run_terminal_chat()


if __name__ == "__main__":
    main()
