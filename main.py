import asyncio
import logging
import sys

from voice.session import VoiceSession


def main():
    # Configure logging – set to DEBUG for verbose output
    level = logging.DEBUG if "--debug" in sys.argv else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    user_id = "augusto"
    session = VoiceSession(user_id=user_id)

    try:
        asyncio.run(session.run())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
