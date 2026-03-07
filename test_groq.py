#!/usr/bin/env python3
"""
Test Groq API key and LLM post-processing without audio hardware.
Sends a hardcoded messy transcript through the post-processor and prints the result.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from freeflow_linux import load_config, post_process, POST_PROCESSING_SYSTEM_PROMPT
from groq import Groq

def main():
    cfg = load_config()
    api_key = cfg.get("api_key", "").strip()

    if not api_key:
        print("ERROR: No Groq API key found. Set api_key in config or GROQ_API_KEY env var.")
        sys.exit(1)

    print(f"Config loaded. API key: {api_key[:8]}...")

    groq_kwargs = {"api_key": api_key}
    if cfg.get("api_base_url"):
        groq_kwargs["base_url"] = cfg["api_base_url"]
    client = Groq(**groq_kwargs)

    # Test 1: basic cleanup
    raw1 = "um so uh I wanted to um say that the the meeting is uh scheduled for tuesday"
    context1 = "Active window: Google Calendar (Chrome)"
    print(f"\nTest 1 - filler word removal:")
    print(f"  Input:   {raw1!r}")
    result1 = post_process(client, raw1, context1)
    print(f"  Output:  {result1!r}")
    assert result1, "Expected non-empty result"
    assert "um" not in result1.lower() or "um" in raw1, "Should remove filler words"

    # Test 2: EMPTY handling
    raw2 = ""
    print(f"\nTest 2 - empty transcript:")
    print(f"  Input:   {raw2!r}")
    result2 = post_process(client, raw2, "")
    print(f"  Output:  {result2!r}")

    # Test 3: code/technical context
    raw3 = "create a new function called get user by eye dee that takes a string argument"
    context3 = "Active window: freeflow_linux.py - VSCode"
    print(f"\nTest 3 - technical context:")
    print(f"  Input:   {raw3!r}")
    result3 = post_process(client, raw3, context3)
    print(f"  Output:  {result3!r}")
    assert result3, "Expected non-empty result"

    print("\nAll tests passed. Groq API key is valid and post-processing works.")

if __name__ == "__main__":
    main()
