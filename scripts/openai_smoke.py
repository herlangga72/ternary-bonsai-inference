#!/usr/bin/env python3
"""Smoke-test the OpenAI-compatible bonsai-server with the standard library only.

    ./target/release/bonsai-server --model Ternary-Bonsai-27B-PQ2_0.gguf --port 8080 &
    python3 scripts/openai_smoke.py [base_url] [model] [prompt]

Exercises GET /v1/models, a non-streaming chat completion and a streaming one,
so it validates exactly the surface an agent framework uses.
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Ternary-Bonsai-27B-PQ2_0"
PROMPT = sys.argv[3] if len(sys.argv) > 3 else "What is the capital of France? Answer briefly."


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    # caller owns the response (streaming needs it open while iterating)
    return urllib.request.urlopen(req, timeout=600)


print("models:", [m["id"] for m in get("/v1/models")["data"]])

body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 24, "temperature": 0}
resp = post("/v1/chat/completions", body)
reply = json.loads(resp.read())
resp.close()
choice = reply["choices"][0]
print("non-stream:", repr(choice["message"]["content"]), choice["finish_reason"], reply["usage"])

print("stream: ", end="", flush=True)
streaming = dict(body, stream=True, max_tokens=16)
r = post("/v1/chat/completions", streaming)
try:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0]["delta"].get("content")
        if delta:
            print(delta, end="", flush=True)
finally:
    r.close()
print()
