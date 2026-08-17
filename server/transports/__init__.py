"""Wire transports: framing and socket handling, nothing else.

Both modules here read the same `AgentSession.Subscriber` and send the same
frames `server/wire.py` produced. The reviewer's check (§10.3 of the phase-8
plan) is whether deleting either file loses anything but that one route — if it
loses *behaviour*, session logic has leaked into the transport layer.
"""
