"""N.I.A. Soldier System — Ephemeral Task Agents for the Polyglot Swarm.

Soldiers are short-lived, single-purpose agents spun up by the Docker Swarm
when NIA routes a request to the ``"swarm"`` target.  Each soldier runs in its
own Docker container (or subprocess) and reports results back via the bridge.

Package Contents:

    builder_cache.py
        Caches compiled soldier Docker images to avoid repeated builds.

    schemas.py
        Pydantic schemas for soldier task payloads and result envelopes.

Lifecycle::

    NIA Decision (swarm) → Docker Bridge → Soldier Container
                                         ↓
                                    Execute task
                                         ↓
                                  Return result JSON
                                         ↓
                                NIA formatter node
"""
