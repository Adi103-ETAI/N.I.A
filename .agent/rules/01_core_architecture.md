# 🏗️ CORE ARCHITECTURE ENFORCEMENT

## 1. THE MONOLITH LAW

N.I.A. is a Monolith. Circular dependencies will crash the runtime.

- **FORBIDDEN:** Importing `tara` inside `nia`.
- **FORBIDDEN:** Importing `nia` inside `iris`.
- **MANDATORY:** Communicate via `State` logic or `ServiceContainer`.

---

## 2. STATE IMMUTABILITY (LangGraph)

You are working with a live `ChromaDB` checkpoint system.

- **RULE:** You CANNOT rename existing keys in `nia.graph.state.NIAState`.
- **EXTENSION:** If you need new data, ADD a new optional key (`total_tokens?: int`). DO NOT change types of existing keys.

---

## 3. CONFIGURATION LOCK

- **Never** hardcode magic numbers (e.g., `timeout=30`).
- **Always** define them in `.env` and expose them via `core.config.Settings`.
- **Access Pattern:** `settings.VOICE_TIMEOUT` (Good) vs `30` (Bad).
