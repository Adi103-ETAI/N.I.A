---
description: Bootstrapping a new development environment for N.I.A.
---
# Development Setup Workflow

1.  **Environment Preparation**
    -   Check if `.venv` exists. If not, create it: `python -m venv .venv`.
    -   *Note: You must activate the virtual environment manually in your terminal before proceeding.*
        -   Windows: `.venv\Scripts\activate`
        -   Linux/Mac: `source .venv/bin/activate`

2.  **Configuration Check**
    -   Check if `.env` exists. If not, copy from example:
        -   Windows: `copy env.example .env`
        -   Linux/Mac: `cp env.example .env`
    -   *Action Required: Open `.env` and fill in your API keys (NVIDIA_API_KEY, etc).*

3.  **Dependency Installation**
    // turbo
    -   Run `pip install --upgrade pip` to ensure pip is current.
    // turbo
    -   Run `pip install -r requirements.txt` to install project dependencies.
    // turbo
    -   Run `playwright install` to download browser binaries.

4.  **Verification**
    -   Run `python scripts/verify_config.py` to validate your `.env` configuration.
    -   Run `python scripts/check_tara.py --check-only` to ensure core systems are operational.

5.  **Ready to Code**
    -   Start the assistant: `python main.py`
