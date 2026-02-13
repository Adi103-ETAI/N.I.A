---
description: Automated health check for N.I.A. v4.0.0 Release
---
# Release Verification Workflow

1.  **Environment Check**
    -   Verify we are in the correct virtual environment.
    -   Run `python --version` to confirm Python 3.10+.

2.  **Dependency Check**
    // turbo
    -   Run `pip install -r requirements.txt` to ensure all dependencies are installed.
    // turbo
    -   Run `playwright install` to ensure browser binaries are present.

3.  **Configuration Verification**
    // turbo
    -   Run `python scripts/verify_config.py` to check environment variables.
    // turbo
    -   Run `python scripts/verify_shadow_config.py` to check secure settings.

4.  **System Integrity**
    // turbo
    -   Run `python scripts/check_tara.py --check-only` to verify tool definitions.
    // turbo
    -   Run `python scripts/verify_hot_swap.py` to check plugin/LLM hot-swap capabilities.

5.  **Test Suite**
    // turbo
    -   Run `pytest` to execute the full test suite.
