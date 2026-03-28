# Coordinator Checkpointing

## Overview
Coordinator now supports crash recovery via AsyncSqliteSaver.

## Usage
```python
# Default
result = await run_coordinator(manifest)

# Custom path
result = await run_coordinator(manifest, db_path="custom/path")
```

## Features
- Auto-persist to `data/checkpoints/coordinator.db`
- Thread isolation per mission_id
- Resume after crash
- Idempotent execution

## Configuration
Pass `db_path` parameter (default: "data/checkpoints")

## Testing
```bash
pytest tests/integration/test_coordinator_resume.py
pytest tests/integration/test_coordinator_idempotency.py
```
