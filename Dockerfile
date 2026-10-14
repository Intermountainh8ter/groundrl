# CPU-only image: the core library, the tests, the audit and the dry runs. No GPU stack.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY train ./train
COPY scripts ./scripts
COPY data ./data
COPY results ./results
COPY docs ./docs

RUN pip install --no-cache-dir -e ".[dev]"

# Default: verify the repository, then print the reward-hacking audit table.
CMD ["sh", "-c", "pytest -q && python scripts/reward_audit.py --split test --boot 200 --out /tmp/audit"]
