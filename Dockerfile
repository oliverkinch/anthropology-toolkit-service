FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.11 /uv /uvx /bin/

ENV PATH="/root/.local/bin:/project/.venv/bin/:${PATH}"

WORKDIR /project

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-cache --no-install-project

COPY . .

EXPOSE 8080

CMD uv run serve
