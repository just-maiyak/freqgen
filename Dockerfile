from ghcr.io/astral-sh/uv:python3.12-bookworm-slim

add . /freqgen
copy dotenv/prod.env /freqgen/.env

workdir /freqgen

run uv sync --frozen --no-cache

cmd ["uv", "run", "uvicorn", "freqgen.api:app", "--host", "0.0.0.0", "--port", "8080"]
