# ---- builder: instala as dependências a partir do lock ----
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
# --frozen: exatamente o uv.lock | --no-dev: sem pytest/ruff | --no-install-project:
# só as deps ficam na .venv; o código entra por COPY no runtime (cache de build melhor)
RUN uv sync --frozen --no-dev --no-install-project

# ---- runtime: só o necessário para rodar o treino ----
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH="/app/src"
COPY src/ ./src/
COPY params.yaml ./
ENTRYPOINT ["python", "-m", "recsys.models.train"]

# ---- serve: API de inferência com o checkpoint embutido (deploy Cloud Run) ----
# Estágio final = default do `docker build`; o serviço de treino do compose
# usa `target: runtime`. O checkpoint vem do `dvc repro` local (fora do git).
FROM runtime AS serve
COPY models/recsys.pt ./models/
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT []
CMD ["sh", "-c", "uvicorn recsys.api:app --host 0.0.0.0 --port ${PORT}"]
