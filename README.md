# ecommerce-recsys

Sistema de recomendação de produtos para e-commerce — **Tech Challenge Fase 02** da pós em
Machine Learning Engineering (FIAP).

Rede neural de embeddings + MLP (PyTorch) treinada sobre o **MovieLens 1M** com formulação
**implícita/Top-N**, em um pipeline reprodutível de ponta a ponta: **uv** (dependências),
**DVC** (dados e pipeline), **MLflow** (tracking + Model Registry) e **Docker** multi-stage.

## Status

- [x] Etapa 1 — Clean Code e Estrutura (interfaces, Factory/Strategy, ruff, pre-commit)
- [x] Etapa 2 — Ambiente e Dependências (uv, `uv.lock`, `.env`, Pydantic Settings)
- [ ] Etapa 3 — Containerização e Versionamento (Docker, DVC, MLflow tracking)
- [ ] Etapa 4 — Rede Neural, Registry e Entrega

## Estrutura

```text
src/recsys/
├── config.py        # paths e seed global
├── data/            # carregamento e validação (MovieLens 1M)
├── features/        # encoders, splits e strategies de pré-processamento
├── models/          # interfaces, Factory, arquiteturas e treino
└── evaluation/      # métricas de ranking (Precision@K, Recall@K, NDCG, MAP)
tests/               # testes automatizados (pytest)
scripts/             # utilitários (validação de ambiente etc.)
data/ · models/      # versionados por DVC (fora do git)
```

## Padrões de projeto

- **Factory** (`recsys.models.factory.ModelFactory`) — cria modelos por nome; novos modelos
  entram via `@ModelFactory.register("nome")` sem alterar código existente (Open/Closed).
- **Strategy** (`recsys.features.strategies.Preprocessor`) — pré-processadores intercambiáveis;
  `ImplicitEncoder` converte ratings 1–5 em feedback implícito binário (positivo se ≥ 4).

## Instalação do zero

Pré-requisito único: [uv](https://docs.astral.sh/uv/) instalado
(`irm https://astral.sh/uv/install.ps1 | iex` no Windows, `curl -LsSf https://astral.sh/uv/install.sh | sh` no Linux/macOS).

```bash
git clone https://github.com/henriquedevops/ecommerce-recsys.git
cd ecommerce-recsys
uv sync --frozen                          # cria .venv e instala EXATAMENTE o uv.lock
uv run python scripts/validate_env.py     # deve terminar com "[OK] Ambiente validado."
uv run pytest -q                          # roda a suíte de testes
```

O `uv sync` baixa o Python 3.11 automaticamente se necessário (`.python-version`) e instala o
**torch CPU-only** via índice dedicado (`[tool.uv.index]` no `pyproject.toml`) — sem os ~2 GB de
wheels CUDA. `--frozen` garante que a instalação usa exatamente as versões travadas no `uv.lock`.

### Configuração (`.env`)

Todas as variáveis têm defaults funcionais em `src/recsys/config.py` (Pydantic Settings).
Para sobrescrever, copie o exemplo e edite:

```bash
cp .env.example .env
```

## Desenvolvimento

```bash
uv run ruff check src tests scripts   # lint (zero erros é requisito)
uv run pytest -q                      # testes
uv run pre-commit install             # hooks de lint/format a cada commit
```
