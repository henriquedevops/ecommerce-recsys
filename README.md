# ecommerce-recsys

Sistema de recomendação de produtos para e-commerce — **Tech Challenge Fase 02** da pós em
Machine Learning Engineering (FIAP).

Rede neural de embeddings + MLP (PyTorch) treinada sobre o **MovieLens 1M** com formulação
**implícita/Top-N**, em um pipeline reprodutível de ponta a ponta: **uv** (dependências),
**DVC** (dados e pipeline), **MLflow** (tracking + Model Registry) e **Docker** multi-stage.

## Status

- [x] Etapa 1 — Clean Code e Estrutura (interfaces, Factory/Strategy, ruff, pre-commit)
- [ ] Etapa 2 — Ambiente e Dependências (uv, `uv.lock`, `.env`, Pydantic Settings)
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

## Desenvolvimento

```bash
uvx ruff check src tests   # lint (zero erros é requisito)
uvx pytest -q              # testes
pre-commit install         # hooks de lint/format a cada commit
```

> A instalação completa do ambiente (`uv sync` + `uv.lock`) chega na Etapa 2.
