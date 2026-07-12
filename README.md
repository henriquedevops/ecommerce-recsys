# ecommerce-recsys

Sistema de recomendação de produtos para e-commerce — **Tech Challenge Fase 02** da pós em
Machine Learning Engineering (FIAP).

Rede neural de embeddings + MLP (PyTorch) treinada sobre o **MovieLens 1M** com formulação
**implícita/Top-N**, em um pipeline reprodutível de ponta a ponta: **uv** (dependências),
**DVC** (dados e pipeline), **MLflow** (tracking + Model Registry) e **Docker** multi-stage.

## Status

- [x] Etapa 1 — Clean Code e Estrutura (interfaces, Factory/Strategy, ruff, pre-commit)
- [x] Etapa 2 — Ambiente e Dependências (uv, `uv.lock`, `.env`, Pydantic Settings)
- [x] Etapa 3 — Containerização e Versionamento (Docker, DVC, MLflow tracking)
- [x] Etapa 4 — Rede Neural, Registry e Entrega (baselines, tuning, Registry, Model Card)

## Resultados

Protocolo: split leave-one-out temporal, ranking sobre o **catálogo inteiro** (3.706 itens)
com máscara de itens vistos — mais rigoroso que amostrar 99 negativos, por isso os valores
absolutos são baixos para todos os modelos.

| Modelo | Precision@10 | Recall@10 | NDCG@10 | MAP |
|---|---|---|---|---|
| Popularidade | 0.0041 | 0.0411 | 0.0193 | 0.0204 |
| Item-KNN (cosseno) | 0.0061 | 0.0615 | 0.0308 | 0.0306 |
| **RecSysMLP (tuned)** | 0.0056 | 0.0557 | 0.0282 | 0.0295 |

O tuning (3 runs MLflow) levou o MLP de NDCG@10 0.0182 → **0.0282 (+55%)**. O item-KNN
segue um baseline forte neste volume de dados — leitura completa em
[`docs/model_card.md`](docs/model_card.md).

## Estrutura

```text
src/recsys/
├── config.py        # paths e seed global (Pydantic Settings + .env)
├── data/            # carregamento e validação (MovieLens 1M)
├── features/        # encoders, splits e strategies de pré-processamento
├── models/          # interfaces, Factory, MLP, baselines, treino, tuning, registry
└── evaluation/      # métricas de ranking, avaliação e comparação de modelos
tests/               # testes automatizados (pytest)
scripts/             # utilitários (validação de ambiente etc.)
docs/                # Model Card
data/ · models/      # versionados por DVC (fora do git)
```

## Padrões de projeto

- **Factory** (`recsys.models.factory.ModelFactory`) — cria modelos por nome; novos modelos
  entram via `@ModelFactory.register("nome")` sem alterar código existente (Open/Closed).
  Catálogo atual: `mlp` (implícito via pipeline), `popularity`, `item_knn`.
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

## Reproduzindo os resultados

```bash
uv run dvc pull      # baixa dados/modelo do remote DVC (ou coloque o MovieLens 1M em data/raw/ml-1m/)
uv run dvc repro     # preprocess -> feature_eng -> train -> evaluate (seeds fixas)
```

Com o tracking apontado para um store com banco (o Registry exige; o file store puro não
suporta), todo o fluxo de experimentos fica disponível:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db     # PowerShell: $env:MLFLOW_TRACKING_URI='sqlite:///mlflow.db'

uv run dvc repro                                   # run "mlp_embedding" (train + test no mesmo run)
uv run python -m recsys.models.tune                # 3 runs de tuning variando hiperparâmetros
uv run python -m recsys.evaluation.compare         # baselines vs MLP -> reports/comparison.json
uv run python -m recsys.models.register            # registra e promove Staging -> Production
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # UI em http://127.0.0.1:5000
```

## Docker

```bash
docker build -t ecommerce-recsys .   # imagem multi-stage (~2 GB com torch CPU)
docker compose up mlflow -d          # MLflow server (backend sqlite, com Registry) em http://127.0.0.1:5000
docker compose up train              # treino no container, logando no server
docker compose down                  # encerra; runs persistem no volume mlflow-data
```

> Use `127.0.0.1` (não `localhost`) para acessar o server no Windows — `localhost` pode
> resolver para IPv6 e não conectar.

## Desenvolvimento

```bash
uv run ruff check src tests scripts   # lint (zero erros é requisito)
uv run pytest -q                      # testes
uv run pre-commit install             # hooks de lint/format a cada commit
```
