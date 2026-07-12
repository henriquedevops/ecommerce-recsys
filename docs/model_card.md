# Model Card — ecommerce-recsys (RecSysMLP)

## 1. Detalhes do Modelo

- **Nome**: `ecommerce-recsys` (arquitetura `RecSysMLP`)
- **Tipo**: recomendador neural — embeddings de usuário/item + MLP (PyTorch)
- **Versão**: 1 (MLflow Model Registry, stage **Production**)
- **Formulação**: feedback implícito / ranking Top-N (rating ≥ 4 → interação positiva)
- **Hiperparâmetros** (vencedores do tuning com 3 runs MLflow): embedding 128,
  MLP 256→128→1, dropout 0.3, lr 0.001, batch 1024, early stopping por NDCG@10
  de validação (patience 3) — melhor época: 15
- **Dataset de treino**: MovieLens 1M (proxy de interações de e-commerce)
- **Seed fixa**: 42 (random, numpy, torch) — treino integralmente reprodutível

## 2. Uso Pretendido

- **Caso de uso**: recomendar Top-N itens a um usuário a partir do seu histórico
  de interações implícitas.
- **Usuários-alvo**: time de produto/e-commerce (recomendação em vitrine, e-mail).
- **Fora do escopo**: decisões críticas automatizadas sem revisão humana;
  usuários ou itens sem histórico (cold-start).

## 3. Performance

Protocolo: split **leave-one-out temporal** (última interação de cada usuário no
teste, penúltima na validação), ranking sobre o **catálogo inteiro** (3.706 itens)
com máscara de itens já vistos — mais rigoroso que amostrar 99 negativos, por isso
os valores absolutos são baixos em todos os modelos.

| Métrica | Popularidade | Item-KNN (cosseno) | **RecSysMLP (tuned)** |
|---|---|---|---|
| Precision@10 | 0.0041 | 0.0061 | 0.0056 |
| Recall@10 | 0.0411 | 0.0615 | 0.0557 |
| NDCG@10 | 0.0193 | 0.0308 | 0.0282 |
| MAP | 0.0204 | 0.0306 | 0.0295 |

Leitura honesta dos números:

- O tuning levou o MLP de NDCG@10 0.0182 para **0.0282 (+55%)** e o colocou muito
  à frente da popularidade (+46%).
- O **item-KNN ainda vence por margem estreita** no teste. Com ~1M interações e
  3.7k itens, métodos de vizinhança são baselines fortes; o MLP tende a ganhar
  vantagem com mais dados, features laterais e mais busca de hiperparâmetros.
- Fonte: `reports/comparison.json` (baselines) e `reports/metrics.json` (MLP),
  ambos gerados pelo pipeline (`dvc repro` + `python -m recsys.evaluation.compare`).

## 4. Dados de Treino

- 1.000.209 ratings → 6.040 usuários e 3.706 itens após limpeza/reindexação.
- Feedback implícito: rating ≥ 4 vira label 1 (563.211 positivos de treino);
  4 negativos amostrados por positivo, re-sorteados a cada época.
- Split leave-one-out temporal: 563.211 train / 6.035 val / 6.035 test.

## 5. Limitações e Vieses

- **Cold-start**: usuários/itens fora do treino não têm embedding → o modelo não
  pontua; exige fallback (ex.: popularidade) em produção.
- **Viés de popularidade**: itens muito interagidos dominam o treino e tendem a
  ser super-recomendados; a cauda longa fica sub-exposta.
- **Viés do dataset**: MovieLens reflete avaliação de filmes nos anos 2000, não
  comportamento real de compra em e-commerce.
- **Feedback implícito**: ausência de interação não significa desinteresse — os
  negativos amostrados contêm falsos negativos por construção.

## 6. Cenários de Falha

- Usuário com pouquíssimas interações → recomendações genéricas, próximas da
  popularidade.
- Drift de catálogo (entrada frequente de itens novos) → degradação progressiva;
  exige re-treino periódico.
- Mudança na distribuição de ratings (ex.: nova escala) quebraria o threshold
  implícito (rating ≥ 4).

## 7. Monitoramento e Ciclo de Vida

- **Métricas online sugeridas**: CTR das recomendações, cobertura de catálogo,
  diversidade das listas Top-N.
- **Re-treino**: periódico ou disparado por drift; o pipeline DVC re-executa tudo
  com um `dvc repro`.
- **Versionamento**: cada re-treino gera um run MLflow; o modelo aprovado vira
  nova versão no Registry e é promovido Staging → Production
  (`python -m recsys.models.register`).
