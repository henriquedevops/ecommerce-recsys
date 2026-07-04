"""Stage DVC ``feature_eng``: feedback implícito e split temporal por usuário."""

from __future__ import annotations

import json

import pandas as pd

from recsys.config import load_params, settings
from recsys.features.strategies import ImplicitEncoder

SplitFrames = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def leave_last_out(positives: pd.DataFrame) -> SplitFrames:
    """Split temporal por usuário (leave-one-out, padrão em RecSys).

    A interação positiva mais recente de cada usuário vai para ``test``, a
    segunda mais recente para ``val`` e o restante para ``train``. Usuários
    com menos de 3 positivos ficam integralmente no treino (não dá para
    avaliar quem quase não tem histórico).

    Args:
        positives: DataFrame de interações positivas com ``timestamp``.

    Returns:
        Tupla ``(train, val, test)``.
    """
    ordered = positives.sort_values(["user_idx", "timestamp"], kind="stable")
    from_end = ordered.groupby("user_idx").cumcount(ascending=False)
    size = ordered.groupby("user_idx")["item_idx"].transform("size")
    eligible = size >= 3
    is_test = eligible & (from_end == 0)
    is_val = eligible & (from_end == 1)
    return ordered[~is_test & ~is_val], ordered[is_val], ordered[is_test]


def build_features() -> dict[str, int]:
    """Aplica o ImplicitEncoder, faz o split e salva os parquets + meta.json.

    Returns:
        Metadados do dataset (``n_users``, ``n_items``) usados pelos embeddings.
    """
    params = load_params()["featurize"]
    clean = pd.read_parquet(settings.data_dir / "interim" / "clean.parquet")
    encoder = ImplicitEncoder(threshold=params["implicit_threshold"])
    train, val, test = leave_last_out(encoder.fit_transform(clean))

    out_dir = settings.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {"train": train, "val": val, "test": test}.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)

    meta = {
        "n_users": int(clean["user_idx"].max()) + 1,
        "n_items": int(clean["item_idx"].max()) + 1,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def main() -> None:
    """Entrypoint do stage: ``interim/clean.parquet`` -> ``processed/*``."""
    meta = build_features()
    out_dir = settings.data_dir / "processed"
    sizes = {
        name: len(pd.read_parquet(out_dir / f"{name}.parquet"))
        for name in ("train", "val", "test")
    }
    print(f"[OK] feature_eng: {sizes} | meta={meta}")


if __name__ == "__main__":
    main()
