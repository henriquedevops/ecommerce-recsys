"""Stage DVC ``preprocess``: limpa e reindexa as interações do MovieLens 1M."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys.config import settings

RAW_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def load_raw_ratings(path: Path) -> pd.DataFrame:
    """Lê o ``ratings.dat`` original (separador ``::``, sem header).

    Args:
        path: Caminho do arquivo bruto do MovieLens 1M.

    Returns:
        DataFrame com colunas ``user_id``, ``item_id``, ``rating``, ``timestamp``.
    """
    return pd.read_csv(
        path, sep="::", engine="python", names=RAW_COLUMNS, encoding="latin-1"
    )


def reindex_contiguous(values: pd.Series) -> pd.Series:
    """Mapeia IDs arbitrários para índices contíguos ``0..N-1``.

    Necessário porque ``nn.Embedding`` exige índices densos começando em zero
    (o MovieLens tem buracos na numeração de filmes).

    Args:
        values: Série com os IDs originais (esparsos).

    Returns:
        Série de índices contíguos, alinhada com a entrada.
    """
    codes, _ = pd.factorize(values, sort=True)
    return pd.Series(codes, index=values.index)


def preprocess(raw_path: Path, out_path: Path) -> pd.DataFrame:
    """Remove duplicatas user-item, reindexa e salva o parquet limpo.

    Args:
        raw_path: Caminho do ``ratings.dat`` bruto.
        out_path: Destino do parquet limpo (``data/interim/clean.parquet``).

    Returns:
        DataFrame limpo com ``user_idx``, ``item_idx``, ``rating``, ``timestamp``.
    """
    frame = load_raw_ratings(raw_path)
    frame = frame.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    frame["user_idx"] = reindex_contiguous(frame["user_id"])
    frame["item_idx"] = reindex_contiguous(frame["item_id"])
    clean = frame[["user_idx", "item_idx", "rating", "timestamp"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path, index=False)
    return clean


def main() -> None:
    """Entrypoint do stage: ``data/raw`` -> ``data/interim/clean.parquet``."""
    raw = settings.data_dir / "raw" / "ml-1m" / "ratings.dat"
    out = settings.data_dir / "interim" / "clean.parquet"
    clean = preprocess(raw, out)
    n_users = clean["user_idx"].nunique()
    n_items = clean["item_idx"].nunique()
    print(
        f"[OK] preprocess: {len(clean):,} interacoes, "
        f"{n_users:,} usuarios, {n_items:,} itens -> {out}"
    )


if __name__ == "__main__":
    main()
