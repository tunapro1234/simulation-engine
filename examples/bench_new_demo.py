"""Bench demo for the dataset tasks registered in the package: synthetic files -> bench.run -> leaderboard.

Offline, no data download, no LLM, runs in a few seconds:

    python examples/bench_new_demo.py

It writes tiny files in the Hillstrom and MovieLens-1M layouts to a temporary catalogue, then scores one own method
per task next to the built-in baselines. `m5` is listed on purpose without files: the leaderboard keeps it as one
`missing_data` row instead of raising. Point `data_root` (or $SIMENGINE_DATA) at a real catalogue to get the numbers
in docs/modules/bench/<task>.md; the synthetic effects below are planted, so the scores here mean nothing.
"""
import os
import tempfile

import numpy as np
import pandas as pd

from simengine import bench
from simengine.bench.tasks import uplift

GENRES = ["Action", "Comedy", "Drama", "Horror"]


def write_hillstrom(root: str, n: int = 4000, seed: int = 0) -> None:
    """E-mail experiment where the womens e-mail lifts visits only for customers who bought womens items."""
    rng = np.random.default_rng(seed)
    d = os.path.join(root, "hillstrom")
    os.makedirs(d)
    seg = rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], n)
    womens = rng.integers(0, 2, n)
    p = 0.10 + 0.15 * ((seg == "Womens E-Mail") & (womens == 1)) + 0.05 * (seg == "Mens E-Mail")
    pd.DataFrame({"recency": rng.integers(1, 13, n), "history_segment": rng.choice(["1) $0 - $100", "2) $100 - $200"], n),
                  "history": rng.gamma(2, 100, n).round(2), "mens": rng.integers(0, 2, n), "womens": womens,
                  "zip_code": rng.choice(["Urban", "Rural"], n), "newbie": rng.integers(0, 2, n),
                  "channel": rng.choice(["Web", "Phone"], n), "segment": seg,
                  "visit": (rng.random(n) < p).astype(int), "conversion": (rng.random(n) < p / 5).astype(int),
                  "spend": 0.0}).to_csv(os.path.join(d, uplift.HillstromUplift.FILE), index=False)


def write_movielens(root: str, n_users: int = 120, n_movies: int = 150, seed: int = 0) -> None:
    """Ratings = movie quality + user taste for the movie's genre + noise, in the ml-1m '::' format."""
    rng = np.random.default_rng(seed)
    d = os.path.join(root, "movielens-1m", "ml-1m")
    os.makedirs(d)
    q, g = rng.normal(0, .8, n_movies), rng.integers(0, len(GENRES), n_movies)
    taste = rng.normal(0, 1, (n_users, len(GENRES)))
    pop = np.exp(np.linspace(1.5, -1.5, n_movies))
    rows = []
    for u in range(1, n_users + 1):
        k = int(rng.integers(20, 40))
        mv = rng.choice(n_movies, k, replace=False, p=pop / pop.sum())
        ts = 970000000 + np.sort(rng.integers(0, 60, k)) * 60
        r = np.clip(np.round(3.4 + q[mv] + .6 * taste[u - 1, g[mv]] + rng.normal(0, .5, k)), 1, 5).astype(int)
        rows += [f"{u}::{m + 1}::{x}::{t}" for m, x, t in zip(mv, r, ts)]
    with open(os.path.join(d, "ratings.dat"), "w", encoding="latin-1") as f:
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(d, "users.dat"), "w", encoding="latin-1") as f:
        f.writelines(f"{u}::{'MF'[u % 2]}::25::{u % 21}::0{u:04d}\n" for u in range(1, n_users + 1))
    with open(os.path.join(d, "movies.dat"), "w", encoding="latin-1") as f:
        f.writelines(f"{m}::Synthetic Film {m:03d} ({1950 + m})::{GENRES[g[m - 1]]}\n" for m in range(1, n_movies + 1))


def womens_buyers(data) -> np.ndarray:
    """Own uplift score: target customers who bought womens items before (only the order matters)."""
    return data.test["womens"].to_numpy(float) + 1e-3 * data.test["recency"].to_numpy(float)


def shrunk_item_mean(data, k: float = 10.0) -> np.ndarray:
    """Own rating method: train movie mean shrunk towards the global mean by k pseudo-ratings."""
    tr, mu = data.train, data.train["rating"].mean()
    s = tr.groupby("movie_id")["rating"].agg(["sum", "count"])
    m = (s["sum"] + k * mu) / (s["count"] + k)
    return data.test["movie_id"].map(m).fillna(mu).to_numpy(float)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="simengine-bench-") as root:
        write_hillstrom(root)
        write_movielens(root)
        tasks = [("hillstrom_uplift", {"arm": "womens", "baselines": "random+t_logistic+class_transform"}),
                 ("movielens", {"target": "rating"}), "m5"]
        table = bench.run(tasks, methods={"hillstrom_uplift": {"womens_buyers": womens_buyers},
                                          "movielens": {"shrunk_item_mean": shrunk_item_mean}},
                          data_root=root, B=200)
        print(bench.to_markdown(table))
        st = table.drop_duplicates("task").set_index("task")["status"]
        print({t: st[t] for t in st.index})
        assert st["m5"] == "missing_data" and st["movielens"] == "ok" and st["hillstrom_uplift"] == "ok"
        print(len(bench.list()), "registered tasks:", ", ".join(bench.list()))
