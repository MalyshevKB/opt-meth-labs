"""Генерация данных варианта 16 ЛР №2: балансировка запасов между складами.

Сеть: 7 складов по кругу + центральный склад. Дороги — кольцо между соседними
складами и «спицы» к центру (две случайные спицы отсутствуют). Все величины
синтетические, генератор с фиксированным seed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

N, Q, Y = 15, 2, 26
VARIANT = 1 + ((17 * N + 7 * Q + Y) % 20)  # = 16
SEED = 10000 + 100 * Q + N                  # = 10215

WEEKS = 4
PALLETS_PER_TRUCK = 33
NAMES = ["Север", "Северо-Восток", "Восток", "Юго-Восток", "Юг", "Запад", "Северо-Запад", "Центр"]
WEEK_SEASON = [1.00, 1.10, 0.95, 1.15]   # сезонность спроса по неделям


def generate(out_dir: Path, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_outer = len(NAMES) - 1
    center = n_outer

    # координаты, км
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n_outer, endpoint=False)
    rad = 150 * rng.uniform(0.8, 1.2, n_outer)
    xy = np.vstack([np.c_[rad * np.cos(ang), rad * np.sin(ang)], [0.0, 0.0]])
    xy[:n_outer] += rng.normal(0, 15, (n_outer, 2))

    # дороги: кольцо + спицы (без двух случайных)
    edges = [(i, (i + 1) % n_outer) for i in range(n_outer)]
    missing = set(rng.choice(n_outer, 2, replace=False))
    edges += [(center, i) for i in range(n_outer) if i not in missing]

    routes = []
    for i, j in edges:
        dist = float(np.hypot(*(xy[i] - xy[j])) * rng.uniform(1.15, 1.35))  # извилистость дорог
        truck_cost = round(2500 + 45 * dist, -1)              # ₽ за рейс фуры
        handling = round(rng.uniform(50, 80), 0)               # ₽ за паллету (погрузка/выгрузка)
        max_trucks = int(rng.integers(3, 7))                   # рейсов в неделю
        for a, b in ((i, j), (j, i)):
            routes.append({"from": NAMES[a], "to": NAMES[b], "dist_km": round(dist, 1),
                           "truck_cost_rub": truck_cost, "handling_rub_per_pallet": handling,
                           "max_trucks_per_week": max_trucks,
                           "cap_pallets_per_week": max_trucks * PALLETS_PER_TRUCK,
                           "cost_rub_per_pallet": round(handling + truck_cost / PALLETS_PER_TRUCK, 2)})
    pd.DataFrame(routes).to_csv(out_dir / "routes.csv", index=False)

    # спрос, паллет/неделю
    base = rng.uniform(60, 160, len(NAMES)).round()
    demand = [{"warehouse": NAMES[w], "week": t + 1,
               "demand_pallets": int(round(base[w] * WEEK_SEASON[t] * rng.uniform(0.9, 1.1)))}
              for w in range(len(NAMES)) for t in range(WEEKS)]
    dem = pd.DataFrame(demand)
    dem.to_csv(out_dir / "demand.csv", index=False)
    D = dem.pivot(index="warehouse", columns="week", values="demand_pallets").loc[NAMES].to_numpy()

    safety = np.round(0.5 * D[:, -1])
    need = D.sum() + safety.sum()
    total_supply = need * 1.08            # запас прочности сети по объёму — 8%

    # поставки от производителя: в Центр на неделях 1 и 3, на Восток — на неделе 2
    deliveries = [("Центр", 1, 0.22), ("Восток", 2, 0.18), ("Центр", 3, 0.22)]
    deliv = [{"warehouse": w, "week": t, "qty_pallets": int(round(total_supply * s))} for w, t, s in deliveries]
    pd.DataFrame(deliv).to_csv(out_dir / "deliveries.csv", index=False)

    # начальные остатки распределены неравномерно — отсюда потребность в балансировке
    init_total = int(round(total_supply)) - sum(d["qty_pallets"] for d in deliv)
    share = rng.dirichlet(np.full(len(NAMES), 0.9))
    init = np.floor(init_total * share)
    init[np.argmax(init)] += init_total - init.sum()
    cap = np.maximum(np.round(1.3 * D.max(1) + 0.2 * init.max()), init + 20)
    cap[center] = max(cap[center], 1.05 * deliv[0]["qty_pallets"] + init[center])
    cap[NAMES.index("Восток")] = max(cap[NAMES.index("Восток")], init[NAMES.index("Восток")] + deliv[1]["qty_pallets"])

    wh = pd.DataFrame({"warehouse": NAMES, "x_km": xy[:, 0].round(1), "y_km": xy[:, 1].round(1),
                       "initial_stock": init.astype(int), "storage_cap": cap.astype(int),
                       "holding_rub_per_pallet_week": rng.uniform(25, 45, len(NAMES)).round(1),
                       "safety_stock_end": safety.astype(int)})
    wh.to_csv(out_dir / "warehouses.csv", index=False)

    meta = {"variant": VARIANT, "N": N, "Q": Q, "Y": Y, "seed": seed, "weeks": WEEKS,
            "pallets_per_truck": PALLETS_PER_TRUCK,
            "units": {"flow": "паллет за неделю", "cost": "₽", "distance": "км"}}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "data")
