"""Генерация данных варианта 16 ЛР №3: план ремонтов оборудования по неделям.

Цех с n единицами оборудования (база — 10), горизонт 8 недель. Каждая единица
изнашивается: вероятность отказа растёт с каждой неделей без ремонта.
Ремонт занимает 1–2 недели, требует часов ремонтной бригады и останавливает
единицу. Собственная бригада ограничена, сверх неё можно вызвать подрядчика
с фиксированной платой за выход на неделю. Все величины синтетические.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

N, Q, Y = 15, 2, 26
VARIANT = 1 + ((17 * N + 7 * Q + Y) % 20)  # = 16
SEED = 10000 + 100 * Q + N                  # = 10215
WEEKS = 8

TYPES = ["Насос", "Компрессор", "Станок ЧПУ", "Пресс", "Конвейер", "Печь", "Котёл", "Экструдер"]


def generate_frames(n_units: int = 10, seed: int = SEED, weeks: int = WEEKS):
    rng = np.random.default_rng(seed)
    scale = n_units / 10
    types = rng.choice(TYPES, n_units)
    units = pd.DataFrame({
        "unit": [f"{t} №{i + 1}" for i, t in enumerate(types)],
        "duration_weeks": rng.choice([1, 2], n_units, p=[0.6, 0.4]),
        "crew_hours_per_week": rng.integers(6, 17, n_units) * 10,          # бригадо-часов в неделю ремонта
        "repair_cost_rub": (rng.uniform(50, 250, n_units) * 1000).round(-3),
        "downtime_loss_rub_per_week": (rng.uniform(40, 200, n_units) * 1000).round(-3),
        "failure_cost_rub": (rng.uniform(1.0, 5.0, n_units) * 1e6).round(-4),
        "failure_prob_week1": rng.uniform(0.005, 0.03, n_units).round(4),
        "prob_growth_per_week": rng.uniform(1.05, 1.25, n_units).round(3),
    })
    # обязательные ремонты (предписание надзора) со сроком
    mand = rng.choice(n_units, max(1, round(0.3 * n_units)), replace=False)
    units["mandatory"] = 0
    units["deadline_week"] = weeks
    units.loc[mand, "mandatory"] = 1
    units.loc[mand, "deadline_week"] = rng.integers(4, weeks + 1, len(mand))

    own = np.full(weeks, 200.0 * scale)
    own[[4, 5]] -= 80 * scale                                              # отпуска на 5–6 неделе
    crew = pd.DataFrame({"week": range(1, weeks + 1), "own_crew_hours": own.round()})

    # пары единиц на одной технологической линии — нельзя останавливать одновременно
    pairs, used = [], set()
    while len(pairs) < max(1, round(0.2 * n_units)):
        a, b = sorted(rng.choice(n_units, 2, replace=False))
        if (a, b) not in used:
            used.add((a, b)); pairs.append((a, b))
    conflicts = pd.DataFrame({"unit_a": [units.unit[a] for a, _ in pairs],
                              "unit_b": [units.unit[b] for _, b in pairs],
                              "reason": "одна технологическая линия"})
    params = {"weeks": weeks, "max_units_down": max(2, round(3 * scale)),
              "contractor_max_hours": round(200 * scale), "contractor_rate_rub_per_hour": 2500,
              "contractor_mobilization_rub": 150000, "post_horizon_weeks": 4}
    return units, crew, conflicts, params


def generate(out_dir: Path, seed: int = SEED) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    units, crew, conflicts, params = generate_frames(10, seed)
    units.to_csv(out_dir / "units.csv", index=False)
    crew.to_csv(out_dir / "crew.csv", index=False)
    conflicts.to_csv(out_dir / "conflicts.csv", index=False)
    meta = {"variant": VARIANT, "N": N, "Q": Q, "Y": Y, "seed": seed, **params,
            "units": {"time": "неделя", "crew": "бригадо-часы", "money": "₽"}}
    (out_dir / "params.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "data")
