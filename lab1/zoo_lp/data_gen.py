"""Генерация данных варианта 16 (рацион кормления в зоопарке).

Опорные значения взяты из справочных таблиц питательности кормов (порядок
величин), затем каждое число случайно возмущается генератором с фиксированным
seed. Так данные уникальны для варианта, но остаются правдоподобными.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

N, Q, Y = 15, 2, 26
VARIANT = 1 + ((17 * N + 7 * Q + Y) % 20)  # = 16
SEED = 10000 + 100 * Q + N                  # = 10215

NUTRIENTS = ["energy_MJ", "protein_g", "fat_g", "fiber_g", "calcium_g"]

# корм, категория, цена ₽/кг, запас кг/сут, затем содержание в 1 кг корма:
# обменная энергия МДж, протеин г, жир г, клетчатка г, кальций г
FEEDS_REF = [
    ("Сено луговое",     "plant",  12, 60, 7.5,  90,  20, 280, 6.0),
    ("Люцерна (сено)",   "plant",  20, 20, 8.5, 170,  25, 250, 14.0),
    ("Морковь",          "plant",  35, 30, 1.6,  10,   2,  28, 0.35),
    ("Яблоки",           "plant",  60, 20, 2.2,   3,   2,  20, 0.06),
    ("Бананы",           "plant", 110, 10, 3.7,  11,   3,  26, 0.05),
    ("Овёс",             "plant",  25, 20, 11.0, 110, 45, 100, 1.0),
    ("Комбикорм",        "plant",  45, 15, 11.5, 160, 40,  90, 9.0),
    ("Рыба (мойва)",     "animal", 150, 12, 6.0, 140, 80,   0, 4.0),
    ("Говядина",         "animal", 400,  6, 8.0, 190, 120,  0, 0.1),
    ("Творог",           "animal", 250,  4, 5.5, 165, 50,   0, 1.6),
]

# вид, голов, макс. потребление кг/гол/сут, макс. доля одного корма,
# мин. и макс. доля кормов животного происхождения (по массе)
ANIMALS_REF = [
    ("Зебра",          8, 12.0, 0.60, 0.0, 0.0),
    ("Шимпанзе",       5,  5.0, 0.35, 0.0, 0.15),
    ("Бурый медведь",  3, 15.0, 0.40, 0.30, 0.70),
]

# нормы на 1 голову в сутки: (min, max); None = нет ограничения
NORMS_REF = {
    "Зебра":         {"energy_MJ": (50, 75), "protein_g": (500, None), "fat_g": (None, 300),
                      "fiber_g": (2000, None), "calcium_g": (25, None)},
    "Шимпанзе":      {"energy_MJ": (8, 14), "protein_g": (120, None), "fat_g": (None, 120),
                      "fiber_g": (110, None), "calcium_g": (4, None)},
    "Бурый медведь": {"energy_MJ": (40, 70), "protein_g": (800, None), "fat_g": (None, 1500),
                      "fiber_g": (250, None), "calcium_g": (12, None)},
}

# какие корма допустимы для вида (ветеринарная совместимость)
FORBIDDEN = {
    "Зебра": {"Рыба (мойва)", "Говядина", "Творог", "Бананы"},
    "Шимпанзе": {"Сено луговое", "Люцерна (сено)", "Говядина"},
    "Бурый медведь": {"Сено луговое", "Люцерна (сено)"},
}


def generate(out_dir: Path, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, cat, cost, supply, *nut in FEEDS_REF:
        rows.append({
            "feed": name, "category": cat,
            "cost_rub_per_kg": round(cost * rng.uniform(0.9, 1.1), 2),
            "supply_kg_per_day": round(supply * rng.uniform(0.9, 1.1), 2),
            **{k: round(v * rng.uniform(0.95, 1.05), 3) for k, v in zip(NUTRIENTS, nut)},
        })
    pd.DataFrame(rows).to_csv(out_dir / "feeds.csv", index=False)

    pd.DataFrame(ANIMALS_REF, columns=[
        "animal", "heads", "max_intake_kg", "max_share_single",
        "min_animal_share", "max_animal_share",
    ]).to_csv(out_dir / "animals.csv", index=False)

    norms = []
    for animal, d in NORMS_REF.items():
        for k, (lo, hi) in d.items():
            noise = rng.uniform(0.95, 1.05)
            norms.append({"animal": animal, "nutrient": k,
                          "min": round(lo * noise, 3) if lo is not None else "",
                          "max": round(hi * noise, 3) if hi is not None else ""})
    pd.DataFrame(norms).to_csv(out_dir / "norms.csv", index=False)

    compat = [{"animal": a, "feed": f[0], "allowed": int(f[0] not in FORBIDDEN[a])}
              for a, *_ in ANIMALS_REF for f in FEEDS_REF]
    pd.DataFrame(compat).to_csv(out_dir / "compat.csv", index=False)

    meta = {"variant": VARIANT, "N": N, "Q": Q, "Y": Y, "seed": seed,
            "units": {"x": "кг корма на 1 голову в сутки", "objective": "₽ в сутки",
                      "energy_MJ": "МДж обменной энергии", "protein_g": "г",
                      "fat_g": "г", "fiber_g": "г", "calcium_g": "г"}}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "data")
