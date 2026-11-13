"""Независимая проверка решения.

Работает только с исходными таблицами данных и таблицей решения (кг/гол/сут).
Не использует ни объект солвера, ни матрицу модели из model.py: каждое
ограничение пересчитывается заново по его содержательной формулировке.
"""
import numpy as np
import pandas as pd

TOL = 1e-7
NUTRIENTS = ["energy_MJ", "protein_g", "fat_g", "fiber_g", "calcium_g"]


def check(feeds: pd.DataFrame, animals: pd.DataFrame, norms: pd.DataFrame,
          compat: pd.DataFrame, solution: pd.DataFrame, reported_obj: float) -> pd.DataFrame:
    F, A = list(feeds.feed), list(animals.animal)
    X = (solution.pivot(index="animal", columns="feed", values="kg_per_head")
         .reindex(index=A, columns=F).fillna(0.0).to_numpy(float))     # A×F
    heads = animals.heads.to_numpy(float)
    nut = feeds[NUTRIENTS].to_numpy(float)                               # F×K
    is_anim = (feeds.category == "animal").to_numpy()
    out = []

    def rec(name, lhs, sense, rhs):
        viol = {"<=": lhs - rhs, ">=": rhs - lhs, "==": abs(lhs - rhs)}[sense]
        out.append({"constraint": name, "lhs": lhs, "sense": sense, "rhs": rhs,
                    "violation": max(0.0, viol), "ok": viol <= TOL * max(1.0, abs(rhs))})

    for a, an in enumerate(A):
        for f, fn in enumerate(F):
            rec(f"x>=0 [{an}|{fn}]", X[a, f], ">=", 0.0)
        allowed = compat[(compat.animal == an)].set_index("feed").loc[F, "allowed"].to_numpy()
        for f in np.where(allowed == 0)[0]:
            rec(f"запрещённый корм [{an}|{F[f]}]", X[a, f], "<=", 0.0)
        intake = nut.T @ X[a]                                            # K
        for k, kn in enumerate(NUTRIENTS):
            r = norms[(norms.animal == an) & (norms.nutrient == kn)].iloc[0]
            if pd.notna(r["min"]):
                rec(f"норма min [{an}|{kn}]", intake[k], ">=", float(r["min"]))
            if pd.notna(r["max"]):
                rec(f"норма max [{an}|{kn}]", intake[k], "<=", float(r["max"]))
        row = animals.iloc[a]
        total = X[a].sum()
        rec(f"объём рациона [{an}]", total, "<=", float(row.max_intake_kg))
        for f in np.where(allowed == 1)[0]:
            rec(f"разнообразие [{an}|{F[f]}]", X[a, f], "<=", float(row.max_share_single) * total)
        if allowed[is_anim].sum() > 0:
            anim = X[a, is_anim].sum()
            rec(f"доля живот. max [{an}]", anim, "<=", float(row.max_animal_share) * total)
            if row.min_animal_share > 0:
                rec(f"доля живот. min [{an}]", anim, ">=", float(row.min_animal_share) * total)
    used = heads @ X                                                     # F
    for f, fn in enumerate(F):
        rec(f"запас [{fn}]", used[f], "<=", float(feeds.supply_kg_per_day[f]))

    obj = float(heads @ X @ feeds.cost_rub_per_kg.to_numpy(float))
    out.append({"constraint": "целевая функция, ₽/сут (пересчёт vs солвер)", "lhs": obj,
                "sense": "==", "rhs": reported_obj, "violation": abs(obj - reported_obj),
                "ok": abs(obj - reported_obj) <= 1e-6 * max(1.0, abs(obj))})
    return pd.DataFrame(out)
