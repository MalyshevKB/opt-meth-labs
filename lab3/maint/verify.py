"""Независимая проверка плана ремонтов.

Получает только исходные таблицы (CSV) и таблицы решения: неделя начала
ремонта каждой единицы и вызов/часы подрядчика по неделям. Объект солвера и
матрица модели не используются; все ограничения, включая связку big-M,
пересчитываются по их предметному смыслу.
"""
import numpy as np
import pandas as pd

TOL = 1e-7


def check(units: pd.DataFrame, crew: pd.DataFrame, conflicts: pd.DataFrame, params: dict,
          schedule: pd.DataFrame, contractor: pd.DataFrame, M: np.ndarray = None,
          reported_obj: float = None, contractor_ok: np.ndarray = None) -> pd.DataFrame:
    T = params["weeks"]
    out = []

    def rec(kind, obj, lhs, sense, rhs):
        viol = {"<=": lhs - rhs, ">=": rhs - lhs, "==": abs(lhs - rhs)}[sense]
        out.append({"check": kind, "object": obj, "lhs": lhs, "sense": sense, "rhs": rhs,
                    "violation": max(0.0, viol), "ok": viol <= TOL * max(1.0, abs(rhs))})

    u = units.set_index("unit")
    start = schedule.set_index("unit").start_week
    down = pd.DataFrame(0, index=u.index, columns=range(1, T + 1))   # 1 — единица в ремонте
    for name, row in u.iterrows():
        s = start.get(name)
        if pd.isna(s):
            rec("обязательный ремонт выполнен", name, 0, ">=", float(row.mandatory))
            continue
        rec("неделя начала целая", name, abs(s - round(s)), "<=", 0.0)
        s = int(round(s)); e = s + int(row.duration_weeks) - 1
        rec("начало не раньше недели 1", name, s, ">=", 1)
        rec("окончание в горизонте", name, e, "<=", T)
        rec("окончание до срока предписания", name, e, "<=", float(row.deadline_week))
        down.loc[name, s:e] = 1

    own = crew.set_index("week").own_crew_hours
    ct = contractor.set_index("week")
    ok_c = np.ones(T) if contractor_ok is None else contractor_ok
    for t in range(1, T + 1):
        called, hours = float(ct.loc[t, "called"]), float(ct.loc[t, "hours"])
        load = float((down[t] * u.crew_hours_per_week).sum())
        rec("часы бригад", f"нед {t}", load, "<=", float(own[t]) + hours)
        rec("вызов подрядчика ∈ {0,1}", f"нед {t}", min(abs(called), abs(called - 1)), "<=", 0.0)
        rec("часы подрядчика ≥ 0", f"нед {t}", hours, ">=", 0.0)
        # смысл связки: часы подрядчика только если он вызван, и не больше его возможностей
        rec("подрядчик: часы ≤ C_max · вызов", f"нед {t}", hours, "<=",
            params["contractor_max_hours"] * round(called) * ok_c[t - 1])
        if M is not None:
            rec("big-M: часы ≤ M · вызов", f"нед {t}", hours, "<=", float(M[t - 1]) * called)
        rec("в ремонте одновременно ≤ K", f"нед {t}", float(down[t].sum()), "<=", params["max_units_down"])
        for _, cf in conflicts.iterrows():
            rec("единицы одной линии не одновременно", f"{cf.unit_a} + {cf.unit_b} | нед {t}",
                float(down.loc[cf.unit_a, t] + down.loc[cf.unit_b, t]), "<=", 1.0)

    if reported_obj is not None:
        H = params["post_horizon_weeks"]
        total = 0.0
        for name, row in u.iterrows():
            risk = row.failure_cost_rub * row.failure_prob_week1 * row.prob_growth_per_week ** np.arange(T + H)
            s = start.get(name)
            if pd.isna(s):
                total += risk.sum()                                  # работает весь горизонт и после него
            else:
                s = int(round(s))
                total += risk[:s - 1].sum() + row.repair_cost_rub + row.downtime_loss_rub_per_week * row.duration_weeks
        total += float((ct.called.round() * params["contractor_mobilization_rub"]
                        + ct.hours * params["contractor_rate_rub_per_hour"]).sum())
        out.append({"check": "целевая функция (пересчёт vs солвер)", "object": "₽", "lhs": total, "sense": "==",
                    "rhs": reported_obj, "violation": abs(total - reported_obj),
                    "ok": abs(total - reported_obj) <= 1e-6 * max(1.0, abs(total))})
    return pd.DataFrame(out)
