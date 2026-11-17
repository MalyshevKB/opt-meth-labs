"""Независимая проверка решения сетевой задачи.

Работает только с исходными таблицами (CSV) и таблицами решения: потоки по
дугам и остатки по складам. Объект солвера и матрица модели не используются.
"""
import numpy as np
import pandas as pd

TOL = 1e-7


def check(warehouses: pd.DataFrame, routes: pd.DataFrame, demand: pd.DataFrame,
          deliveries: pd.DataFrame, flows: pd.DataFrame, stock: pd.DataFrame,
          reported_obj: float = None, closed: set = frozenset(), trucks: pd.DataFrame = None,
          truck_size: int = 33) -> pd.DataFrame:
    """flows: from, to, week, flow_pallets; stock: warehouse, week, stock_end
    (+ shortage, если есть). closed — множество (from, to, week) закрытых дуг."""
    W = list(warehouses.warehouse)
    T = int(demand.week.max())
    out = []

    def rec(kind, obj, lhs, sense, rhs):
        viol = {"<=": lhs - rhs, ">=": rhs - lhs, "==": abs(lhs - rhs)}[sense]
        out.append({"check": kind, "object": obj, "lhs": lhs, "sense": sense, "rhs": rhs,
                    "violation": max(0.0, viol), "ok": viol <= TOL * max(1.0, abs(rhs))})

    f = flows.set_index(["from", "to", "week"]).flow_pallets
    st = stock.set_index(["warehouse", "week"])
    dem = demand.set_index(["warehouse", "week"]).demand_pallets
    dlv = deliveries.groupby(["warehouse", "week"]).qty_pallets.sum()
    wh = warehouses.set_index("warehouse")

    # границы на дугах
    for _, r in routes.iterrows():
        for t in range(1, T + 1):
            key = (r["from"], r["to"], t)
            x = float(f.get(key, 0.0))
            cap = 0.0 if key in closed else float(r.cap_pallets_per_week)
            name = f"{r['from']}→{r['to']} | нед {t}"
            rec("дуга: поток ≥ 0", name, x, ">=", 0.0)
            rec("дуга: поток ≤ пропускная способность", name, x, "<=", cap)
            if trucks is not None:
                y = float(trucks.get(key, 0.0))
                rec("дуга: рейсов ≤ лимит", name, y, "<=", 0.0 if key in closed else float(r.max_trucks_per_week))
                rec("дуга: поток ≤ 33·рейсы", name, x, "<=", truck_size * y)
                rec("дуга: рейсы целые", name, abs(y - round(y)), "<=", 0.0)
    # потоки только по существующим дугам
    known = {(r["from"], r["to"]) for _, r in routes.iterrows()}
    for (a, b, t), x in f.items():
        if (a, b) not in known and abs(x) > TOL:
            rec("поток по несуществующей дуге", f"{a}→{b} | нед {t}", abs(x), "<=", 0.0)

    # баланс в каждой вершине (склад, неделя) и границы остатков
    for w in W:
        prev = float(wh.loc[w, "initial_stock"])
        for t in range(1, T + 1):
            inflow = float(f[[k for k in f.index if k[1] == w and k[2] == t]].sum())
            outflow = float(f[[k for k in f.index if k[0] == w and k[2] == t]].sum())
            I = float(st.loc[(w, t), "stock_end"])
            short = float(st.loc[(w, t), "shortage"]) if "shortage" in st.columns else 0.0
            lhs = prev + float(dlv.get((w, t), 0.0)) + inflow - outflow + short - I
            rec("баланс узла", f"{w} | нед {t}", lhs, "==", float(dem[(w, t)]))
            rec("остаток ≥ 0", f"{w} | нед {t}", I, ">=", 0.0)
            rec("остаток ≤ вместимость", f"{w} | нед {t}", I, "<=", float(wh.loc[w, "storage_cap"]))
            prev = I
        rec("страховой остаток в конце", w, prev, ">=", float(wh.loc[w, "safety_stock_end"]))

    if reported_obj is not None:
        rc = routes.set_index(["from", "to"])
        if trucks is None:
            transport = sum(x * rc.loc[(a, b), "cost_rub_per_pallet"] for (a, b, t), x in f.items())
        else:
            transport = sum(x * rc.loc[(a, b), "handling_rub_per_pallet"] for (a, b, t), x in f.items()) + \
                        sum(y * rc.loc[(a, b), "truck_cost_rub"] for (a, b, t), y in trucks.items())
        holding = sum(st.loc[(w, t), "stock_end"] * wh.loc[w, "holding_rub_per_pallet_week"]
                      for w in W for t in range(1, T + 1))
        obj = float(transport + holding)
        out.append({"check": "целевая функция (пересчёт vs солвер)", "object": "₽ за горизонт",
                    "lhs": obj, "sense": "==", "rhs": reported_obj, "violation": abs(obj - reported_obj),
                    "ok": abs(obj - reported_obj) <= 1e-6 * max(1.0, abs(obj))})
    return pd.DataFrame(out)
