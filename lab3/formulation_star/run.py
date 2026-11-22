"""Сравнение исходной MILP и формулировки с разбиением недельной нагрузки."""
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from maint.data_gen import generate_frames
from maint.model import build_model, contractor_table, from_frames, schedule_table, solve
from maint.verify import check

RESULTS = Path(__file__).resolve().parent / "results"
OPTIONS = {"threads": 1, "random_seed": 0, "time_limit": 60.0, "mip_rel_gap": 0.0}
TOL = 1e-6


def split_model(data, base):
    """u[i,t] — часть индикатора простоя, отнесённая к режиму с подрядчиком."""
    keys = [(i, t) for t in range(data.T) for i in range(len(data.units))]
    n = len(base.c)
    rows, cols, values, upper, names = [], [], [], [], []
    active = {}

    def add(entries, rhs, name):
        for j, value in entries:
            rows.append(len(upper))
            cols.append(j)
            values.append(value)
        upper.append(rhs)
        names.append(name)

    for j, (i, t) in enumerate(keys, n):
        starts = [k for k, (unit, start) in enumerate(base.z_keys)
                  if unit == i and start <= t < start + data.p[i] and base.col_hi[k] > 0]
        active[i, t] = (j, starts)
        w = base.w0 + t
        label = f"{i + 1},{t + 1}"
        add([(j, 1), (w, -1)], 0, f"on_indicator[{label}]")
        add([(j, 1)] + [(k, -1) for k in starts], 0, f"off_nonnegative[{label}]")
        add([(j, -1), (w, 1)] + [(k, 1) for k in starts], 1, f"off_indicator[{label}]")

    for t in range(data.T):
        w, o = base.w0 + t, base.o0 + t
        for weights, limit, group, extra in [
            (data.h, data.own[t], "crew", [(o, -1)]),
            (np.ones(len(data.units)), data.K, "down", []),
        ]:
            on = [(active[i, t][0], weight) for i, weight in enumerate(weights)]
            off = [(k, weight) for i, weight in enumerate(weights) for k in active[i, t][1]]
            add(on + [(w, -limit)] + extra, 0, f"split_on_{group}[{t + 1}]")
            add(off + [(j, -weight) for j, weight in on] + [(w, limit)],
                limit, f"split_off_{group}[{t + 1}]")

        # Покрытия для режима без подрядчика. Малый экземпляр позволяет перебор,
        # на больших достаточно проверять префиксы по убыванию трудоёмкости.
        eligible = [i for i in range(len(data.units)) if active[i, t][1]]
        if len(data.units) <= 10:
            candidates = (group for k in range(1, data.K + 1)
                          for group in combinations(eligible, k))
        else:
            ordered = sorted(eligible, key=lambda i: -data.h[i])
            candidates = (ordered[:k] for k in range(1, min(data.K, len(ordered)) + 1))
        for group in candidates:
            hours = sum(data.h[i] for i in group)
            if hours <= data.own[t] or hours - min(data.h[i] for i in group) > data.own[t]:
                continue
            entries = [(w, len(group) - 1)]
            for i in group:
                j, starts = active[i, t]
                entries.extend([(j, -1)] + [(k, 1) for k in starts])
            add(entries, len(group) - 1, f"off_cover[{t + 1},{len(upper)}]")

    extra = sp.csc_matrix((values, (rows, cols)), shape=(len(upper), n + len(keys)))
    original = sp.hstack([base.A, sp.csc_matrix((base.A.shape[0], len(keys)))])
    model = replace(
        base, A=sp.vstack([original, extra], format="csc"),
        c=np.r_[base.c, np.zeros(len(keys))],
        col_lo=np.r_[base.col_lo, np.zeros(len(keys))],
        col_hi=np.r_[base.col_hi, np.ones(len(keys))],
        integrality=np.r_[base.integrality, np.zeros(len(keys), dtype=int)],
        row_lo=np.r_[base.row_lo, np.full(len(upper), -np.inf)],
        row_hi=np.r_[base.row_hi, upper], row_names=base.row_names + names,
        col_names=base.col_names + [f"u[{i + 1},{t + 1}]" for i, t in keys],
    )
    return model, active


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for size in (10, 20, 40, 80):
        frames = generate_frames(size)
        data = from_frames(*frames)
        base = build_model(data)
        strengthened, active = split_model(data, base)
        pair = {}
        for label, model in (("base", base), ("split", strengthened)):
            lp = solve(model, relax=True, log_file=RESULTS / f"{size}_{label}_lp.log",
                       options=OPTIONS)
            mip = solve(model, log_file=RESULTS / f"{size}_{label}_mip.log", options=OPTIONS)
            if lp.status != "Optimal" or mip.x is None:
                raise RuntimeError(f"{size}/{label}: LP {lp.status}, MILP {mip.status}")
            checks = check(*frames, schedule_table(data, base, mip.x),
                           contractor_table(data, base, mip.x), base.M, mip.objective)
            if not checks.ok.all():
                raise RuntimeError(f"Недопустимый план: {size}/{label}")
            pair[label] = (lp, mip)
            rows.append({
                "units": size, "formulation": label, "variables": len(model.c),
                "rows": model.A.shape[0], "lp_rub": lp.objective, "lp_seconds": lp.time_s,
                "mip_rub": mip.objective, "mip_bound_rub": mip.info["mip_dual_bound"],
                "lp_gap_pct": 100 * (mip.objective - lp.objective) / abs(mip.objective),
                "mip_gap_pct": 100 * mip.info["mip_gap"], "nodes": mip.info["mip_nodes"],
                "mip_seconds": mip.time_s, "mip_status": mip.status, "checks_passed": len(checks),
            })

        base_lp, base_mip = pair["base"]
        split_lp, split_mip = pair["split"]
        if split_lp.objective < base_lp.objective - TOL:
            raise RuntimeError(f"Усиленная LP дала меньшую границу: {size}")
        if base_mip.status == split_mip.status == "Optimal" and not np.isclose(
            base_mip.objective, split_mip.objective, atol=1e-4, rtol=0
        ):
            raise RuntimeError(f"Целочисленные оптимумы различаются: {size}")
        # Явное продолжение решения исходной MILP в усиленную модель.
        x = base_mip.x
        u = [round(x[base.w0 + t]) * sum(round(x[k]) for k in starts)
             for (_, t), (_, starts) in active.items()]
        load = strengthened.A @ np.r_[x, u]
        if np.any(load > strengthened.row_hi + TOL) or np.any(load < strengthened.row_lo - TOL):
            raise RuntimeError(f"Продолжение исходного плана нарушает ограничения: {size}")
        print(f"{size}: LP +{split_lp.objective - base_lp.objective:.2f} RUB, "
              f"MILP {base_mip.status} / {split_mip.status}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "comparison.csv", index=False, float_format="%.12g")


if __name__ == "__main__":
    main()
