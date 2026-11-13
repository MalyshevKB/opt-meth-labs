"""ЛР №1, вариант 16: рацион кормления в зоопарке. Запуск: python run.py"""
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zoo_lp import data_gen
from zoo_lp.model import build_model, deletion_filter, elastic, load_data, solve
from zoo_lp.verify import check

sys.stdout.reconfigure(encoding="utf-8")  # чтобы «₽» не ломал вывод в консоли cp1251/cp866 и при перенаправлении

ROOT = Path(__file__).resolve().parent
DATA, RES = ROOT / "data", ROOT / "results"
TOL = 1e-9


# ---------- загрузка ----------
def prepare():
    data_gen.generate(DATA)
    RES.mkdir(exist_ok=True)
    raw = {k: pd.read_csv(DATA / f"{k}.csv") for k in ("feeds", "animals", "norms", "compat")}
    return load_data(DATA), raw


# ---------- сценарии ----------
def scenarios(d, key_feed):
    hay, alf = d.feeds.index("Сено луговое"), d.feeds.index("Люцерна (сено)")
    k = d.feeds.index(key_feed)
    # (код, описание, множители запасов по кормам, множитель поголовья)
    def sup(*pairs):
        s = np.ones(len(d.feeds))
        for i, v in pairs:
            s[i] = v
        return s
    return [
        ("base", "Базовый", sup(), 1.0),
        ("deficit", f"Дефицит: запас «{key_feed}» −20%", sup((k, 0.8)), 1.0),
        ("demand", "Рост спроса: поголовье всех видов +20%", sup(), 1.2),
        ("stress1", "Стресс-1: сено и люцерна −20% и поголовье +20%", sup((hay, 0.8), (alf, 0.8)), 1.2),
        ("stress2", "Стресс-2 (засуха): сено и люцерна −40% и поголовье +20%", sup((hay, 0.6), (alf, 0.6)), 1.2),
    ]


def apply(d, s_mult, h_mult):
    dd = d.copy()
    dd.supply = dd.supply * s_mult
    dd.heads = dd.heads * h_mult
    return dd


def dual_forecast(d, m, r, s_mult, h_mult):
    """Прогноз по теневым ценам базового решения.

    Поголовье входит и в цель, и в строки запасов, поэтому
    z(k·heads, S) = k · z(heads, S/k): рост поголовья в k раз эквивалентен
    уменьшению запасов в k раз с последующим масштабированием стоимости.
    """
    sup_rows = [m.row_names.index(f"supply[{f}]") for f in d.feeds]
    y = r.row_dual[sup_rows]
    db = d.supply * s_mult / h_mult - d.supply
    return h_mult * (r.objective + y @ db)


# ---------- таблицы решения ----------
def solution_table(d, x):
    X = x.reshape(len(d.animals), len(d.feeds))
    rows = []
    for a, an in enumerate(d.animals):
        for f, fn in enumerate(d.feeds):
            rows.append({"animal": an, "feed": fn, "kg_per_head": X[a, f], "heads": d.heads[a],
                         "kg_total": X[a, f] * d.heads[a],
                         "cost_rub_day": X[a, f] * d.heads[a] * d.cost[f]})
    return pd.DataFrame(rows)


def main():
    d, raw = prepare()
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    summary = {"meta": meta}

    # ---------- базовый сценарий ----------
    m = build_model(d)
    base = solve(m, log_file=RES / "highs_base.log")
    summary["model_size"] = {"variables": int(m.A.shape[1]), "constraints": int(m.A.shape[0]),
                             "nonzeros": int(np.count_nonzero(m.A))}
    summary["base"] = {"status": base.status, "objective": base.objective,
                       "iterations": base.iterations}
    if base.status != "Optimal":
        sys.exit(f"Базовый сценарий: {base.status}")
    sol = solution_table(d, base.x)
    sol.to_csv(RES / "solution_base.csv", index=False)

    chk = check(raw["feeds"], raw["animals"], raw["norms"], raw["compat"],
                sol[["animal", "feed", "kg_per_head"]], base.objective)
    chk.to_csv(RES / "check_base.csv", index=False)
    summary["check_base"] = {"n": len(chk), "all_ok": bool(chk.ok.all()),
                             "max_violation": float(chk.violation[:-1].max()),
                             "obj_recomputed": float(chk.lhs.iloc[-1])}

    # нутриенты рациона и их нормы
    X = base.x.reshape(len(d.animals), len(d.feeds))
    nut = pd.DataFrame(X @ d.nutr, index=d.animals, columns=data_gen.NUTRIENTS)
    nut["mass_kg"] = X.sum(1)
    nut.to_csv(RES / "ration_nutrients_base.csv")

    # ---------- теневые цены и диапазоны ----------
    rg = base.ranging
    sp = []
    for i, nm in enumerate(m.row_names):
        act, lo, hi = base.row_act[i], m.row_lo[i], m.row_hi[i]
        active = abs(act - lo) < 1e-7 * max(1, abs(lo)) or abs(act - hi) < 1e-7 * max(1, abs(hi))
        if not active and abs(base.row_dual[i]) < TOL:
            continue
        sp.append({"constraint": nm, "group": m.row_group[i], "activity": act,
                   "lower": lo, "upper": hi, "shadow_price": base.row_dual[i],
                   "rhs_range_dn": rg.row_bound_dn.value_[i] if rg else np.nan,
                   "rhs_range_up": rg.row_bound_up.value_[i] if rg else np.nan})
    sp = pd.DataFrame(sp)
    sp.to_csv(RES / "shadow_prices.csv", index=False)

    rc = []
    for j, nm in enumerate(m.col_names):
        a, f = divmod(j, len(d.feeds))
        rc.append({"variable": nm, "animal": d.animals[a], "feed": d.feeds[f],
                   "allowed": int(d.allowed[a, f]), "value": base.x[j], "cost": m.c[j],
                   "reduced_cost": base.col_dual[j],
                   "cost_range_dn": rg.col_cost_dn.value_[j] if rg else np.nan,
                   "cost_range_up": rg.col_cost_up.value_[j] if rg else np.nan})
    rc = pd.DataFrame(rc)
    rc.to_csv(RES / "cost_ranging.csv", index=False)

    supply_sp = sp[sp.group == "supply"].copy()
    supply_sp["abs"] = supply_sp.shadow_price.abs()
    key_row = supply_sp.sort_values("abs", ascending=False).iloc[0]
    key_feed = key_row.constraint[len("supply["):-1]
    summary["key_feed"] = key_feed

    # ---------- сценарии ----------
    sc_rows, rations = [], []
    for code, desc, s_mult, h_mult in scenarios(d, key_feed):
        dd = apply(d, s_mult, h_mult)
        mm = build_model(dd)
        r = solve(mm, log_file=RES / f"highs_{code}.log")
        row = {"scenario": code, "description": desc, "status": r.status,
               "objective_rub_day": r.objective,
               "delta_vs_base": r.objective - base.objective,
               "forecast_by_shadow_prices": dual_forecast(d, m, base, s_mult, h_mult),
               "simplex_iterations": r.iterations["simplex"]}
        row["forecast_error"] = row["objective_rub_day"] - row["forecast_by_shadow_prices"]
        if r.status == "Optimal":
            t = solution_table(dd, r.x)
            t["scenario"] = code
            rations.append(t)
            rf = raw["feeds"].copy(); rf["supply_kg_per_day"] = dd.supply
            ra = raw["animals"].copy(); ra["heads"] = dd.heads
            c = check(rf, ra, raw["norms"], raw["compat"], t[["animal", "feed", "kg_per_head"]], r.objective)
            row["check_all_ok"] = bool(c.ok.all())
            row["check_max_violation"] = float(c.violation[:-1].max())
            row["cost_per_head_zebra"] = float(t[t.animal == "Зебра"].cost_rub_day.sum() / dd.heads[0])
            sup_rows = [mm.row_names.index(f"supply[{f}]") for f in dd.feeds]
            row["binding_supplies"] = "; ".join(
                f"{f} ({r.row_dual[i]:.2f} ₽/кг)" for f, i in zip(dd.feeds, sup_rows)
                if abs(r.row_dual[i]) > TOL)
        else:
            iis = deletion_filter(mm)
            pd.DataFrame({"constraint": iis}).to_csv(RES / f"iis_{code}.csv", index=False)
            v_all = elastic(mm)
            v_sup = elastic(mm, groups={"supply"})
            v_norm = elastic(mm, groups={"norm"})
            el = [{"variant": "все ограничения", "constraint": n, "bound": b, "rhs": rhs, "relaxation": v}
                  for n, b, rhs, v in v_all[1]]
            el += [{"variant": "только запасы", "constraint": n, "bound": b, "rhs": rhs, "relaxation": v}
                   for n, b, rhs, v in v_sup[1]]
            el += [{"variant": "только нормы", "constraint": n, "bound": b, "rhs": rhs, "relaxation": v}
                   for n, b, rhs, v in v_norm[1]]
            pd.DataFrame(el).to_csv(RES / f"elastic_{code}.csv", index=False)
            summary[f"infeasible_{code}"] = {"iis": iis, "elastic": el}
        sc_rows.append(row)
    sc = pd.DataFrame(sc_rows)
    sc.to_csv(RES / "scenarios.csv", index=False)
    rations = pd.concat(rations)
    rations.to_csv(RES / "ration_by_scenario.csv", index=False)

    # ---------- точность прогноза теневой цены: параметрический анализ ----------
    kf = d.feeds.index(key_feed)
    y_key = float(key_row.shadow_price)
    par = []
    for mult in np.round(np.arange(0.40, 1.61, 0.02), 2):
        s = np.ones(len(d.feeds)); s[kf] = mult
        r = solve(build_model(apply(d, s, 1.0)), ranging=False)
        par.append({"supply_mult": mult, "supply_kg": d.supply[kf] * mult, "status": r.status,
                    "objective": r.objective,
                    "forecast": base.objective + y_key * d.supply[kf] * (mult - 1)})
    par = pd.DataFrame(par)
    par.to_csv(RES / "parametric_key_supply.csv", index=False)

    # ---------- доп. задание: симплекс vs внутренняя точка ----------
    cmp_rows = []
    for solver in ("simplex", "ipm"):
        for presolve in ("on", "off"):
            times, it = [], None
            for _ in range(30):
                r = solve(m, solver=solver, presolve=presolve, ranging=False)
                times.append(r.time_s); it = r.iterations
            cmp_rows.append({"solver": solver, "presolve": presolve, "status": r.status,
                             "objective": r.objective, "median_time_ms": 1000 * statistics.median(times),
                             **{f"iter_{k}": v for k, v in it.items()}})
    pd.DataFrame(cmp_rows).to_csv(RES / "solver_comparison.csv", index=False)

    summary["scenarios"] = sc.to_dict(orient="records")
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")
    plots(d, sol, sp, rations, sc, par, key_feed, base.objective, y_key)
    print_report(d, base, chk, sp, sc, key_feed)


# ---------- графики ----------
def plots(d, sol, sp, rations, sc, par, key_feed, z0, y_key):
    colors = plt.cm.tab10(np.arange(len(d.feeds)))
    feasible = [s for s in sc.scenario if s in set(rations.scenario)]
    fig, axes = plt.subplots(1, len(d.animals), figsize=(15, 5), sharey=False)
    for ax, an in zip(axes, d.animals):
        bottom = np.zeros(len(feasible))
        for f, fn in enumerate(d.feeds):
            vals = [rations[(rations.scenario == s) & (rations.animal == an) & (rations.feed == fn)]
                    .kg_per_head.sum() for s in feasible]
            if max(vals) > 1e-9:
                ax.bar(feasible, vals, bottom=bottom, color=colors[f], label=fn)
                bottom += vals
        ax.set_title(an); ax.set_ylabel("кг на голову в сутки"); ax.tick_params(axis="x", rotation=30)
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Оптимальный рацион по сценариям")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(RES / "ration_scenarios.png", dpi=130); plt.close(fig)

    nz = sp[sp.shadow_price.abs() > TOL].sort_values("shadow_price")
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(nz) + 1.5))
    ax.barh(nz.constraint, nz.shadow_price, color=["#c0392b" if v < 0 else "#2e86c1" for v in nz.shadow_price])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("теневая цена, ₽/сут на единицу правой части")
    ax.set_title("Теневые цены активных ограничений (базовый сценарий)")
    fig.tight_layout(); fig.savefig(RES / "shadow_prices.png", dpi=130); plt.close(fig)

    ok = par[par.status == "Optimal"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ok.supply_kg, ok.objective, "o-", ms=3, label="фактический оптимум z*(b)")
    ax.plot(par.supply_kg, par.forecast, "--", label=f"прогноз z0 + y·Δb, y = {y_key:.2f} ₽/кг")
    b0 = d.supply[d.feeds.index(key_feed)]
    ax.axvline(b0, color="gray", lw=0.8); ax.axvline(0.8 * b0, color="gray", lw=0.8, ls=":")
    ax.annotate("база", (b0, z0), textcoords="offset points", xytext=(5, 10))
    ax.annotate("−20%", (0.8 * b0, z0), textcoords="offset points", xytext=(5, 30))
    ax.set_xlabel(f"запас «{key_feed}», кг/сут"); ax.set_ylabel("стоимость рациона, ₽/сут")
    ax.set_title("Параметрический анализ: оптимум против прогноза теневой цены")
    ax.legend(); fig.tight_layout(); fig.savefig(RES / "parametric_key_supply.png", dpi=130); plt.close(fig)

    used = sol.groupby("feed", sort=False).kg_total.sum().reindex(d.feeds)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(d.feeds, d.supply, color="#d5dbdb", label="запас")
    ax.bar(d.feeds, used, color="#1f618d", label="расход в оптимуме")
    ax.set_ylabel("кг/сут"); ax.tick_params(axis="x", rotation=35); ax.legend()
    ax.set_title("Использование запасов кормов (базовый сценарий)")
    fig.tight_layout(); fig.savefig(RES / "supply_usage.png", dpi=130); plt.close(fig)


def print_report(d, base, chk, sp, sc, key_feed):
    print(f"Вариант 16, seed {data_gen.SEED}. Статус: {base.status}, "
          f"стоимость рациона {base.objective:.2f} ₽/сут, итераций симплекса {base.iterations['simplex']}")
    print(f"Независимая проверка: {len(chk)} условий, все выполнены: {bool(chk.ok.all())}")
    print(f"Ключевой ресурс по теневой цене: {key_feed}")
    print(sc[["scenario", "status", "objective_rub_day", "forecast_by_shadow_prices"]].to_string(index=False))
    print(f"Результаты: {RES}")


if __name__ == "__main__":
    main()
