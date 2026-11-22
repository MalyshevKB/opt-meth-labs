"""ЛР №3, вариант 16: план ремонтов оборудования по неделям с бюджетом бригад.
Запуск: python run.py"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from maint import data_gen
from maint.heuristic import greedy
from maint.model import (build_model, contractor_table, from_frames, load_data, schedule_table, solve,
                         tight_M)
from maint.verify import check

sys.stdout.reconfigure(encoding="utf-8")  # чтобы «₽» не ломал вывод в консоли cp1251/cp866 и при перенаправлении

ROOT = Path(__file__).resolve().parent
DATA, RES = ROOT / "data", ROOT / "results"
SKIP_PENALTY = 1e9    # ₽ — штраф за невыполненный обязательный ремонт (диагностика несовместности)


def raw_tables():
    return ([pd.read_csv(DATA / f) for f in ("units.csv", "crew.csv", "conflicts.csv")],
            json.loads((DATA / "params.json").read_text(encoding="utf-8")))


def verify(raw, params, d, sched, contr, M=None, obj=None):
    """Проверка на исходных CSV с поправками сценария (часы бригады, лимит K, доступность подрядчика)."""
    units, crew, conflicts = raw
    crew = crew.copy(); crew["own_crew_hours"] = d.own
    p = dict(params, max_units_down=d.K)
    return check(units, crew, conflicts, p, sched, contr, M, obj, contractor_ok=d.contractor_ok)


def gap_pct(milp, lp):
    return 100 * (milp - lp) / abs(milp)


def greedy_tables(d, start, called, hours):
    s = pd.DataFrame({"unit": d.units, "start_week": [None if x is None else x + 1 for x in start]})
    c = pd.DataFrame({"week": range(1, d.T + 1), "called": called, "hours": hours})
    return s, c


# ---------- сценарии ----------
def scenarios(d):
    def deficit(dd): dd.own = dd.own * 0.8
    def stress(dd): deficit(dd); dd.contractor_ok[:3] = 0
    def stress2(dd): dd.own = dd.own * 0.6; dd.contractor_ok[:] = 0
    return [("base", "Базовый", lambda dd: None),
            ("deficit", "Дефицит: собственная бригада −20%", deficit),
            ("stress", "Стресс: бригада −20% и подрядчик недоступен на неделях 1–3", stress),
            ("stress2", "Стресс-2: бригада −40% и подрядчика нет", stress2)]


def diagnose_infeasible(d):
    """Обязательные ремонты становятся «желательными» с огромным штрафом:
    решение показывает, какие из них выполнить невозможно."""
    dd = d.copy()
    mand = np.where(dd.mandatory == 1)[0]
    dd.mandatory = np.zeros_like(dd.mandatory)
    dd.post = dd.post.copy(); dd.post[mand] += SKIP_PENALTY
    m = build_model(dd)
    r = solve(m)
    s = schedule_table(dd, m, r.x)
    skipped = [d.units[i] for i in mand if pd.isna(s.start_week[i])]
    reasons = []
    for i in mand:
        if d.units[i] in skipped:
            wk = range(0, min(d.deadline[i], d.T) - d.p[i] + 1)
            cap = max(d.own[t:t + d.p[i]].min() + d.C_max * d.contractor_ok[t:t + d.p[i]].min() for t in wk)
            reasons.append({"unit": d.units[i], "deadline_week": int(d.deadline[i]),
                            "crew_hours_needed_per_week": float(d.h[i]),
                            "max_hours_available_before_deadline": float(cap)})
    return skipped, reasons


def main():
    data_gen.generate(DATA)
    RES.mkdir(exist_ok=True)
    raw, params = raw_tables()
    d = load_data(DATA)
    summary = {"meta": params}

    # ---------- базовый MILP и LP-релаксация ----------
    m = build_model(d)
    base = solve(m, log_file=RES / "highs_base.log")
    relax = solve(m, relax=True, log_file=RES / "highs_base_relaxation.log")
    sched, contr = schedule_table(d, m, base.x), contractor_table(d, m, base.x)
    sched.to_csv(RES / "schedule_base.csv", index=False)
    contr.to_csv(RES / "contractor_base.csv", index=False)
    chk = verify(raw, params, d, sched, contr, m.M, base.objective)
    chk.to_csv(RES / "check_base.csv", index=False)
    summary["model_size"] = {"variables": int(m.A.shape[1]), "binary": int(m.integrality.sum()),
                             "continuous": int((m.integrality == 0).sum()), "constraints": int(m.A.shape[0]),
                             "nonzeros": int(m.A.nnz), "M": m.M.tolist()}
    summary["base"] = {"status": base.status, "objective": base.objective, "time_s": base.time_s, **base.info,
                       "lp_relaxation": relax.objective, "integrality_gap_pct": gap_pct(base.objective, relax.objective),
                       "check_n": len(chk), "check_ok": bool(chk.ok.all()),
                       "check_max_violation": float(chk.violation.max())}
    summary["base"]["cost_breakdown"] = cost_breakdown(d, m, base.x)
    summary["relaxation"] = {"cost_breakdown": cost_breakdown(d, m, relax.x)}

    # где «ломается» непрерывное решение
    frac = [{"variable": m.col_names[k], "lp_value": relax.x[k], "milp_value": base.x[k]}
            for k in range(len(m.c)) if m.integrality[k] and 1e-6 < relax.x[k] < 1 - 1e-6]
    pd.DataFrame(frac).to_csv(RES / "relaxation_fractional.csv", index=False)
    lp_contr = contractor_table(d, m, relax.x)
    lp_contr.to_csv(RES / "contractor_relaxation.csv", index=False)
    summary["relaxation"]["n_fractional"] = len(frac)
    summary["relaxation"]["contractor"] = lp_contr.to_dict(orient="records")

    # ---------- сценарии ----------
    rows, sched_all, contr_all = [], [], []
    for code, desc, mod in scenarios(d):
        dd = d.copy(); mod(dd)
        mm = build_model(dd)
        r = solve(mm, log_file=RES / f"highs_{code}.log")
        row = {"scenario": code, "description": desc, "status": r.status, "objective_rub": r.objective}
        if r.status == "Optimal":
            rl = solve(mm, relax=True)
            s, c = schedule_table(dd, mm, r.x), contractor_table(dd, mm, r.x)
            ch = verify(raw, params, dd, s, c, mm.M, r.objective)
            t0 = time.perf_counter(); gst, gcl, ghr, gcost, gok = greedy(dd); gt = time.perf_counter() - t0
            gs, gc = greedy_tables(dd, gst, gcl, ghr)
            gch = verify(raw, params, dd, gs, gc, None, gcost)
            row.update(delta_vs_base=r.objective - base.objective, lp_relaxation=rl.objective,
                       integrality_gap_pct=gap_pct(r.objective, rl.objective), milp_time_s=r.time_s,
                       nodes=r.info["mip_nodes"], check_ok=bool(ch.ok.all()),
                       repaired=", ".join(f"{u} (нед {int(w)})" for u, w in zip(s.unit, s.start_week) if pd.notna(w)),
                       not_repaired=", ".join(s.unit[s.start_week.isna()]),
                       contractor_weeks=", ".join(f"нед {w}: {h:.0f} ч" for w, h, k in zip(c.week, c.hours, c.called) if k > 0.5),
                       greedy_cost=gcost, greedy_gap_pct=100 * (gcost / r.objective - 1), greedy_time_s=gt,
                       greedy_check_ok=bool(gch.ok.all()),
                       greedy_not_repaired=", ".join(gs.unit[gs.start_week.isna()]))
            s["scenario"] = code; c["scenario"] = code
            sched_all.append(s); contr_all.append(c)
        else:
            skipped, reasons = diagnose_infeasible(dd)
            row["infeasible_mandatory"] = ", ".join(skipped)
            summary[f"infeasible_{code}"] = reasons
        rows.append(row)
    sc = pd.DataFrame(rows)
    sc.to_csv(RES / "scenarios.csv", index=False)
    pd.concat(sched_all).to_csv(RES / "schedule_by_scenario.csv", index=False)
    pd.concat(contr_all).to_csv(RES / "contractor_by_scenario.csv", index=False)
    summary["scenarios"] = sc.to_dict(orient="records")

    # ---------- выбор big-M ----------
    summary["bigM"] = bigM_experiment(d, raw, params, base)

    # ---------- масштабирование ----------
    summary["scaling"] = scaling()

    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    plots(d, sc)
    print(f"Вариант 16, seed {data_gen.SEED}. Базовый MILP: {base.status}, {base.objective:,.0f} ₽; "
          f"LP-релаксация {relax.objective:,.0f} ₽, зазор {summary['base']['integrality_gap_pct']:.2f}%")
    print(f"Проверка: {len(chk)} условий, все выполнены: {bool(chk.ok.all())}")
    print(sc[["scenario", "status", "objective_rub"]].to_string(index=False))
    print(pd.DataFrame(summary["scaling"])[["n_units", "milp_status", "milp_time_s", "integrality_gap_pct",
                                            "greedy_gap_pct"]].to_string(index=False))


def cost_breakdown(d, m, x):
    """Слагаемые цели: ремонт, простой, риск отказов, подрядчик."""
    z = x[:m.w0]
    rep = sum(z[k] * (d.c[i]) for k, (i, _) in enumerate(m.z_keys))
    down = sum(z[k] * d.v[i] * d.p[i] for k, (i, _) in enumerate(m.z_keys))
    started = np.zeros((len(d.units), d.T))
    for k, (i, tau) in enumerate(m.z_keys):
        started[i, tau:] += z[k]
    done = np.array([z[[k for k, (ii, _) in enumerate(m.z_keys) if ii == i]].sum() for i in range(len(d.units))])
    risk = float((d.rho * (1 - started)).sum() + (d.post * (1 - done)).sum())
    contractor = float((x[m.w0:m.w0 + d.T] * d.F + x[m.o0:m.o0 + d.T] * d.rate).sum())
    return {"repair": float(rep), "downtime": float(down), "failure_risk": risk, "contractor": contractor}


def bigM_experiment(d, raw, params, base):
    """Сравнение обоснованного M и завышенных значений: сила LP-релаксации, ветвление, корректность."""
    rows = []
    instances = [(10, d), (40, from_frames(*data_gen.generate_frames(40)))]
    for n, dd in instances:
        Mt = tight_M(dd)
        for label, M in [("обоснованное (min(C_max, пик−бригада))", Mt), ("10 × обоснованное", 10 * Mt),
                         ("10^6", np.full(dd.T, 1e6)), ("10^9", np.full(dd.T, 1e9))]:
            if n == 40 and label == "10^9":
                continue
            mm = build_model(dd, M)
            rl = solve(mm, relax=True)
            r = solve(mm, options={"time_limit": 120.0})
            rows.append({"n_units": n, "M": label, "M_value": float(M.max()), "presolve": "on",
                         "lp_relaxation": rl.objective, "milp": r.objective, "status": r.status,
                         "integrality_gap_pct": gap_pct(r.objective, rl.objective),
                         "time_s": r.time_s, "nodes": r.info["mip_nodes"],
                         "lp_w_max": float(rl.x[mm.w0:mm.w0 + dd.T].max())})
    # численная ловушка: огромное M без presolve
    for M_label, M in [("обоснованное (min(C_max, пик−бригада))", tight_M(d)), ("10^9", np.full(d.T, 1e9))]:
        mm = build_model(d, M)
        r = solve(mm, options={"presolve": "off"}, log_file=RES / f"highs_bigM_presolve_off_{'tight' if M.max() < 1e6 else '1e9'}.log")
        s, c = schedule_table(d, mm, r.x), contractor_table(d, mm, r.x)
        c["called_raw"] = r.x[mm.w0:mm.w0 + d.T]
        ch = verify(raw, params, d, s, c.assign(called=c.called_raw), None, None)
        c.to_csv(RES / f"contractor_presolve_off_{'tight' if M.max() < 1e6 else '1e9'}.csv", index=False)
        bad = ch[~ch.ok]
        rows.append({"n_units": 10, "M": M_label, "M_value": float(M.max()), "presolve": "off",
                     "milp": r.objective, "status": r.status, "time_s": r.time_s, "nodes": r.info["mip_nodes"],
                     "w_week1": float(r.x[mm.w0]), "o_week1": float(r.x[mm.o0]),
                     "physical_check_ok": bool(ch.ok.all()),
                     "violations": "; ".join(f"{b.check} [{b.object}]: {b.lhs:.1f} > {b.rhs:.1f}" for _, b in bad.iterrows())})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "bigM_experiment.csv", index=False)
    return df.to_dict(orient="records")


def scaling():
    rows = []
    for n in (10, 20, 40, 80):
        dd = from_frames(*data_gen.generate_frames(n))
        mm = build_model(dd)
        rl = solve(mm, relax=True)
        r = solve(mm, options={"time_limit": 60.0})
        t0 = time.perf_counter(); _, _, _, gcost, gok = greedy(dd); gt = time.perf_counter() - t0
        rows.append({"n_units": n, "binary_vars": int(mm.integrality.sum()), "constraints": int(mm.A.shape[0]),
                     "lp_time_s": rl.time_s, "milp_time_s": r.time_s, "milp_status": r.status,
                     "nodes": r.info["mip_nodes"], "final_mip_gap_pct": 100 * r.info["mip_gap"],
                     "milp": r.objective, "lp_relaxation": rl.objective,
                     "integrality_gap_pct": gap_pct(r.objective, rl.objective),
                     "greedy": gcost, "greedy_feasible": gok, "greedy_gap_pct": 100 * (gcost / r.objective - 1),
                     "greedy_time_s": gt})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "scaling.csv", index=False)
    return df.to_dict(orient="records")


# ---------- графики ----------
def plots(d, sc):
    sched = pd.read_csv(RES / "schedule_by_scenario.csv")
    contr = pd.read_csv(RES / "contractor_by_scenario.csv")
    codes = list(sched.scenario.unique())
    fig, axes = plt.subplots(2, len(codes), figsize=(5.2 * len(codes), 9),
                             gridspec_kw={"height_ratios": [3, 1.4]}, sharex=True)
    p = dict(zip(d.units, d.p)); h = dict(zip(d.units, d.h))
    for col, code in enumerate(codes):
        ax = axes[0, col]
        s = sched[sched.scenario == code].reset_index(drop=True)
        dd = d.copy()
        for _, mod in [(c, f) for c, _, f in scenarios(d) if c == code]:
            mod(dd)
        for y, r in s.iterrows():
            mand = d.mandatory[d.units.index(r.unit)]
            if pd.notna(r.start_week):
                ax.barh(y, p[r.unit], left=r.start_week - 0.5, color="#c0392b" if mand else "#2471a3")
                ax.text(r.start_week - 0.4, y, f"{h[r.unit]:.0f} ч", va="center", color="w", fontsize=7)
            else:
                ax.text(0.6, y, "не ремонтируется", va="center", fontsize=7, color="#7f8c8d")
            if mand:
                ax.plot(d.deadline[d.units.index(r.unit)] + 0.5, y, "k|", ms=12)
        ax.set_yticks(range(len(s)), s.unit if col == 0 else [""] * len(s), fontsize=8)
        ax.invert_yaxis(); ax.set_title(code); ax.grid(axis="x", alpha=0.3)
        ax2 = axes[1, col]
        down = np.zeros(d.T)
        for _, r in s.iterrows():
            if pd.notna(r.start_week):
                down[int(r.start_week) - 1:int(r.start_week) - 1 + p[r.unit]] += h[r.unit]
        c = contr[contr.scenario == code]
        wk = np.arange(1, d.T + 1)
        ax2.bar(wk, np.minimum(down, dd.own), color="#2471a3", label="своя бригада")
        ax2.bar(wk, c.hours, bottom=np.minimum(down, dd.own), color="#e67e22", label="подрядчик")
        ax2.step(wk, dd.own, where="mid", color="k", lw=1, label="лимит бригады")
        ax2.set_xticks(wk); ax2.set_xlabel("неделя")
        ax2.set_ylim(0, 1.15 * max(down.max(), dd.own.max()))
        if col == 0:
            ax2.set_ylabel("бригадо-часы"); ax2.legend(fontsize=7)
    fig.suptitle("План ремонтов по сценариям (красные — обязательные, | — срок предписания)")
    fig.tight_layout(); fig.savefig(RES / "schedule_scenarios.png", dpi=120); plt.close(fig)

    sca = pd.read_csv(RES / "scaling.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(sca.n_units, sca.milp_time_s, "o-", label="MILP (лимит 60 с)")
    ax[0].plot(sca.n_units, sca.lp_time_s, "s-", label="LP-релаксация")
    ax[0].plot(sca.n_units, sca.greedy_time_s, "^-", label="жадная эвристика")
    ax[0].set_yscale("log"); ax[0].set_xlabel("число единиц оборудования"); ax[0].set_ylabel("время, с")
    ax[0].set_title("Время решения"); ax[0].legend()
    ax[1].plot(sca.n_units, sca.integrality_gap_pct, "o-", label="интегральный зазор (MILP vs LP)")
    ax[1].plot(sca.n_units, sca.final_mip_gap_pct, "s-", label="остаточный зазор B&B")
    ax[1].plot(sca.n_units, sca.greedy_gap_pct, "^-", label="эвристика vs MILP")
    ax[1].set_xlabel("число единиц оборудования"); ax[1].set_ylabel("%"); ax[1].set_title("Зазоры")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(RES / "scaling.png", dpi=130); plt.close(fig)

    bm = pd.read_csv(RES / "bigM_experiment.csv")
    bm = bm[bm.presolve == "on"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for n, g in bm.groupby("n_units"):
        ax[0].plot(g.M_value, g.integrality_gap_pct, "o-", label=f"{n} единиц")
        ax[1].plot(g.M_value, g.nodes, "o-", label=f"{n} единиц")
    for a, t in zip(ax, ["Интегральный зазор, %", "Узлов ветвления"]):
        a.set_xscale("log"); a.set_xlabel("M"); a.set_title(t); a.legend()
    fig.tight_layout(); fig.savefig(RES / "bigM.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
