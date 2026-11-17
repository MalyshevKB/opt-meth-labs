"""ЛР №2, вариант 16: балансировка запасов между складами розничной сети.
Запуск: python run.py"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from netflow import data_gen
from netflow.heuristic import greedy
from netflow.model import build_model, flows_table, inventory_table, load_data, solve, x_idx
from netflow.verify import check

sys.stdout.reconfigure(encoding="utf-8")  # чтобы «₽» не ломал вывод в консоли cp1251/cp866 и при перенаправлении

ROOT = Path(__file__).resolve().parent
DATA, RES = ROOT / "data", ROOT / "results"
EMERGENCY = 1500.0     # ₽/паллет — экстренная поставка от производителя (оценка непокрытого спроса)
BIG_PENALTY = 1e5      # ₽/паллет — для поиска минимального дефицита


def raw_tables():
    return {k: pd.read_csv(DATA / f"{k}.csv") for k in ("warehouses", "routes", "demand", "deliveries")}


def verify(raw, d, flows, stock, obj, trucks=None):
    """Проверка на исходных CSV с учётом изменений сценария (спрос, закрытые дуги, пропускные способности)."""
    routes = raw["routes"].copy(); routes["cap_pallets_per_week"] = d.r_cap
    routes["max_trucks_per_week"] = d.r_trucks_max
    demand = raw["demand"].copy()
    demand["demand_pallets"] = [d.demand[d.wh.index(w), t - 1] for w, t in zip(demand.warehouse, demand.week)]
    closed = {(d.wh[d.r_from[r]], d.wh[d.r_to[r]], t + 1)
              for r in range(len(d.r_from)) for t in range(d.weeks) if d.r_open[r, t] == 0}
    return check(raw["warehouses"], routes, demand, raw["deliveries"], flows, stock, obj, closed,
                 trucks=trucks, truck_size=d.truck_size)


def reverse(d, r):
    return [k for k in range(len(d.r_from)) if d.r_from[k] == d.r_to[r] and d.r_to[k] == d.r_from[r]][0]


def min_shortage(d):
    m = build_model(d, shortage_penalty=BIG_PENALTY)
    r = solve(m)
    inv = inventory_table(d, m, r.x)
    return inv[inv.shortage > 1e-6], float(inv.shortage.sum())


def max_demand_growth(d, lo=1.0, hi=1.5, tol=1e-4):
    """Бисекция по множителю спроса: наибольший рост, при котором задача совместна."""
    def ok(k):
        dd = d.copy(); dd.demand = dd.demand * k
        return solve(build_model(dd)).status == "Optimal"
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def main():
    data_gen.generate(DATA)
    RES.mkdir(exist_ok=True)
    raw = raw_tables()
    d = load_data(DATA)
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    summary = {"meta": meta}
    nR, T, nW = len(d.r_from), d.weeks, len(d.wh)

    # ---------- базовое решение ----------
    m = build_model(d)
    base = solve(m, log_file=RES / "highs_base.log")
    summary["model_size"] = {"variables": int(m.A.shape[1]), "constraints": int(m.A.shape[0]),
                             "nonzeros": int(m.A.nnz), "nodes": nW * T, "transfer_arcs": nR * T}
    summary["base"] = {"status": base.status, "objective": base.objective, **base.info,
                       "integral_solution": bool(np.allclose(base.x, np.round(base.x), atol=1e-7))}
    fl, inv = flows_table(d, m, base.x), inventory_table(d, m, base.x)
    fl.to_csv(RES / "flows_base.csv", index=False)
    inv.to_csv(RES / "stock_base.csv", index=False)
    chk = verify(raw, d, fl, inv, base.objective)
    chk.to_csv(RES / "check_base.csv", index=False)
    summary["check_base"] = {"n": len(chk), "all_ok": bool(chk.ok.all()),
                             "max_violation": float(chk.violation.max()),
                             "by_type": chk.groupby("check").size().to_dict()}
    X = base.x[:m.n_x].reshape(T, nR).T          # R×T
    transport = float((d.r_cost[:, None] * X).sum())
    summary["base"]["transport_cost"] = transport
    summary["base"]["holding_cost"] = base.objective - transport
    summary["base"]["pallets_moved"] = float(X.sum())

    # ---------- потенциалы узлов и узкие места ----------
    pi = base.row_dual.reshape(T, nW).T            # W×T: ∂z/∂(спрос узла)
    pd.DataFrame(pi, index=d.wh, columns=[f"нед {t+1}" for t in range(T)]).to_csv(RES / "potentials.csv")
    bott = []
    for r in range(nR):
        for t in range(T):
            j = x_idx(d, r, t)
            xv, u = base.x[j], d.r_cap[r]
            dpi = pi[d.r_to[r], t] - pi[d.r_from[r], t]
            bott.append({"arc": d.route_name(r), "week": t + 1, "flow": xv, "cap": u,
                         "load": xv / u, "saturated": bool(xv > 0 and abs(xv - u) < 1e-6),
                         "cost_per_pallet": d.r_cost[r], "potential_diff": dpi,
                         "reduced_cost": base.col_dual[j], "c_minus_dpi": d.r_cost[r] - dpi})
    bott = pd.DataFrame(bott)
    bott.to_csv(RES / "arcs_base.csv", index=False)
    sat = bott[bott.saturated].sort_values("reduced_cost")
    sat.to_csv(RES / "bottlenecks.csv", index=False)

    # проверка прогноза теневой цены: +1 паллета пропускной способности на самой «дорогой» дуге
    top = sat.iloc[0]
    r_top = [r for r in range(nR) if d.route_name(r) == top.arc][0]
    # +1 паллета только в ту неделю, где дуга насыщена: правим верхнюю границу переменной
    m1 = build_model(d); m1.col_hi[x_idx(d, r_top, int(top.week) - 1)] += 1
    z1 = solve(m1).objective
    summary["capacity_plus_one"] = {"arc": top.arc, "week": int(top.week),
                                    "predicted": float(top.reduced_cost), "actual": z1 - base.objective}

    # ---------- сценарии ----------
    busiest = int(np.argmax(X.sum(1)))
    rev = reverse(d, busiest)
    summary["busiest_arc"] = {"arc": d.route_name(busiest), "total_flow": float(X[busiest].sum()),
                              "by_week": X[busiest].tolist(), "cap": float(d.r_cap[busiest])}

    def sc_overload(dd): dd.r_cap = dd.r_cap.copy(); dd.r_cap[busiest] *= 0.5
    def sc_failure(dd): dd.r_open[[busiest, rev]] = 0
    def sc_demand(dd): dd.demand = dd.demand * 1.2
    def sc_stress(dd): sc_failure(dd); dd.demand = dd.demand * 1.05
    scen = [
        ("base", "Базовый", lambda dd: None),
        ("overload", f"Перегрузка: пропускная способность {d.route_name(busiest)} −50%", sc_overload),
        ("failure", f"Отказ участка: дорога {d.wh[d.r_from[busiest]]}—{d.wh[d.r_to[busiest]]} закрыта", sc_failure),
        ("demand", "Рост спроса +20% на всех складах", sc_demand),
        ("stress", "Стресс: отказ дороги и рост спроса +5%", sc_stress),
    ]
    # прогноз по теневой цене для перегрузки: Σ_t d_{r,t} · Δu
    pred_overload = base.objective + sum(base.col_dual[x_idx(d, busiest, t)] * (-0.5 * d.r_cap[busiest])
                                         for t in range(T))
    rows, flows_all = [], []
    for code, desc, mod in scen:
        dd = d.copy(); mod(dd)
        mm = build_model(dd)
        r = solve(mm, log_file=RES / f"highs_{code}.log")
        row = {"scenario": code, "description": desc, "status": r.status, "objective_rub": r.objective,
               "delta_rub": r.objective - base.objective, "delta_pct": 100 * (r.objective / base.objective - 1)}
        if code == "overload":
            row["forecast_by_shadow_price"] = pred_overload
        if r.status == "Optimal":
            f, s = flows_table(dd, mm, r.x), inventory_table(dd, mm, r.x)
            c = verify(raw, dd, f, s, r.objective)
            row.update(check_all_ok=bool(c.ok.all()), check_max_violation=float(c.violation.max()),
                       saturated_arcs=int(((f.flow_pallets > 0) & (abs(f.flow_pallets - f.cap_pallets) < 1e-6)).sum()),
                       pallets_moved=float(f.flow_pallets.sum()))
            f["scenario"] = code; flows_all.append(f)
        else:
            sh, total = min_shortage(dd)
            sh.to_csv(RES / f"shortage_{code}.csv", index=False)
            row["min_shortage_pallets"] = total
            summary[f"shortage_{code}"] = sh.to_dict(orient="records")
        rows.append(row)
    sc = pd.DataFrame(rows)
    summary["max_demand_growth"] = max_demand_growth(d)
    sc.to_csv(RES / "scenarios.csv", index=False)
    pd.concat(flows_all).to_csv(RES / "flows_by_scenario.csv", index=False)
    summary["scenarios"] = sc.to_dict(orient="records")
    summary["supply_balance"] = {"total_supply": float(d.init.sum() + d.deliv.sum()),
                                 "total_demand": float(d.demand.sum()), "safety": float(d.safety.sum())}

    # ---------- жадная эвристика ----------
    t0 = time.perf_counter(); gx, gI, gs, gcost = greedy(d); g_time = time.perf_counter() - t0
    gflows = pd.DataFrame([{"from": d.wh[d.r_from[r]], "to": d.wh[d.r_to[r]], "week": t + 1,
                            "flow_pallets": gx[r, t], "cap_pallets": d.r_cap[r]} for t in range(T) for r in range(nR)])
    gstock = pd.DataFrame([{"warehouse": d.wh[w], "week": t + 1, "stock_end": gI[w, t], "shortage": gs[w, t]}
                           for t in range(T) for w in range(nW)])
    gflows.to_csv(RES / "flows_greedy.csv", index=False); gstock.to_csv(RES / "stock_greedy.csv", index=False)
    gchk = verify(raw, d, gflows, gstock, gcost)
    me = build_model(d, shortage_penalty=EMERGENCY)
    le = solve(me)
    summary["greedy"] = {"cost_without_shortage": gcost, "shortage_pallets": float(gs.sum()),
                         "cost_with_emergency": gcost + EMERGENCY * gs.sum(), "time_s": g_time,
                         "lp_time_s": base.time_s, "lp_cost": base.objective,
                         "lp_elastic_cost": le.objective,
                         "gap_pct": 100 * ((gcost + EMERGENCY * gs.sum()) / base.objective - 1),
                         "check_ok_with_shortage_as_supply": bool(gchk.ok.all()),
                         "shortage_by_node": gstock[gstock.shortage > 0].to_dict(orient="records"),
                         "pallets_moved": float(gx.sum())}

    # ---------- дополнительное задание: неделимые рейсы (MILP) ----------
    summary["milp"] = milp_part(d, raw, base, X)

    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    plots(d, X, inv, pi, sc, gx, summary)
    print(f"Вариант 16, seed {data_gen.SEED}. Базовый: {base.status}, {base.objective:,.0f} ₽ за 4 недели")
    print(f"Проверка: {len(chk)} условий, все выполнены: {bool(chk.ok.all())}")
    print(sc[["scenario", "status", "objective_rub", "delta_pct"]].to_string(index=False))
    print(f"Макс. рост спроса без дефицита: {100 * (summary['max_demand_growth'] - 1):.2f}%")
    g = summary["greedy"]; mi = summary["milp"]
    print(f"Жадная: дефицит {g['shortage_pallets']:.0f} паллет, стоимость с экстренными поставками "
          f"{g['cost_with_emergency']:,.0f} ₽ (+{g['gap_pct']:.1f}% к ЛП)")
    print(f"MILP: {mi['milp_objective']:,.0f} ₽, зазор к LP-релаксации {mi['integrality_gap_pct']:.2f}%, "
          f"рейсов {mi['milp_trucks']:.0f}")


def milp_part(d, raw, base, X_lp):
    nR, T = len(d.r_from), d.weeks
    mt = build_model(d, trucks=True)
    relax = solve(mt, relax=True)
    milp = solve(mt, log_file=RES / "highs_milp.log", options={"mip_rel_gap": 0.0, "time_limit": 300.0})
    Xm = milp.x[:mt.n_x].reshape(T, nR).T
    Ym = milp.x[mt.n_x + mt.n_I:mt.n_x + mt.n_I + mt.n_y].reshape(T, nR).T
    fm, sm = flows_table(d, mt, milp.x), inventory_table(d, mt, milp.x)
    fm.to_csv(RES / "flows_milp.csv", index=False); sm.to_csv(RES / "stock_milp.csv", index=False)
    trucks = fm.set_index(["from", "to", "week"]).trucks
    chk = verify(raw, d, fm, sm, milp.objective, trucks=trucks)
    chk.to_csv(RES / "check_milp.csv", index=False)

    # наивное округление LP: те же потоки, рейсов ceil(x/33)
    Y_round = np.ceil(X_lp / d.truck_size - 1e-9)
    I_lp = base.x[base.x.size - len(d.wh) * T:]  # остатки идут после потоков
    naive = float((d.r_handling[:, None] * X_lp).sum() + (d.r_truck[:, None] * Y_round).sum()
                  + (np.repeat(d.hold[None, :], T, 0).ravel() * I_lp).sum())

    # тот же MILP с допуском зазора по умолчанию (1e-4): сколько стоит доказательство оптимальности
    fast = solve(mt)

    # масштабирование: время MILP при росте горизонта (повтор недель), лимит 30 с
    scale = [{"weeks": T, "int_vars": int(mt.integrality.sum()), "lp_time_s": relax.time_s,
              "milp_time_s": milp.time_s, "milp_status": milp.status, "nodes": milp.info["mip_nodes"],
              "final_mip_gap_pct": 100 * milp.info["mip_gap"],
              "gap_to_lp_pct": 100 * (milp.objective / relax.objective - 1)}]
    for reps in (2, 3):
        dd = d.copy(); dd.weeks = T * reps
        dd.demand = np.tile(d.demand, reps); dd.deliv = np.tile(d.deliv, reps)
        dd.r_open = np.ones((nR, dd.weeks))
        if reps > 1:   # без повторных начальных остатков сеть не совместна — добавим их как поставку
            for k in range(1, reps):
                dd.deliv[:, k * T] += d.init
        mm = build_model(dd, trucks=True)
        lr = solve(mm, relax=True)
        rr = solve(mm, options={"mip_rel_gap": 0.0, "time_limit": 30.0}, keep_incumbent=True)
        scale.append({"weeks": dd.weeks, "int_vars": int(mm.integrality.sum()), "lp_time_s": lr.time_s,
                      "milp_time_s": rr.time_s, "milp_status": rr.status, "nodes": rr.info["mip_nodes"],
                      "final_mip_gap_pct": 100 * rr.info["mip_gap"],
                      "gap_to_lp_pct": 100 * (rr.objective / lr.objective - 1)})
    pd.DataFrame(scale).to_csv(RES / "milp_scaling.csv", index=False)

    per_arc = pd.DataFrame({"arc": [d.route_name(r) for r in range(nR)],
                            "lp_pallets": X_lp.sum(1), "milp_pallets": Xm.sum(1),
                            "lp_trucks_fractional": X_lp.sum(1) / d.truck_size,
                            "lp_trucks_rounded": Y_round.sum(1), "milp_trucks": Ym.sum(1)})
    per_arc.to_csv(RES / "lp_vs_milp_arcs.csv", index=False)
    used = Ym > 0.5
    return {"lp_relaxation": relax.objective, "lp_original": base.objective, "milp_objective": milp.objective,
            "milp_status": milp.status, "milp_time_s": milp.time_s, "lp_time_s": relax.time_s,
            "milp_default_gap_objective": fast.objective, "milp_default_gap_time_s": fast.time_s,
            "milp_default_gap_nodes": fast.info["mip_nodes"], "milp_default_gap_final": fast.info["mip_gap"],
            "mip_nodes": milp.info["mip_nodes"], "mip_gap": milp.info["mip_gap"],
            "integrality_gap_pct": 100 * (milp.objective / relax.objective - 1),
            "naive_rounding_cost": naive, "naive_vs_milp_pct": 100 * (naive / milp.objective - 1),
            "milp_trucks": float(Ym.sum()), "lp_trucks_fractional": float(X_lp.sum() / d.truck_size),
            "lp_trucks_rounded": float(Y_round.sum()),
            "milp_avg_load_pct": float(100 * Xm[used].sum() / (Ym[used].sum() * d.truck_size)),
            "lp_rounded_avg_load_pct": float(100 * X_lp.sum() / (Y_round.sum() * d.truck_size)),
            "milp_pallets_moved": float(Xm.sum()), "lp_pallets_moved": float(X_lp.sum()),
            "check_ok": bool(chk.ok.all()), "check_n": len(chk), "scaling": scale}


# ---------- графики ----------
def _graph(d):
    G = nx.DiGraph()
    for r in range(len(d.r_from)):
        G.add_edge(d.wh[d.r_from[r]], d.wh[d.r_to[r]], r=r)
    pos = {w: d.xy[i] for i, w in enumerate(d.wh)}
    return G, pos


def plots(d, X, inv, pi, sc, gx, summary):
    G, pos = _graph(d)
    cs = "arc3,rad=0.12"

    # схема сети
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color="#aed6f1", node_size=1300, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={w: f"{w}\nост {int(d.init[i])}\nвмест {int(d.cap[i])}"
                                            for i, w in enumerate(d.wh)}, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, connectionstyle=cs, arrowsize=12, node_size=1300, ax=ax, edge_color="#566573")
    lab = {(d.wh[d.r_from[r]], d.wh[d.r_to[r]]): f"{int(d.r_cap[r])} п/нед\n{d.r_cost[r]:.0f} ₽"
           for r in range(len(d.r_from))}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=lab, connectionstyle=cs, font_size=6, ax=ax)
    for i, w in enumerate(d.wh):
        dl = d.deliv[i]
        if dl.sum() > 0:
            ax.annotate("поставка: " + ", ".join(f"нед {t+1}: {int(v)}" for t, v in enumerate(dl) if v > 0),
                        pos[w], xytext=(0, -42), textcoords="offset points", ha="center", fontsize=7, color="#1e8449")
    ax.set_title("Сеть складов: пропускная способность (паллет/нед) и стоимость (₽/паллет)")
    ax.axis("off"); fig.tight_layout(); fig.savefig(RES / "network_scheme.png", dpi=130); plt.close(fig)

    # потоки по неделям
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for t, ax in enumerate(axes.ravel()):
        nx.draw_networkx_nodes(G, pos, node_color="#d6eaf8", node_size=1100, ax=ax)
        nx.draw_networkx_labels(G, pos, labels={w: f"{w}\nспрос {int(d.demand[i, t])}\nост {int(inv[(inv.warehouse == w) & (inv.week == t + 1)].stock_end.iloc[0])}"
                                                for i, w in enumerate(d.wh)}, font_size=6.5, ax=ax)
        nx.draw_networkx_edges(G, pos, connectionstyle=cs, node_size=1100, ax=ax, edge_color="#e5e8e8", arrowsize=8)
        used = [(d.wh[d.r_from[r]], d.wh[d.r_to[r]]) for r in range(len(d.r_from)) if X[r, t] > 1e-9]
        if used:
            rr = [G.edges[e]["r"] for e in used]
            sat = [abs(X[r, t] - d.r_cap[r]) < 1e-6 for r in rr]
            nx.draw_networkx_edges(G, pos, edgelist=used, connectionstyle=cs, node_size=1100, ax=ax, arrowsize=14,
                                   width=[1 + 5 * X[r, t] / d.r_cap.max() for r in rr],
                                   edge_color=["#c0392b" if s else "#2471a3" for s in sat])
            nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{X[G.edges[e]['r'], t]:.0f}/{d.r_cap[G.edges[e]['r']]:.0f}"
                                                              for e in used}, connectionstyle=cs, font_size=7, ax=ax)
        ax.set_title(f"Неделя {t+1}: поток/пропускная способность (красные — насыщены)")
        ax.axis("off")
    fig.tight_layout(); fig.savefig(RES / "flows_by_week.png", dpi=120); plt.close(fig)

    # остатки
    fig, ax = plt.subplots(figsize=(9, 5))
    for w in d.wh:
        s = inv[inv.warehouse == w]
        ax.plot([0] + list(s.week), [d.init[d.wh.index(w)]] + list(s.stock_end), "o-", label=w)
    ax.set_xlabel("неделя (0 — начало)"); ax.set_ylabel("остаток, паллет"); ax.set_xticks(range(d.weeks + 1))
    ax.set_title("Остатки на складах в оптимальном плане"); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(RES / "stock_base.png", dpi=130); plt.close(fig)

    # потенциалы
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pi, cmap="viridis")
    ax.set_xticks(range(d.weeks), [f"нед {t+1}" for t in range(d.weeks)]); ax.set_yticks(range(len(d.wh)), d.wh)
    for i in range(len(d.wh)):
        for t in range(d.weeks):
            ax.text(t, i, f"{pi[i, t]:.0f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, label="₽ за доп. паллету спроса")
    ax.set_title("Потенциалы узлов (двойственные переменные баланса)")
    fig.tight_layout(); fig.savefig(RES / "potentials.png", dpi=130); plt.close(fig)

    # сценарии
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ok = sc[sc.status == "Optimal"]
    ax.bar(ok.scenario, ok.objective_rub / 1e6, color="#2471a3")
    for _, r in sc[sc.status != "Optimal"].iterrows():
        ax.bar(r.scenario, 0, color="w")
        ax.text(r.scenario, 0.05, f"Infeasible\nдефицит ≥ {r.min_shortage_pallets:.0f} п.", ha="center", fontsize=8, color="#c0392b")
    for _, r in ok.iterrows():
        ax.text(r.scenario, r.objective_rub / 1e6, f"{r.objective_rub/1e6:.2f}\n(+{r.delta_pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("стоимость, млн ₽ за 4 недели"); ax.set_title("Сценарный анализ")
    ax.set_ylim(0, ok.objective_rub.max() / 1e6 * 1.2)
    fig.tight_layout(); fig.savefig(RES / "scenarios.png", dpi=130); plt.close(fig)

    # LP vs MILP по дугам
    la = pd.read_csv(RES / "lp_vs_milp_arcs.csv")
    la = la[(la.lp_pallets > 0) | (la.milp_pallets > 0)]
    fig, ax = plt.subplots(figsize=(12, 5))
    xi = np.arange(len(la))
    ax.bar(xi - 0.27, la.lp_trucks_fractional, 0.27, label="LP: дробные рейсы (x/33)")
    ax.bar(xi, la.lp_trucks_rounded, 0.27, label="LP + округление вверх")
    ax.bar(xi + 0.27, la.milp_trucks, 0.27, label="MILP: целые рейсы")
    ax.set_xticks(xi, la.arc, rotation=45, ha="right", fontsize=8); ax.set_ylabel("рейсов за 4 недели")
    ax.set_title("Число рейсов по дугам: LP против MILP"); ax.legend()
    fig.tight_layout(); fig.savefig(RES / "lp_vs_milp.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
