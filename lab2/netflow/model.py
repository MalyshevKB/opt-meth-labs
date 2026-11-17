"""Загрузка данных, построение модели потока минимальной стоимости в
развёрнутой по времени сети и решение HiGHS (LP и MILP)."""
import copy
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import highspy
import numpy as np
import pandas as pd
import scipy.sparse as sp

INF = highspy.kHighsInf


@dataclass
class Data:
    wh: list                 # W — склады
    weeks: int               # T
    init: np.ndarray         # W, начальный остаток, паллет
    cap: np.ndarray          # W, вместимость склада, паллет
    hold: np.ndarray         # W, хранение, ₽/паллет/нед
    safety: np.ndarray       # W, страховой остаток на конец горизонта
    xy: np.ndarray           # W×2, координаты, км
    demand: np.ndarray       # W×T, паллет/нед
    deliv: np.ndarray        # W×T, поставки производителя, паллет/нед
    r_from: np.ndarray       # R — индексы склада-отправителя
    r_to: np.ndarray         # R
    r_cost: np.ndarray       # R, ₽/паллет (линейная аппроксимация стоимости рейса)
    r_cap: np.ndarray        # R, паллет/нед
    r_truck: np.ndarray      # R, ₽ за рейс фуры
    r_handling: np.ndarray   # R, ₽/паллет
    r_trucks_max: np.ndarray  # R, рейсов/нед
    r_open: np.ndarray       # R×T, 1 — дуга доступна
    truck_size: int = 33

    def copy(self):
        return copy.deepcopy(self)

    def route_name(self, r):
        return f"{self.wh[self.r_from[r]]}→{self.wh[self.r_to[r]]}"


def load_data(data_dir: Path) -> Data:
    import json
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    wh = pd.read_csv(data_dir / "warehouses.csv")
    routes = pd.read_csv(data_dir / "routes.csv")
    dem = pd.read_csv(data_dir / "demand.csv")
    dlv = pd.read_csv(data_dir / "deliveries.csv")
    W, T = list(wh.warehouse), meta["weeks"]
    idx = {w: i for i, w in enumerate(W)}
    D = dem.pivot(index="warehouse", columns="week", values="demand_pallets").loc[W, range(1, T + 1)]
    R = np.zeros((len(W), T))
    for _, r in dlv.iterrows():
        R[idx[r.warehouse], r.week - 1] += r.qty_pallets
    return Data(
        wh=W, weeks=T, init=wh.initial_stock.to_numpy(float), cap=wh.storage_cap.to_numpy(float),
        hold=wh.holding_rub_per_pallet_week.to_numpy(float), safety=wh.safety_stock_end.to_numpy(float),
        xy=wh[["x_km", "y_km"]].to_numpy(float), demand=D.to_numpy(float), deliv=R,
        r_from=routes["from"].map(idx).to_numpy(), r_to=routes["to"].map(idx).to_numpy(),
        r_cost=routes.cost_rub_per_pallet.to_numpy(float), r_cap=routes.cap_pallets_per_week.to_numpy(float),
        r_truck=routes.truck_cost_rub.to_numpy(float), r_handling=routes.handling_rub_per_pallet.to_numpy(float),
        r_trucks_max=routes.max_trucks_per_week.to_numpy(float),
        r_open=np.ones((len(routes), T)), truck_size=meta["pallets_per_truck"],
    )


@dataclass
class Model:
    c: np.ndarray
    A: sp.csc_matrix
    row_lo: np.ndarray
    row_hi: np.ndarray
    col_lo: np.ndarray
    col_hi: np.ndarray
    integrality: np.ndarray
    row_names: list
    col_names: list
    # срезы групп переменных
    n_x: int = 0
    n_I: int = 0
    n_y: int = 0
    n_s: int = 0


def x_idx(d: Data, r, t):  # перевозка по дуге r в неделю t
    return t * len(d.r_from) + r


def build_model(d: Data, trucks: bool = False, shortage_penalty: float = None) -> Model:
    """Узлы (w, t): склад w в неделю t. Переменные:
    x[r,t] — паллет по дуге r в неделю t; I[w,t] — остаток в конце недели t;
    y[r,t] — число рейсов (только trucks=True, целые);
    s[w,t] — неудовлетворённый спрос (только при shortage_penalty, эластичная версия).
    Баланс узла: I[w,t-1] + поставка + Σвход x − Σвыход x − I[w,t] (+ s) = спрос.
    """
    nW, T, nR = len(d.wh), d.weeks, len(d.r_from)
    n_x, n_I = nR * T, nW * T
    n_y = n_x if trucks else 0
    n_s = n_I if shortage_penalty is not None else 0
    n = n_x + n_I + n_y + n_s
    I0 = n_x
    def I_idx(w, t): return I0 + t * nW + w
    Y0, S0 = n_x + n_I, n_x + n_I + n_y

    c = np.zeros(n); lo = np.zeros(n); hi = np.full(n, INF); integ = np.zeros(n, dtype=int)
    names = [None] * n
    for t in range(T):
        for r in range(nR):
            j = x_idx(d, r, t)
            names[j] = f"x[{d.route_name(r)}|w{t+1}]"
            hi[j] = d.r_cap[r] * d.r_open[r, t]
            if trucks:
                c[j] = d.r_handling[r]
                integ[j] = 1                                  # паллеты неделимы
                k = Y0 + j
                c[k] = d.r_truck[r]; hi[k] = d.r_trucks_max[r] * d.r_open[r, t]; integ[k] = 1
                names[k] = f"y[{d.route_name(r)}|w{t+1}]"
            else:
                c[j] = d.r_cost[r]
        for w in range(nW):
            j = I_idx(w, t)
            names[j] = f"I[{d.wh[w]}|w{t+1}]"
            c[j] = d.hold[w]; hi[j] = d.cap[w]
            if t == T - 1:
                lo[j] = d.safety[w]
            if n_s:
                k = S0 + t * nW + w
                c[k] = shortage_penalty; names[k] = f"short[{d.wh[w]}|w{t+1}]"

    rows, cols, vals = [], [], []
    rlo, rhi, rnames = [], [], []
    for t in range(T):
        for w in range(nW):
            i = len(rlo)
            for r in np.where(d.r_to == w)[0]:
                rows.append(i); cols.append(x_idx(d, r, t)); vals.append(1.0)
            for r in np.where(d.r_from == w)[0]:
                rows.append(i); cols.append(x_idx(d, r, t)); vals.append(-1.0)
            rows.append(i); cols.append(I_idx(w, t)); vals.append(-1.0)
            rhs = d.demand[w, t] - d.deliv[w, t]
            if t > 0:
                rows.append(i); cols.append(I_idx(w, t - 1)); vals.append(1.0)
            else:
                rhs -= d.init[w]
            if n_s:
                rows.append(i); cols.append(S0 + t * nW + w); vals.append(1.0)
            rlo.append(rhs); rhi.append(rhs); rnames.append(f"balance[{d.wh[w]}|w{t+1}]")
    if trucks:  # x[r,t] <= размер фуры · y[r,t]
        for j in range(n_x):
            i = len(rlo)
            rows += [i, i]; cols += [j, Y0 + j]; vals += [1.0, -float(d.truck_size)]
            rlo.append(-INF); rhi.append(0.0); rnames.append(f"truck[{names[j][2:-1]}]")
    A = sp.csc_matrix((vals, (rows, cols)), shape=(len(rlo), n))
    return Model(c=c, A=A, row_lo=np.array(rlo), row_hi=np.array(rhi), col_lo=lo, col_hi=hi,
                 integrality=integ if trucks else np.zeros(n, dtype=int), row_names=rnames,
                 col_names=names, n_x=n_x, n_I=n_I, n_y=n_y, n_s=n_s)


@dataclass
class Result:
    status: str
    objective: float = np.nan
    x: np.ndarray = None
    row_dual: np.ndarray = None
    col_dual: np.ndarray = None
    time_s: float = np.nan
    info: dict = field(default_factory=dict)
    ranging: object = None


def solve(m: Model, log_file: Path = None, mip: bool = None, relax: bool = False,
          ranging: bool = False, options: dict = None, keep_incumbent: bool = False) -> Result:
    """keep_incumbent: при остановке MILP по лимиту времени вернуть лучшее найденное решение."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", log_file is not None)
    h.setOptionValue("log_to_console", False)
    if log_file is not None:
        # HiGHS не открывает пути с не-ASCII символами под Windows — пишем во временный файл
        fd, tmp_log = tempfile.mkstemp(suffix=".log"); os.close(fd)
        h.setOptionValue("log_file", tmp_log)
    for k, v in (options or {}).items():
        h.setOptionValue(k, v)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = len(m.c), len(m.row_lo)
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = m.c, m.col_lo, m.col_hi
    lp.row_lower_, lp.row_upper_ = m.row_lo, m.row_hi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = m.A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = m.A.indices.astype(np.int32)
    lp.a_matrix_.value_ = m.A.data.astype(float)
    is_mip = m.integrality.any() and not relax if mip is None else mip
    if is_mip:
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
                           for v in m.integrality]
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus())
    if log_file is not None:
        h.setOptionValue("log_file", "")
        shutil.move(tmp_log, log_file)
    inf = h.getInfo()
    res = Result(status=status, time_s=elapsed,
                 info={"simplex_iterations": inf.simplex_iteration_count,
                       "mip_nodes": inf.mip_node_count, "mip_gap": inf.mip_gap,
                       "mip_dual_bound": inf.mip_dual_bound})
    if status == "Optimal" or (keep_incumbent and inf.primal_solution_status == 2):
        s = h.getSolution()
        res.objective = inf.objective_function_value
        res.x = np.array(s.col_value)
        if not is_mip:
            res.row_dual = np.array(s.row_dual)
            res.col_dual = np.array(s.col_dual)
            if ranging:
                st, rg = h.getRanging()
                if st == highspy.HighsStatus.kOk:
                    res.ranging = rg
    return res


def flows_table(d: Data, m: Model, x: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in range(d.weeks):
        for r in range(len(d.r_from)):
            j = x_idx(d, r, t)
            row = {"from": d.wh[d.r_from[r]], "to": d.wh[d.r_to[r]], "week": t + 1,
                   "flow_pallets": x[j], "cap_pallets": d.r_cap[r] * d.r_open[r, t]}
            if m.n_y:
                row["trucks"] = x[m.n_x + m.n_I + j]
            rows.append(row)
    return pd.DataFrame(rows)


def inventory_table(d: Data, m: Model, x: np.ndarray) -> pd.DataFrame:
    nW = len(d.wh)
    rows = [{"warehouse": d.wh[w], "week": t + 1, "stock_end": x[m.n_x + t * nW + w],
             **({"shortage": x[m.n_x + m.n_I + m.n_y + t * nW + w]} if m.n_s else {})}
            for t in range(d.weeks) for w in range(nW)]
    return pd.DataFrame(rows)
