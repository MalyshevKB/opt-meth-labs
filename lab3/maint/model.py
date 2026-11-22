"""Данные, MILP-модель плана ремонтов и решение HiGHS."""
import copy
import json
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
    units: list
    T: int
    p: np.ndarray          # длительность ремонта, нед
    h: np.ndarray          # бригадо-часов в неделю ремонта
    c: np.ndarray          # стоимость ремонта (материалы), ₽
    v: np.ndarray          # потери от простоя, ₽/нед
    rho: np.ndarray        # I×T: ожидаемые потери от отказа за неделю работы без ремонта, ₽
    post: np.ndarray       # I: риск после горизонта, если ремонт не сделан, ₽
    mandatory: np.ndarray  # I, 0/1
    deadline: np.ndarray   # I, последняя неделя окончания ремонта
    own: np.ndarray        # T: часы собственной бригады
    conflicts: list        # пары индексов
    K: int                 # не более K единиц в ремонте одновременно
    C_max: float           # часы подрядчика в неделю
    rate: float            # ₽/час подрядчика
    F: float               # ₽ за выход подрядчика на неделю
    contractor_ok: np.ndarray = None  # T, 1 — подрядчик доступен

    def copy(self):
        return copy.deepcopy(self)


def from_frames(units: pd.DataFrame, crew: pd.DataFrame, conflicts: pd.DataFrame, params: dict) -> Data:
    T, H = params["weeks"], params["post_horizon_weeks"]
    g = units.prob_growth_per_week.to_numpy(float)[:, None]
    base = (units.failure_cost_rub * units.failure_prob_week1).to_numpy(float)[:, None]
    weeks = np.arange(T + H)[None, :]
    risk = base * g ** weeks                       # риск в неделю t = Цена отказа · P(отказ) · рост^(t-1)
    idx = {u: i for i, u in enumerate(units.unit)}
    return Data(
        units=list(units.unit), T=T, p=units.duration_weeks.to_numpy(int),
        h=units.crew_hours_per_week.to_numpy(float), c=units.repair_cost_rub.to_numpy(float),
        v=units.downtime_loss_rub_per_week.to_numpy(float), rho=risk[:, :T], post=risk[:, T:].sum(1),
        mandatory=units.mandatory.to_numpy(int), deadline=units.deadline_week.to_numpy(int),
        own=crew.own_crew_hours.to_numpy(float),
        conflicts=[(idx[a], idx[b]) for a, b in zip(conflicts.unit_a, conflicts.unit_b)],
        K=params["max_units_down"], C_max=params["contractor_max_hours"],
        rate=params["contractor_rate_rub_per_hour"], F=params["contractor_mobilization_rub"],
        contractor_ok=np.ones(T),
    )


def load_data(data_dir: Path) -> Data:
    params = json.loads((data_dir / "params.json").read_text(encoding="utf-8"))
    return from_frames(pd.read_csv(data_dir / "units.csv"), pd.read_csv(data_dir / "crew.csv"),
                       pd.read_csv(data_dir / "conflicts.csv"), params)


def tight_M(d: Data) -> np.ndarray:
    """Наименьшее обоснованное M_t для связки o_t <= M_t · w_t.

    Подрядчик нужен только сверх собственной бригады, а нагрузка недели не
    больше суммы K самых трудоёмких ремонтов (одновременно в ремонте ≤ K
    единиц). Кроме того, подрядчик не даёт больше C_max часов.
    """
    peak = np.sort(d.h)[::-1][:d.K].sum()
    return np.minimum(d.C_max, np.maximum(0.0, peak - d.own)) * d.contractor_ok


@dataclass
class Model:
    c: np.ndarray
    const: float
    A: sp.csc_matrix
    row_lo: np.ndarray
    row_hi: np.ndarray
    col_lo: np.ndarray
    col_hi: np.ndarray
    integrality: np.ndarray
    row_names: list
    col_names: list
    z_keys: list            # (i, τ) для каждой переменной z
    w0: int                 # индекс первой w_t
    o0: int                 # индекс первой o_t
    M: np.ndarray


def build_model(d: Data, M: np.ndarray = None) -> Model:
    """z[i,τ] ∈ {0,1} — ремонт единицы i начинается в неделю τ;
    w[t] ∈ {0,1} — подрядчик вызван на неделю t; o[t] ≥ 0 — часы подрядчика."""
    nI, T = len(d.units), d.T
    M = tight_M(d) if M is None else np.asarray(M, float) * d.contractor_ok
    z_keys = [(i, tau) for i in range(nI) for tau in range(T) if tau + d.p[i] <= T]
    nz = len(z_keys)
    w0, o0 = nz, nz + T
    n = nz + 2 * T
    c = np.zeros(n); lo = np.zeros(n); hi = np.ones(n); integ = np.zeros(n, dtype=int)
    names = [None] * n
    # целевая функция: константа = «никого не ремонтируем», z уменьшает риск и добавляет затраты ремонта
    const = float(d.rho.sum() + d.post.sum())
    for k, (i, tau) in enumerate(z_keys):
        c[k] = d.c[i] + d.v[i] * d.p[i] - d.rho[i, tau:].sum() - d.post[i]
        integ[k] = 1
        names[k] = f"z[{d.units[i]}|нед {tau + 1}]"
        if tau + d.p[i] > d.deadline[i]:                       # окончание позже срока предписания
            hi[k] = 0.0
    for t in range(T):
        c[w0 + t] = d.F; integ[w0 + t] = 1; names[w0 + t] = f"w[нед {t + 1}]"
        hi[w0 + t] = d.contractor_ok[t]
        c[o0 + t] = d.rate; hi[o0 + t] = INF; names[o0 + t] = f"o[нед {t + 1}]"

    rows, cols, vals, rlo, rhi, rn = [], [], [], [], [], []

    def add(entries, l, h_, name):
        i = len(rlo)
        for j, a in entries:
            rows.append(i); cols.append(j); vals.append(a)
        rlo.append(l); rhi.append(h_); rn.append(name)

    def down(i, t):  # переменные z, при которых единица i стоит в ремонте в неделю t
        return [k for k, (ii, tau) in enumerate(z_keys) if ii == i and tau <= t < tau + d.p[i]]

    for i in range(nI):  # (1) ремонт не более одного раза; обязательный — ровно один раз
        add([(k, 1.0) for k, (ii, _) in enumerate(z_keys) if ii == i],
            1.0 if d.mandatory[i] else -INF, 1.0, f"once[{d.units[i]}]")
    for t in range(T):
        # (2) часы бригад: собственные + подрядчик
        add([(k, d.h[i]) for i in range(nI) for k in down(i, t)] + [(o0 + t, -1.0)],
            -INF, d.own[t], f"crew[нед {t + 1}]")
        # (3) big-M: часы подрядчика только при его вызове
        add([(o0 + t, 1.0), (w0 + t, -M[t])], -INF, 0.0, f"bigM[нед {t + 1}]")
        # (4) не более K единиц в ремонте одновременно
        add([(k, 1.0) for i in range(nI) for k in down(i, t)], -INF, d.K, f"down[нед {t + 1}]")
        # (5) единицы одной линии не ремонтируются одновременно
        for a, b in d.conflicts:
            add([(k, 1.0) for k in down(a, t) + down(b, t)], -INF, 1.0,
                f"conflict[{d.units[a]}+{d.units[b]}|нед {t + 1}]")
    A = sp.csc_matrix((vals, (rows, cols)), shape=(len(rlo), n))
    return Model(c=c, const=const, A=A, row_lo=np.array(rlo), row_hi=np.array(rhi), col_lo=lo, col_hi=hi,
                 integrality=integ, row_names=rn, col_names=names, z_keys=z_keys, w0=w0, o0=o0, M=M)


@dataclass
class Result:
    status: str
    objective: float = np.nan
    x: np.ndarray = None
    time_s: float = np.nan
    info: dict = field(default_factory=dict)


def solve(m: Model, relax: bool = False, log_file: Path = None, options: dict = None) -> Result:
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
    lp.offset_ = m.const
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = m.A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = m.A.indices.astype(np.int32)
    lp.a_matrix_.value_ = m.A.data.astype(float)
    if not relax:
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
                           for v in m.integrality]
    h.passModel(lp)
    t0 = time.perf_counter(); h.run(); elapsed = time.perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus())
    if log_file is not None:
        h.setOptionValue("log_file", "")
        shutil.move(tmp_log, log_file)
    inf = h.getInfo()
    res = Result(status=status, time_s=elapsed,
                 info={"mip_nodes": inf.mip_node_count, "mip_gap": inf.mip_gap,
                       "mip_dual_bound": inf.mip_dual_bound, "simplex_iterations": inf.simplex_iteration_count})
    if inf.primal_solution_status == 2:            # есть допустимое решение (в т.ч. при лимите времени)
        res.objective = inf.objective_function_value
        res.x = np.array(h.getSolution().col_value)
    return res


def schedule_table(d: Data, m: Model, x: np.ndarray) -> pd.DataFrame:
    """Таблица решения: неделя начала ремонта каждой единицы (пусто — не ремонтируется)."""
    rows = []
    for i, u in enumerate(d.units):
        ks = [k for k, (ii, _) in enumerate(m.z_keys) if ii == i and x[k] > 0.5]
        start = m.z_keys[ks[0]][1] + 1 if ks else None
        rows.append({"unit": u, "start_week": start, "end_week": start + d.p[i] - 1 if ks else None,
                     "mandatory": int(d.mandatory[i]), "deadline_week": int(d.deadline[i])})
    return pd.DataFrame(rows)


def contractor_table(d: Data, m: Model, x: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"week": range(1, d.T + 1), "called": np.round(x[m.w0:m.w0 + d.T], 9),
                         "hours": x[m.o0:m.o0 + d.T]})
