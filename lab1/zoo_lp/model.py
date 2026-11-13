"""Загрузка данных, построение ЛП-модели рациона и решение HiGHS."""
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

from .data_gen import NUTRIENTS

INF = highspy.kHighsInf


@dataclass
class Data:
    feeds: list            # F
    animals: list          # A
    category: np.ndarray   # F, "plant"/"animal"
    cost: np.ndarray       # F, ₽/кг
    supply: np.ndarray     # F, кг/сут на весь зоопарк
    nutr: np.ndarray       # F×K, содержание нутриента в 1 кг
    heads: np.ndarray      # A
    max_intake: np.ndarray       # A, кг/гол/сут
    max_share: np.ndarray        # A, макс. доля одного корма
    min_anim_share: np.ndarray   # A
    max_anim_share: np.ndarray   # A
    norm_lo: np.ndarray    # A×K (NaN = нет)
    norm_hi: np.ndarray    # A×K (NaN = нет)
    allowed: np.ndarray    # A×F, 0/1

    def copy(self):
        return copy.deepcopy(self)


def load_data(data_dir: Path) -> Data:
    feeds = pd.read_csv(data_dir / "feeds.csv")
    animals = pd.read_csv(data_dir / "animals.csv")
    norms = pd.read_csv(data_dir / "norms.csv")
    compat = pd.read_csv(data_dir / "compat.csv")
    F, A = list(feeds.feed), list(animals.animal)
    lo = norms.pivot(index="animal", columns="nutrient", values="min").loc[A, NUTRIENTS]
    hi = norms.pivot(index="animal", columns="nutrient", values="max").loc[A, NUTRIENTS]
    allowed = compat.pivot(index="animal", columns="feed", values="allowed").loc[A, F]
    return Data(
        feeds=F, animals=A, category=feeds.category.to_numpy(),
        cost=feeds.cost_rub_per_kg.to_numpy(float), supply=feeds.supply_kg_per_day.to_numpy(float),
        nutr=feeds[NUTRIENTS].to_numpy(float), heads=animals.heads.to_numpy(float),
        max_intake=animals.max_intake_kg.to_numpy(float), max_share=animals.max_share_single.to_numpy(float),
        min_anim_share=animals.min_animal_share.to_numpy(float),
        max_anim_share=animals.max_animal_share.to_numpy(float),
        norm_lo=lo.to_numpy(float), norm_hi=hi.to_numpy(float), allowed=allowed.to_numpy(int),
    )


@dataclass
class Model:
    """ЛП в форме HiGHS: min c'x, row_lo <= A x <= row_hi, col_lo <= x <= col_hi."""
    c: np.ndarray
    A: np.ndarray
    row_lo: np.ndarray
    row_hi: np.ndarray
    col_lo: np.ndarray
    col_hi: np.ndarray
    row_names: list
    row_group: list
    col_names: list


def var_index(d: Data, a: int, f: int) -> int:
    return a * len(d.feeds) + f


def build_model(d: Data) -> Model:
    nA, nF, nK = len(d.animals), len(d.feeds), len(NUTRIENTS)
    n = nA * nF
    c = np.zeros(n)
    col_hi = np.full(n, INF)
    col_names = []
    for a in range(nA):
        for f in range(nF):
            j = var_index(d, a, f)
            c[j] = d.heads[a] * d.cost[f]          # ₽/сут за 1 кг/гол
            if not d.allowed[a, f]:
                col_hi[j] = 0.0                     # корм запрещён для вида
            col_names.append(f"x[{d.animals[a]}|{d.feeds[f]}]")

    rows, lo, hi, names, groups = [], [], [], [], []

    def add(coef, l, h, name, group):
        rows.append(coef); lo.append(l); hi.append(h); names.append(name); groups.append(group)

    is_anim = (d.category == "animal").astype(float)
    for a in range(nA):
        sl = slice(a * nF, (a + 1) * nF)
        # (1) нормы нутриентов на голову
        for k in range(nK):
            l, h = d.norm_lo[a, k], d.norm_hi[a, k]
            if np.isnan(l) and np.isnan(h):
                continue
            row = np.zeros(n); row[sl] = d.nutr[:, k]
            add(row, -INF if np.isnan(l) else l, INF if np.isnan(h) else h,
                f"norm[{d.animals[a]}|{NUTRIENTS[k]}]", "norm")
        # (2) вместимость желудка: суммарная масса рациона
        row = np.zeros(n); row[sl] = 1.0
        add(row, -INF, d.max_intake[a], f"intake[{d.animals[a]}]", "intake")
        # (3) разнообразие: каждый корм не более s_a от массы рациона
        for f in range(nF):
            if not d.allowed[a, f]:
                continue
            row = np.zeros(n); row[sl] = -d.max_share[a]; row[var_index(d, a, f)] += 1.0
            add(row, -INF, 0.0, f"variety[{d.animals[a]}|{d.feeds[f]}]", "variety")
        # (4) доля кормов животного происхождения в [p_min, p_max]
        if is_anim @ d.allowed[a] > 0:
            row = np.zeros(n); row[sl] = is_anim - d.max_anim_share[a]
            add(row, -INF, 0.0, f"anim_max[{d.animals[a]}]", "anim_share")
            if d.min_anim_share[a] > 0:
                row = np.zeros(n); row[sl] = is_anim - d.min_anim_share[a]
                add(row, 0.0, INF, f"anim_min[{d.animals[a]}]", "anim_share")
    # (5) запас корма на складе зоопарка, кг/сут
    for f in range(nF):
        row = np.zeros(n)
        for a in range(nA):
            row[var_index(d, a, f)] = d.heads[a]
        add(row, -INF, d.supply[f], f"supply[{d.feeds[f]}]", "supply")

    return Model(c=c, A=np.array(rows), row_lo=np.array(lo), row_hi=np.array(hi),
                 col_lo=np.zeros(n), col_hi=col_hi, row_names=names, row_group=groups,
                 col_names=col_names)


@dataclass
class Result:
    status: str
    objective: float = np.nan
    x: np.ndarray = None
    row_act: np.ndarray = None
    row_dual: np.ndarray = None
    col_dual: np.ndarray = None
    iterations: dict = field(default_factory=dict)
    time_s: float = np.nan
    ranging: object = None


def to_highs_lp(m: Model) -> highspy.HighsLp:
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = len(m.c), len(m.row_lo)
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = m.c, m.col_lo, m.col_hi
    lp.row_lower_, lp.row_upper_ = m.row_lo, m.row_hi
    # A хранится по столбцам (CSC)
    starts, idx, vals = [0], [], []
    for j in range(m.A.shape[1]):
        nz = np.nonzero(m.A[:, j])[0]
        idx.extend(nz); vals.extend(m.A[nz, j]); starts.append(len(idx))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.array(starts, dtype=np.int32)
    lp.a_matrix_.index_ = np.array(idx, dtype=np.int32)
    lp.a_matrix_.value_ = np.array(vals, dtype=float)
    return lp


def solve(m: Model, solver: str = "simplex", log_file: Path = None,
          presolve: str = "on", ranging: bool = True) -> Result:
    h = highspy.Highs()
    h.setOptionValue("output_flag", log_file is not None)
    h.setOptionValue("log_to_console", False)
    if log_file is not None:
        # HiGHS (C++) не открывает пути с не-ASCII символами под Windows,
        # поэтому лог пишется во временный файл и затем переносится
        fd, tmp_log = tempfile.mkstemp(suffix=".log")
        os.close(fd)
        h.setOptionValue("log_file", tmp_log)
    h.setOptionValue("solver", solver)       # "simplex" (двойственный) или "ipm"
    h.setOptionValue("presolve", presolve)
    h.passModel(to_highs_lp(m))
    t0 = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus())
    if log_file is not None:
        h.setOptionValue("log_file", "")
        shutil.move(tmp_log, log_file)
    info = h.getInfo()
    res = Result(status=status, time_s=elapsed,
                 iterations={"simplex": info.simplex_iteration_count,
                             "ipm": info.ipm_iteration_count,
                             "crossover": info.crossover_iteration_count})
    if status == "Optimal":
        sol = h.getSolution()
        res.objective = info.objective_function_value
        res.x = np.array(sol.col_value)
        res.row_act = np.array(sol.row_value)
        res.row_dual = np.array(sol.row_dual)
        res.col_dual = np.array(sol.col_dual)
        if ranging:
            st, rng = h.getRanging()
            if st == highspy.HighsStatus.kOk:
                res.ranging = rng
    return res


def subset(m: Model, rows) -> Model:
    rows = list(rows)
    return Model(c=m.c, A=m.A[rows], row_lo=m.row_lo[rows], row_hi=m.row_hi[rows],
                 col_lo=m.col_lo, col_hi=m.col_hi, row_names=[m.row_names[i] for i in rows],
                 row_group=[m.row_group[i] for i in rows], col_names=m.col_names)


def deletion_filter(m: Model) -> list:
    """Неприводимая несовместная подсистема (IIS) методом удаления.

    Строки по одной выбрасываются; если без строки система всё ещё
    несовместна, строка не нужна для конфликта и удаляется окончательно.
    Границы переменных (x >= 0, запреты кормов) считаются жёсткими.
    """
    keep = list(range(len(m.row_lo)))
    for i in list(keep):
        trial = [j for j in keep if j != i]
        if solve(subset(m, trial), ranging=False).status == "Infeasible":
            keep = trial
    return [m.row_names[i] for i in keep]


def elastic(m: Model, groups=None) -> tuple:
    """Эластичная задача: минимум суммарного относительного нарушения ограничений.

    К строкам из групп `groups` (по умолчанию ко всем) добавляются
    неотрицательные невязки e+ / e- в единицах |правой части|, остальные
    строки и все границы переменных остаются жёсткими. Ненулевые невязки
    показывают, какие ограничения приходится ослабить и насколько.
    """
    nR, n = m.A.shape
    scale = np.maximum(1.0, np.abs(np.where(np.isfinite(m.row_hi), m.row_hi,
                                            np.where(np.isfinite(m.row_lo), m.row_lo, 1.0))))
    soft = np.array([groups is None or g in groups for g in m.row_group])
    low_rows = np.where(np.isfinite(m.row_lo) & soft)[0]
    up_rows = np.where(np.isfinite(m.row_hi) & soft)[0]
    E_lo = np.zeros((nR, len(low_rows))); E_lo[low_rows, range(len(low_rows))] = scale[low_rows]
    E_up = np.zeros((nR, len(up_rows))); E_up[up_rows, range(len(up_rows))] = -scale[up_rows]
    ne = len(low_rows) + len(up_rows)
    em = Model(c=np.concatenate([np.zeros(n), np.ones(ne)]),
               A=np.hstack([m.A, E_lo, E_up]), row_lo=m.row_lo, row_hi=m.row_hi,
               col_lo=np.zeros(n + ne), col_hi=np.concatenate([m.col_hi, np.full(ne, INF)]),
               row_names=m.row_names, row_group=m.row_group, col_names=None)
    r = solve(em, ranging=False)
    if r.status != "Optimal":
        return np.nan, []
    e = r.x[n:]
    viol = []
    for i, val in zip(low_rows, e[:len(low_rows)]):
        if val > 1e-9:
            viol.append((m.row_names[i], "нижняя граница", m.row_lo[i], val * scale[i]))
    for i, val in zip(up_rows, e[len(low_rows):]):
        if val > 1e-9:
            viol.append((m.row_names[i], "верхняя граница", m.row_hi[i], val * scale[i]))
    return r.objective, viol
