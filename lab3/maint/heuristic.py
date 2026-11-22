"""Жадная эвристика: единицы по очереди ставятся в самую выгодную неделю.

Порядок: сначала обязательные ремонты по возрастанию срока, затем остальные
по убыванию риска первой недели. Для каждой единицы перебираются допустимые
недели начала (лимит K, конфликты линий, часы бригады + подрядчик ≤ C_max)
и выбирается неделя с минимальным приростом затрат с учётом найма подрядчика.
Необязательная единица не ремонтируется, если любой вариант увеличивает затраты.
"""
import numpy as np


def greedy(d):
    nI, T = len(d.units), d.T
    order = sorted(range(nI), key=lambda i: (-d.mandatory[i], d.deadline[i] if d.mandatory[i] else 0, -d.rho[i, 0]))
    down = np.zeros((nI, T), dtype=int)
    load = np.zeros(T)
    start = [None] * nI

    def contractor_cost(ld):
        extra = np.maximum(0.0, ld - d.own)
        return float((extra * d.rate + (extra > 1e-9) * d.F).sum())

    for i in order:
        best = None
        for tau in range(T - d.p[i] + 1):
            if tau + d.p[i] > d.deadline[i]:
                continue
            wk = range(tau, tau + d.p[i])
            if any(down[:, t].sum() + 1 > d.K for t in wk):
                continue
            if any(down[j, t] for a, b in d.conflicts for j in ((b,) if a == i else (a,) if b == i else ()) for t in wk):
                continue
            new = load.copy(); new[list(wk)] += d.h[i]
            extra = new - d.own
            if np.any(extra > d.C_max * d.contractor_ok + 1e-9):
                continue
            delta = (d.c[i] + d.v[i] * d.p[i] - d.rho[i, tau:].sum() - d.post[i]
                     + contractor_cost(new) - contractor_cost(load))
            if best is None or delta < best[0]:
                best = (delta, tau)
        if best is None or (best[0] >= 0 and not d.mandatory[i]):
            continue
        tau = best[1]
        start[i] = tau
        down[i, tau:tau + d.p[i]] = 1
        load[tau:tau + d.p[i]] += d.h[i]
    hours = np.maximum(0.0, load - d.own)
    called = (hours > 1e-9).astype(float)
    cost = float(d.rho.sum() + d.post.sum())
    for i in range(nI):
        if start[i] is not None:
            cost += d.c[i] + d.v[i] * d.p[i] - d.rho[i, start[i]:].sum() - d.post[i]
    cost += float((called * d.F + hours * d.rate).sum())
    feasible = all(start[i] is not None for i in range(nI) if d.mandatory[i])
    return start, called, hours, cost, feasible
