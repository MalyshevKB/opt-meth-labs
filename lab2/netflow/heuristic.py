"""Жадная эвристика «сначала самые дешёвые маршруты» (для сравнения с ЛП).

Работает неделя за неделей, не заглядывая вперёд:
1. Для каждого склада считается прогнозный остаток после поставки и продаж недели.
2. Дефицит — то, что нужно, чтобы остаток не ушёл в минус (в последнюю неделю —
   до страхового запаса). Излишек — остаток сверх собственного спроса до конца
   горизонта и страхового запаса, а также всё, что не помещается на склад.
3. Пока есть дефицит, выбирается самая дешёвая пара «излишек → дефицит» по
   кратчайшему пути (Дейкстра по остаточным пропускным способностям),
   отправляется максимум возможного. Если излишков не хватило, во второй проход
   разрешается брать из текущего положительного остатка любых складов.
4. Затем вывозится всё, что не помещается на склад, в ближайшие склады с местом.
"""
import networkx as nx
import numpy as np


def _ship(G, resid, src_amt, dst_amt, cost_fn):
    """Повторяет: самая дешёвая пара источник→сток по кратчайшему пути."""
    moves = []
    while True:
        best = None
        H = nx.DiGraph([(a, b, {"w": w}) for (a, b), w in cost_fn.items() if resid[(a, b)] > 1e-9])
        H.add_nodes_from(G.nodes)
        for s in [k for k, v in src_amt.items() if v > 1e-9]:
            dist, path = nx.single_source_dijkstra(H, s, weight="w")
            for t in [k for k, v in dst_amt.items() if v > 1e-9 and k != s]:
                if t in dist and (best is None or dist[t] < best[0]):
                    best = (dist[t], s, t, path[t])
        if best is None:
            return moves
        _, s, t, path = best
        arcs = list(zip(path[:-1], path[1:]))
        q = min(src_amt[s], dst_amt[t], min(resid[a] for a in arcs))
        for a in arcs:
            resid[a] -= q
        src_amt[s] -= q; dst_amt[t] -= q
        moves.append((arcs, q))


def greedy(d):
    nW, T, nR = len(d.wh), d.weeks, len(d.r_from)
    arc_id = {(int(d.r_from[r]), int(d.r_to[r])): r for r in range(nR)}
    cost = {a: d.r_cost[r] for a, r in arc_id.items()}
    G = nx.DiGraph(list(arc_id))
    x = np.zeros((nR, T)); I = np.zeros((nW, T)); short = np.zeros((nW, T))
    prev = d.init.copy()
    for t in range(T):
        resid = {a: d.r_cap[r] * d.r_open[r, t] for a, r in arc_id.items()}
        avail = prev + d.deliv[:, t] - d.demand[:, t]
        floor = d.safety if t == T - 1 else np.zeros(nW)
        future = d.demand[:, t + 1:].sum(1) + d.safety
        need = {w: max(0.0, floor[w] - avail[w]) for w in range(nW)}
        surplus = {w: max(0.0, avail[w] - future[w], avail[w] - d.cap[w]) if need[w] == 0 else 0.0
                   for w in range(nW)}
        moves = _ship(G, resid, surplus, need, cost)
        if any(v > 1e-9 for v in need.values()):          # второй проход: берём из любых остатков
            spare = {w: max(0.0, avail[w] - floor[w] - sum(q for arcs, q in moves if arcs[0][0] == w))
                     if need[w] == 0 else 0.0 for w in range(nW)}
            moves += _ship(G, resid, spare, need, cost)
        for arcs, q in moves:
            for a in arcs:
                x[arc_id[a], t] += q
        net = avail + np.array([x[[arc_id[a] for a in arc_id if a[1] == w], t].sum()
                                - x[[arc_id[a] for a in arc_id if a[0] == w], t].sum() for w in range(nW)])
        over = {w: max(0.0, net[w] - d.cap[w]) for w in range(nW)}   # не помещается на склад
        room = {w: max(0.0, d.cap[w] - net[w]) for w in range(nW)}
        for arcs, q in _ship(G, resid, over, room, cost):
            for a in arcs:
                x[arc_id[a], t] += q
            net[arcs[0][0]] -= q; net[arcs[-1][1]] += q
        short[:, t] = np.maximum(0.0, floor - net)                    # непокрытый дефицит
        I[:, t] = net + short[:, t]
        prev = I[:, t]
    cost_total = float((d.r_cost[:, None] * x).sum() + (d.hold[:, None] * I).sum())
    return x, I, short, cost_total
