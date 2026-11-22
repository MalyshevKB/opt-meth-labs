"""Решение полной двойственной задачи и проверка условий оптимальности."""
from pathlib import Path

import numpy as np
import pandas as pd

from zoo_lp.model import Model, build_model, load_data, solve

LAB = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
TOL = 1e-7


def canonical_form(model):
    """Получить Gx ≥ b, x ≥ 0, сохранив связь каждой строки с исходной границей."""
    if np.any(model.col_lo != 0):
        raise ValueError("Ожидаются нулевые нижние границы переменных.")

    rows, bounds, names = [], [], []
    for i, name in enumerate(model.row_names):
        if np.isfinite(model.row_lo[i]):
            rows.append(model.A[i])
            bounds.append(model.row_lo[i])
            names.append(f"{name}: lower")
        if np.isfinite(model.row_hi[i]):
            rows.append(-model.A[i])
            bounds.append(-model.row_hi[i])
            names.append(f"{name}: upper")

    for j in np.flatnonzero(np.isfinite(model.col_hi)):
        row = np.zeros_like(model.c)
        row[j] = -1
        rows.append(row)
        bounds.append(-model.col_hi[j])
        names.append(f"{model.col_names[j]}: upper")
    return np.array(rows), np.array(bounds), names


def build_dual(primal, matrix, rhs, names):
    # max b.T @ y при G.T @ y <= c, y >= 0. Решаем как min -b.T @ y.
    return Model(
        c=-rhs, A=matrix.T, row_lo=np.full_like(primal.c, -np.inf),
        row_hi=primal.c, col_lo=np.zeros_like(rhs), col_hi=np.full_like(rhs, np.inf),
        row_names=primal.col_names, row_group=["dual"] * len(primal.c), col_names=names,
    )


def main():
    RESULTS.mkdir(exist_ok=True)
    primal = build_model(load_data(LAB / "data"))
    matrix, rhs, names = canonical_form(primal)
    direct = solve(primal, ranging=False, log_file=RESULTS / "highs_primal.log")
    dual = solve(build_dual(primal, matrix, rhs, names), ranging=False,
                 log_file=RESULTS / "highs_dual.log")
    if direct.status != "Optimal" or dual.status != "Optimal":
        raise RuntimeError(f"Статусы задач: {direct.status}, {dual.status}")

    x, y = direct.x, dual.x
    primal_slack = matrix @ x - rhs
    dual_slack = primal.c - matrix.T @ y
    primal_objective, dual_objective = primal.c @ x, rhs @ y
    errors = {
        "primal_nonnegativity": max(0.0, -x.min()),
        "dual_nonnegativity": max(0.0, -y.min()),
        "primal_constraints": max(0.0, -primal_slack.min()),
        "dual_constraints": max(0.0, -dual_slack.min()),
        "row_complementarity": np.max(np.abs(y * primal_slack)),
        "column_complementarity": np.max(np.abs(x * dual_slack)),
        "primal_objective_recalculation": abs(primal_objective - direct.objective),
        "dual_objective_recalculation": abs(dual_objective + dual.objective),
        "duality_gap": abs(primal_objective - dual_objective),
    }
    pd.DataFrame({
        "constraint": names, "canonical_rhs": rhs, "primal_slack": primal_slack,
        "dual_value": y, "rhs_sensitivity": [
            -value if name.endswith(": upper") else value for name, value in zip(names, y)
        ],
    }).to_csv(RESULTS / "multipliers.csv", index=False, float_format="%.12g")
    pd.DataFrame({
        "variable": primal.col_names, "primal_value": x, "objective_coefficient": primal.c,
        "dual_activity": matrix.T @ y, "dual_slack": dual_slack,
    }).to_csv(RESULTS / "dual_constraints.csv", index=False, float_format="%.12g")
    checks = pd.Series(errors, name="absolute_error").rename_axis("check").reset_index()
    checks["tolerance"] = TOL
    checks["ok"] = checks.absolute_error.le(TOL)
    checks.to_csv(RESULTS / "checks.csv", index=False, float_format="%.12g")
    if not checks.ok.all():
        raise RuntimeError(checks.to_string(index=False))

    # Увеличиваем запас люцерны на 1 кг/сут и решаем прямую задачу повторно.
    i = primal.row_names.index("supply[Люцерна (сено)]")
    j = names.index(f"{primal.row_names[i]}: upper")
    old_supply = primal.row_hi[i]
    primal.row_hi[i] += 1.0
    changed = solve(primal, ranging=False, log_file=RESULTS / "highs_supply_plus_one.log")
    if changed.status != "Optimal":
        raise RuntimeError(f"Статус задачи с изменённым запасом: {changed.status}")
    predicted = -y[j]
    actual = changed.objective - primal_objective
    pd.DataFrame([{
        "old_supply_kg_day": old_supply, "new_supply_kg_day": primal.row_hi[i],
        "new_objective_rub_day": changed.objective,
        "predicted_change_rub_day": predicted, "actual_change_rub_day": actual,
        "absolute_error": abs(predicted - actual),
    }]).to_csv(RESULTS / "supply_plus_one.csv", index=False, float_format="%.12g")
    if abs(predicted - actual) > TOL:
        raise RuntimeError("Прогноз теневой цены не совпал с повторным решением.")

    print(f"Primal: {primal_objective:.10f}, dual: {dual_objective:.10f} RUB/day")
    print(f"Dual: {len(rhs)} variables, {len(primal.c)} constraints.")
    print(f"Checks passed: {len(checks)}. Supply +1: {actual:.6f} RUB/day.")


if __name__ == "__main__":
    main()
