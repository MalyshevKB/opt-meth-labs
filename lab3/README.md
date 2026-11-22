# ЛР №3. MILP-модель дискретного выбора — вариант 16

**План ремонтов оборудования по неделям с бюджетом бригад** (10 единиц, 8 недель).
Группа 527 (Q = 2), N = 15, Y = 26 → V = 16, seed = 10215.

Отчёт: [REPORT.md](REPORT.md).

Задание со звёздочкой по двум MILP-формулировкам описано в [formulation_star/](formulation_star/).

## Установка

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

(на Linux/macOS: `.venv/bin/python`). Проверено на Python 3.14.7, HiGHS 1.15.1 (`highspy`).

## Запуск

Одна команда запускает расчёт сценариев, эксперимент с big-M и масштабирование до 80 единиц.

```bash
.venv/Scripts/python run.py
```

Пример вывода:

```
Вариант 16, seed 10215. Базовый MILP: Optimal, 4,068,743 ₽; LP-релаксация 3,655,124 ₽, зазор 10.17%
Проверка: 102 условий, все выполнены: True
```

## Структура

| Путь | Назначение |
|---|---|
| `maint/data_gen.py` | генерация данных (seed 10215); `generate_frames(n)` даёт экземпляры любого размера |
| `maint/model.py` | загрузка данных, MILP-модель, обоснованное big-M, решение HiGHS (MILP и LP-релаксация) |
| `maint/verify.py` | независимая проверка плана, включая смысл связки big-M (без объекта солвера) |
| `maint/heuristic.py` | жадная эвристика вставки |
| `run.py` | базовое решение, LP-релаксация, сценарии, эксперимент с big-M, масштабирование, графики |
| `data/` | `units.csv`, `crew.csv`, `conflicts.csv`, `params.json` |
| `results/` | логи `highs_*.log`, планы и проверки (CSV), сценарии, графики, `summary.json` |

Основные результаты в `results/`:

- `schedule_base.csv`, `contractor_base.csv` — оптимальный план ремонтов и вызовы подрядчика;
- `check_base.csv` — независимая проверка (102 условия);
- `relaxation_fractional.csv`, `contractor_relaxation.csv` — LP-релаксация: дробные переменные;
- `scenarios.csv`, `schedule_by_scenario.csv`, `contractor_by_scenario.csv` — сценарии;
- `bigM_experiment.csv`, `contractor_presolve_off_1e9.csv` — выбор M и численная ловушка;
- `scaling.csv` — время и зазоры для 10, 20, 40, 80 единиц;
- `schedule_scenarios.png`, `scaling.png`, `bigM.png`.
