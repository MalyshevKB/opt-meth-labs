# ЛР №2. Транспортные и сетевые модели — вариант 16

**Балансировка запасов между складами розничной сети** (8 складов, горизонт 4 недели).
Группа 527 (Q = 2), N = 15, Y = 26 → V = 16, seed = 10215.
Дополнительное задание: целочисленная версия (неделимые рейсы фур), сравнение LP и MILP.

Отчёт: [REPORT.md](REPORT.md).

## Установка

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

(на Linux/macOS: `.venv/bin/python`). Проверено на Python 3.14.7, HiGHS 1.15.1 (`highspy`), NetworkX 3.7.

## Запуск

Одна команда (≈ 2 минуты, из них ~1,5 минуты — MILP и тест масштабирования):

```bash
.venv/Scripts/python run.py
```

Пример вывода:

```
Вариант 16, seed 10215. Базовый: Optimal, 1,359,054 ₽ за 4 недели
Проверка: 297 условий, все выполнены: True
Макс. рост спроса без дефицита: 9.10%
Жадная: дефицит 287 паллет, стоимость с экстренными поставками 1,735,166 ₽ (+27.7% к ЛП)
MILP: 1,379,831 ₽, зазор к LP-релаксации 1.53%, рейсов 93
```

## Структура

| Путь | Назначение |
|---|---|
| `netflow/data_gen.py` | генерация сети, спроса, поставок и остатков (seed 10215) |
| `netflow/model.py` | загрузка данных, модель потока в развёрнутой по времени сети (LP, MILP, версия с дефицитом), решение HiGHS |
| `netflow/verify.py` | независимая проверка баланса в каждой вершине и границ на каждой дуге (без объекта солвера) |
| `netflow/heuristic.py` | жадная эвристика «сначала самые дешёвые маршруты» |
| `run.py` | сценарии, узкие места, потенциалы, сравнение с эвристикой, MILP, графики |
| `data/` | `warehouses.csv`, `routes.csv`, `demand.csv`, `deliveries.csv`, `meta.json` |
| `results/` | логи `highs_*.log`, решения и проверки (CSV), сценарии, графики (PNG), `summary.json` |

Основные результаты в `results/`:

- `flows_base.csv`, `stock_base.csv` — оптимальный план перевозок и остатки;
- `check_base.csv`, `check_milp.csv` — независимая проверка;
- `arcs_base.csv`, `bottlenecks.csv`, `potentials.csv` — загрузка дуг, узкие места, потенциалы узлов;
- `scenarios.csv`, `flows_by_scenario.csv`, `shortage_demand.csv` — сценарный анализ;
- `flows_greedy.csv`, `stock_greedy.csv` — решение жадной эвристики;
- `flows_milp.csv`, `lp_vs_milp_arcs.csv`, `milp_scaling.csv`, `highs_milp.log` — дополнительное задание;
- `network_scheme.png`, `flows_by_week.png`, `stock_base.png`, `potentials.png`, `scenarios.png`, `lp_vs_milp.png`.
