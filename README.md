# HELIX — position-sizing diagnostics & trading-bot architecture

A research project in two parts:

1. **`helix_diagnostics.py`** — a runnable diagnostic tool for a crypto
   position-sizing formula. It answers one question with numbers: does *binary
   Kelly* mis-size versus sizing on the real payoff distribution, and does the
   meta-model actually have an edge or is it fitting noise?
2. **`docs/`** — the full system architecture for a production-grade automated
   trading bot, plus the position-sizing formula the diagnostic tool tests.

> **Architecture and research only.** This repository contains no live trading
> strategy, no execution engine, and no API keys. It is for backtesting and
> research. Nothing here is financial advice.

## Why this exists

The formula uses binary Kelly: `f = p - (1 - p) / b`. But triple-barrier trades
have three outcomes (take-profit, stop, timeout). A timeout exits at market with
a P&L between `-1` and `+b`, so the payoff is **not** the binary `{+b, -1}` that
Kelly assumes. When timeouts are frequent or exit far from a full stop, binary
Kelly mis-sizes (usually *over*-sizes). This tool measures exactly when that
happens and by how much.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Run the synthetic self-test

```bash
python helix_diagnostics.py
```

Runs the four checks on a built-in synthetic ledger so you can see the output
shape end to end.

## Run on your own trades

Build a pandas DataFrame with one row per closed trade:

| column        | meaning                                                          |
| ------------- | ---------------------------------------------------------------- |
| `R`           | realized return in **risk units** (full stop = `-1.0`, TP at reward:risk `b` = `+b`; i.e. `pnl / risk_per_trade`) |
| `exit_reason` | one of `"tp"`, `"sl"`, `"timeout"` (case-insensitive)            |
| `t0`, `t1`    | integer bar indices: entry, and label resolution (label end)     |
| `p`           | optional meta-model win probability at entry                     |

```python
import pandas as pd
from helix_diagnostics import analyze_timeout_structure

ledger = pd.read_csv("my_trades.csv")
print(analyze_timeout_structure(ledger, b_design=2.0).summary())
```

See the `_demo()` function in `helix_diagnostics.py` for a full worked example
including the zero-alpha gate and purged walk-forward.

## The four checks

1. **Timeout structure & the binary-vs-empirical Kelly sizing gap** — the crux.
   Reports timeout rate, the signed sizing direction (OVER = danger, UNDER =
   safe), the log-growth gap and ratio, the structural trigger, and a warning
   when uncapped raw Kelly exceeds the `2.0` cap (an instability signal).
2. **Calibration of `p`** — Brier score, ECE, reliability bins. Kelly reacts to
   the absolute value of `p`, so calibration matters.
3. **Zero-alpha gate** — replace the alpha feature with noise; the model must
   lose both its selection lift *and* its p-spread. Uses purged, forward-only
   out-of-fold predictions.
4. **Purged, embargoed walk-forward splits** — López de Prado purge plus a real
   pre-test embargo gap, for honest out-of-sample evaluation.

## Design principles carried through the docs

- **Research–live parity.** The same strategy code runs in backtest, paper, and
  live. The barriers that define reward:risk `b` are the exact barriers the
  executor would enforce live, so backtest odds equal live odds.
- **Keep all preprocessing inside CV folds.** Never evaluate Kelly or the
  zero-alpha test in-sample.
- **Key hygiene (documented, not implemented here).** Trade-only API
  permissions, withdrawals disabled, one scoped key per venue per instance,
  stored in a secrets manager, never in source. See
  `docs/trading-bot-architecture.md`.

## Layout

```
helix_diagnostics.py            # the runnable diagnostic tool
requirements.txt
docs/
  trading-bot-architecture.md   # full event-driven system architecture
  helix-unified-formula.md      # the position-sizing formula under test
```

## License

MIT — see `LICENSE`.
