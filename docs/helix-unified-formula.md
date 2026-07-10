# HELIX — The Bred Formula

*A single closed-loop decision formula for the HELIX trading bot, cross-bred from the strongest quantitative primitives across the leading open-source trading repos. Each "gene" is a battle-tested piece of math lifted from a specific implementation; this document splices them into one coherent pipeline from raw signal → conviction → size → order price → risk gate.*

> This is an engineering/quant synthesis for research and backtesting. It is not financial advice. Trading is risky; every term below must be validated in backtest and dry-run before any capital is risked, with trade-only API keys (withdrawals disabled).

---

## 0. The idea in one line

**Blend many signals into one conviction, convert conviction into a win-probability and an edge, size the bet with friction-adjusted fractional Kelly scaled to a volatility target, throttle it by drawdown and correlation, gate it through hard risk rules, and quote it at an inventory-aware reservation price.**

The elegance is that the loop *closes*: the same profit-target/stop barriers that define the reward:risk ratio `b` (which feeds Kelly) are the exact barriers the executor enforces on the live order. Nothing is estimated twice with different numbers.

---

## 1. The parental genes (what we bred from)

| Gene | Formula it contributes | Source repo (code) |
|---|---|---|
| **G1 — Signal blend** | Normalized ensemble score `A ∈ [-1,1]` | OctoBot evaluators/`eval_note` — https://github.com/Drakkar-Software/OctoBot-Tentacles · intelligent-trading-bot `signal_sets` — https://github.com/asavinov/intelligent-trading-bot |
| **G2 — Meta-label probability** | `p = P(trade succeeds \| signal)` via triple-barrier + meta-model | mlfinlab (Hudson & Thames) — https://github.com/hudson-and-thames/mlfinlab · López de Prado *AFML* Ch.3 |
| **G3 — Edge / payoff** | Reward:risk `b` from triple-barrier geometry | mlfinlab triple-barrier · Freqtrade ROI/stoploss — https://github.com/freqtrade/freqtrade |
| **G4 — Bet fraction** | Fractional Kelly `f = λ·(p − (1−p)/b)` | machine-learning-for-trading (Kelly sizing) — https://github.com/stefan-jansen/machine-learning-for-trading |
| **G5 — Volatility target** | Scale by `σ\*/σ̂`, cap at `W_max` | FinRL / vol-targeting quant practice — https://github.com/AI4Finance-Foundation/FinRL · vectorbt — https://github.com/polakowo/vectorbt |
| **G6 — Drawdown throttle & risk gate** | Multipliers + hard protections | Freqtrade protections (`MaxDrawdown`,`StoplossGuard`,`CooldownPeriod`) · NautilusTrader `RiskEngine` — https://github.com/nautechsystems/nautilus_trader |
| **G7 — Inventory-aware quote** | Reservation price + optimal spread | Hummingbot `avellaneda_market_making.pyx` — https://github.com/hummingbot/hummingbot |
| **G8 — Executor barriers** | Stop / target / timeout that enforce G3 | Hummingbot `strategy_v2` executors · Freqtrade `custom_stoploss` |

---

## 2. The genes, stated precisely

### G1 — Composite alpha (signal blend)
Each evaluator `i` emits a raw feature `xᵢ` (an indicator, a model output, a sentiment score). Normalize each to a comparable scale with a rolling z-score, squash to `[-1,1]`, and take a weighted, confidence-scaled sum:

```
zᵢ    = (xᵢ − μᵢ) / (σᵢ + ε)              # rolling standardization
eᵢ    = tanh(zᵢ)                          # OctoBot-style eval_note ∈ [-1,1]
A     = tanh( Σᵢ wᵢ · eᵢ )                # composite alpha ∈ [-1,1]
dir   = sign(A)                           # +1 long, −1 short, 0 flat
conf  = |A|                               # raw conviction ∈ [0,1]
```

- `wᵢ` are ensemble weights (equal-weight is a robust default; or inverse-variance / performance-decayed).
- The outer `tanh` keeps the blend bounded no matter how many evaluators you add.
- **Why bred this way:** OctoBot normalizes every evaluator to one `eval_note` before a strategy aggregates them; intelligent-trading-bot aggregates model predictions into "one intelligent indicator." We keep both ideas: normalize, then bound.

### G2 — Win probability via triple-barrier meta-labeling
Don't trust `conf` as a probability. Instead, learn `p` from history using López de Prado's **triple-barrier** labels feeding a **meta-model**:

```
Barriers for each candidate entry at price s, horizon H, volatility σ̂:
   upper (take-profit) : s · (1 + u·σ̂)
   lower (stop-loss)   : s · (1 − d·σ̂)
   vertical (timeout)  : t + H
Label y = +1 if upper hit first, −1 if lower first, 0 if timeout.

Meta-model (e.g. gradient-boosted trees) trained on features + primary signal:
   p = P(primary signal is correct | features)      # meta-label probability
```

- The meta-model **decides whether to act** on the primary signal `dir`, not the direction itself — this is the whole point of meta-labeling (raise precision, filter false positives).
- **Fallback without ML:** map conviction to probability directly, `p = 0.5 + 0.5·conf·κ_cal`, where `κ_cal ∈ (0,1]` is a calibration haircut fit on backtest hit-rate. Never let `p` exceed a hard ceiling (e.g. 0.75) — overconfidence is what blows up Kelly.

### G3 — Payoff ratio (edge geometry)
The same barriers define the reward:risk ratio the bet actually faces:

```
b = (u · σ̂) / (d · σ̂) = u / d
```

Using ATR instead of return-vol: `b = (target_ATR_mult) / (stop_ATR_mult)`. This `b` is not a guess — it is *exactly* what the executor (G8) will enforce.

### G4 — Friction-adjusted fractional Kelly
Optimal growth fraction, then de-risked:

```
f_kelly = p − (1 − p) / b                 # classic Kelly (edge/odds)
f_fric  = f_kelly − c / b                 # subtract round-trip cost/edge drag
f       = clip( λ · f_fric , 0 , f_cap )  # fractional Kelly, non-negative
```

- `λ` = Kelly fraction. **Default λ = 0.25 (quarter-Kelly).** Half-Kelly captures ~75% of growth at ~50% of the drawdown; quarter-Kelly is the practitioner sweet spot (10–25% of full Kelly is the documented professional range).
- `c` = round-trip cost as a fraction (fees + expected slippage). Friction-adjustment prevents sizing up trades whose edge is eaten by costs.
- If `f ≤ 0` after friction → **no trade** (the edge doesn't survive costs). This single check kills most overfit signals.

### G5 — Volatility targeting
Convert the bet fraction into a portfolio weight that holds *risk* roughly constant across regimes:

```
σ\*     = target_annual_vol / sqrt(periods_per_year)     # e.g. 0.15/sqrt(252)
σ̂      = EWMA volatility forecast of the instrument
w_vol  = f · dir · (σ\* / (σ̂ + ε))
```

- Low-vol regime → larger notional; high-vol regime → smaller. This is what keeps the equity curve smooth and is standard in FinRL-style and vectorbt sizing.
- **Leverage cap** applied later in G6 via `W_max` so quiet markets can't produce absurd leverage.

### G6 — Drawdown throttle, correlation haircut, hard gates
Overlay portfolio-level protection (Freqtrade protections + Nautilus RiskEngine):

```
D(dd)   = clip(1 − dd_current / dd_max , D_floor , 1)      # linear de-gearing
H(ρ)    = 1 − ρ_penalty · max(0, ρ_to_book)                # correlation haircut
G_hard  = 1 if ALL pass else 0:
            • open_positions < N_max
            • per-instrument notional < cap
            • gross/net leverage < L_max
            • not in cooldown after k consecutive losses (StoplossGuard)
            • data fresh (no stale-feed / abnormal-gap circuit-breaker)
```

- `D(dd)` scales the whole book down as drawdown approaches `dd_max`; at `dd_max` it can floor to `D_floor` (e.g. 0.2) rather than 0, or hard-stop new entries.
- `H(ρ)` prevents stacking correlated bets (the classic "five EMA variants fire together" drawdown amplifier).

### G7 — Inventory-aware quote (Avellaneda–Stoikov)
Where to actually place the order. For **maker/market-making** execution, don't quote at mid — quote around the inventory-adjusted **reservation price**, exactly as Hummingbot implements it (`_optimal_bid = reservation_price − optimal_spread/2`, `_optimal_ask = reservation_price + optimal_spread/2`):

```
r      = s − q · γ · σ̂² · (T − t)                         # reservation price
spread = γ · σ̂² · (T − t) + (2/γ) · ln(1 + γ/κ)           # optimal total spread
bid    = r − spread/2
ask    = r + spread/2
```

- `q` = current inventory (signed), `γ` = inventory risk aversion, `κ` = order-book liquidity/intensity, `(T−t)` = time left in session (use the infinite-horizon variant for 24/7 crypto).
- Effect: when you're long (`q>0`), `r` drops below mid → you quote to sell down inventory. Inventory risk is priced, not ignored.
- For **taker/momentum** execution, skip A–S and cross the spread (or use next-open in backtest); A–S is the maker leg.

### G8 — Executor enforces the barriers
The order is handed to an Executor (Hummingbot `strategy_v2` pattern) that owns its lifecycle and enforces the *same* `u·σ̂` target, `d·σ̂` stop, and `H` timeout that defined `b` in G3 — closing the loop.

---

## 3. The bred formula (single expression)

**HELIX Composite Position Weight**

```
                ┌ G4 ┐ ┌  G1 ┐ ┌───── G5 ─────┐ ┌ G6 ┐┌ G6 ┐┌ G6 ┐
   w_target =   f  ·  dir ·  ( σ\* / σ̂ )      ·  D(dd)· H(ρ)· G_hard
   w_final  =   clip( w_target , −W_max , +W_max )
```

fully expanded:

```
w_final = clip(
    clip( λ · [ (p − (1−p)/b) − c/b ] , 0 , f_cap )        # G4 fractional Kelly, friction-adj
  · sign(A)                                                #  G1 direction from blended alpha
  · ( σ\* / (σ̂ + ε) )                                      #  G5 volatility targeting
  · clip(1 − dd/dd_max , D_floor , 1)                      #  G6 drawdown throttle
  · (1 − ρ_penalty · max(0, ρ_to_book))                    #  G6 correlation haircut
  · 𝟙[ all hard risk gates pass ]                          #  G6 non-bypassable gate
  , −W_max , +W_max )
```

with

```
A   = tanh( Σᵢ wᵢ · tanh(zᵢ) )                              # G1 composite alpha
p   = meta_model(features)   OR   0.5 + 0.5·|A|·κ_cal       # G2 win probability (ceiling ≤ p_max)
b   = u / d                                                 # G3 reward:risk (same barriers as executor)
```

and the order is quoted at (maker leg):

```
price_bid/ask = ( s − q·γ·σ̂²·(T−t) )  ∓  ½[ γ·σ̂²·(T−t) + (2/γ)·ln(1+γ/κ) ]   # G7 A–S
```

`w_final` is the fraction of NAV to hold in the instrument. Multiply by NAV and divide by price to get units; the executor (G8) then works the order and enforces the barriers.

---

## 4. Reference implementation (pseudocode)

```python
def helix_target_weight(inst, book, portfolio, cfg, meta_model=None):
    # ---- G1: blend signals into bounded alpha ----
    notes = [math.tanh(zscore(f.value, f.mean, f.std)) for f in inst.features]
    A     = math.tanh(sum(w*e for w, e in zip(cfg.weights, notes)))
    direction = 0 if abs(A) < cfg.entry_threshold else (1 if A > 0 else -1)
    if direction == 0:
        return 0.0

    # ---- G3: barriers -> reward:risk (same barriers the executor will use) ----
    sigma_hat = ewma_vol(inst.returns, cfg.vol_halflife)
    u, d = cfg.tp_mult, cfg.sl_mult
    b = u / d

    # ---- G2: probability of success ----
    if meta_model is not None:
        p = meta_model.predict_proba(inst.features)[1]
    else:
        p = 0.5 + 0.5 * abs(A) * cfg.calib_haircut
    p = min(p, cfg.p_max)                       # never over-trust

    # ---- G4: friction-adjusted fractional Kelly ----
    c = inst.round_trip_cost                    # fees + expected slippage (fraction)
    f_kelly = p - (1 - p) / b
    f = clamp(cfg.kelly_lambda * (f_kelly - c / b), 0.0, cfg.f_cap)
    if f == 0.0:
        return 0.0                              # edge doesn't survive costs -> skip

    # ---- G5: volatility targeting ----
    sigma_star = cfg.target_vol / math.sqrt(cfg.periods_per_year)
    w = f * direction * (sigma_star / (sigma_hat + EPS))

    # ---- G6: drawdown throttle + correlation haircut + hard gates ----
    w *= clamp(1 - portfolio.drawdown / cfg.dd_max, cfg.dd_floor, 1.0)
    w *= (1 - cfg.rho_penalty * max(0.0, portfolio.corr_to_book(inst)))
    if not portfolio.risk_gates_pass(inst, cfg):   # Nautilus RiskEngine / Freqtrade protections
        return 0.0

    return clamp(w, -cfg.w_max, cfg.w_max)


def helix_quote(inst, book, portfolio, cfg):     # G7: maker leg only
    s   = book.mid
    q   = portfolio.inventory(inst)              # signed
    sig = ewma_vol(inst.returns, cfg.vol_halflife)
    T_t = cfg.session_time_left                  # or infinite-horizon variant for 24/7
    g, k = cfg.gamma, estimate_kappa(book)       # liquidity/intensity from order book
    r      = s - q * g * sig**2 * T_t
    spread = g * sig**2 * T_t + (2 / g) * math.log(1 + g / k)
    return (r - spread / 2, r + spread / 2)      # (bid, ask)
```

---

## 5. Default parameters & guardrails

| Symbol | Meaning | Default | Guardrail / note |
|---|---|---|---|
| `wᵢ` | evaluator weights | equal-weight | inverse-variance if signals differ in reliability |
| `entry_threshold` | min \|A\| to trade | 0.15 | filters noise near zero |
| `p_max` | probability ceiling | 0.75 | prevents Kelly blow-ups from overconfidence |
| `κ_cal` | calibration haircut (no-ML mode) | 0.6 | fit to realized hit-rate |
| `u`, `d` | TP / SL vol-multiples | 2.0 / 1.0 | ⇒ `b = 2`; must equal executor barriers |
| `λ` (kelly_lambda) | Kelly fraction | **0.25** | 0.10–0.50 range; never full Kelly live |
| `c` | round-trip cost | per-venue | fees + modeled slippage |
| `f_cap` | max Kelly fraction | 0.20 | hard cap before vol scaling |
| `target_vol` | annual vol target | 0.15 | portfolio risk budget |
| `dd_max` | drawdown for full de-gear | 0.20 | Freqtrade `MaxDrawdown` analog |
| `D_floor` | min throttle at max dd | 0.20 | or 0 to hard-stop entries |
| `ρ_penalty` | correlation haircut strength | 0.5 | curbs stacking correlated bets |
| `W_max` | max weight per instrument | 1.0–2.0 | caps leverage in quiet markets |
| `γ` | inventory risk aversion (A–S) | tune | higher ⇒ more inventory-averse |
| `κ` | order-book liquidity (A–S) | from book | higher ⇒ tighter spreads |

---

## 6. Why this specific breed (design rationale)

- **Probability and payoff are separated** (G2 vs G3). Kelly needs both; conflating them (using conviction as probability with a fixed R:R) is the most common sizing error. Meta-labeling gives an honest `p`; the barriers give an honest `b`.
- **Friction is inside the sizing** (G4), not bolted on. `f − c/b` means a signal must beat its own transaction costs *before* it earns size — the single most effective overfit filter.
- **Risk is multiplicative and non-bypassable** (G6). Every factor is ≤ 1 (except vol scaling, which is capped), so risk overlays can only ever *shrink* a position, never inflate it. The hard gate is a literal `×0`.
- **The quote prices inventory** (G7). Most retail bots quote at mid and accumulate toxic inventory; A–S is the one widely-implemented academic fix (Hummingbot ships it verbatim).
- **The loop is closed** (G3 ↔ G8). The barriers that set `b` are the barriers the executor enforces, so backtest odds and live odds are the same numbers — the research-live-parity principle applied to sizing, not just to code paths.

---

## 7. Validation lineage

| Formula term | Cross-referenced against | Confidence |
|---|---|---|
| Bounded ensemble alpha `A` | OctoBot `eval_note`, intelligent-trading-bot signal aggregation, general ensemble practice | High |
| Triple-barrier labels + meta-model `p` | mlfinlab / Hudson & Thames, López de Prado *AFML* Ch.3, multiple independent reimplementations | High |
| Fractional Kelly `λ(p−(1−p)/b)` | machine-learning-for-trading, arXiv sizing surveys, practitioner consensus on λ∈[0.1,0.5] | High |
| Friction adjustment `−c/b` | friction-adjusted Kelly derivations (arXiv), Freqtrade fee-aware backtests | Medium |
| Volatility targeting `σ\*/σ̂` | FinRL, vectorbt, standard managed-vol practice | High |
| Drawdown / correlation / hard gates | Freqtrade protections, NautilusTrader RiskEngine | High |
| Avellaneda–Stoikov reservation + spread | Hummingbot `avellaneda_market_making.pyx` (verified in code), A–S 2008 paper, academic reimplementations | High |
| Executor-enforced barriers | Hummingbot `strategy_v2` executors, Freqtrade `custom_stoploss` | High |

---

## 8. Known failure modes (validate these first)

1. **`p` is not calibrated** → Kelly over-sizes. Always calibrate the meta-model (reliability curve) and cap at `p_max`; prefer quarter-Kelly.
2. **`b` from barriers ≠ what the executor does** → the whole edge estimate is wrong. Enforce them from one config object (G3 = G8).
3. **Correlated book** → `H(ρ)` needs a live correlation estimate; stale correlations understate risk in stress.
4. **`σ̂` lag** → EWMA vol reacts slowly to regime shifts; pair with a fast circuit-breaker on abnormal gaps.
5. **A–S `κ` mis-estimation** → spreads too tight in thin books; floor the spread and cap order size.
6. **Look-ahead in features feeding `A` or the meta-model** → run Freqtrade-style `lookahead-analysis` before trusting any backtest of this formula.

*Build it, backtest it, run the look-ahead and Monte-Carlo checks, then dry-run for weeks before a cent of live capital. The math is only as good as the data hygiene behind `p`, `b`, and `σ̂`.*
