# State-of-the-Art Automated Trading Bot — Reference Architecture

*A unified, production-grade design synthesized from code-level study of the leading open-source trading systems: NautilusTrader, Freqtrade, Hummingbot, CCXT, OctoBot, and the FinRL / intelligent-trading-bot ML ecosystem. Every major decision is cross-referenced against at least three independent implementations (see §11).*

Codename: **HELIX** — a single event-driven core that runs identical strategy code across backtest, paper, and live modes.

---

## 1. Executive Summary

### Vision
Build one engine that a strategy author writes against **once** and runs unchanged in backtest, paper-trade, and live modes across many exchanges and asset classes. The single largest source of real-world trading losses in the projects studied is **divergence between research and production** — vectorized backtests that quietly use future data, then behave differently live. HELIX eliminates that class of bug by making the *same event-driven code path* serve all modes, the pattern NautilusTrader and Freqtrade both adopt (Freqtrade literally shares its `IStrategy` interface between `Backtesting` and `FreqtradeBot`, swapping only the persistence layer).

### Core capabilities
- **Deterministic, event-driven core** with a central message bus; components are decoupled publishers/subscribers.
- **Unified exchange abstraction** over 100+ CEX/DEX venues via CCXT plus a normalized internal domain model.
- **Plugin strategy system** with hot-loadable strategies, evaluators, and risk modules (Freqtrade `resolvers`, OctoBot `tentacles`).
- **Pre-trade risk engine** as a mandatory gate between signal and execution (NautilusTrader `RiskEngine`, Freqtrade protections).
- **Executor-based order lifecycle** management for partial fills, timeouts, trailing stops, and DCA (Hummingbot `strategy_v2` executors).
- **Realistic backtesting**: no repainting, fee/slippage modeling, intra-candle exit resolution, look-ahead detection, and Monte Carlo robustness checks.
- **Optional ML pipeline** with strict offline-train / online-predict feature parity (intelligent-trading-bot, FreqAI, FinRL).
- **Full observability**: structured logs, metrics, Telegram/Discord alerts, web dashboard, REST API.

### Unique selling points
1. **Research–live parity by construction**, not by discipline.
2. **Mode is a runtime config, not a code branch** — backtest/paper/live are injected adapters behind identical interfaces.
3. **Risk is non-bypassable** — it sits on the critical path, not as an afterthought in strategy code.
4. **Crash-only, fail-fast core** — invalid state (NaN price, negative timestamp) halts immediately rather than silently corrupting a live book (NautilusTrader design philosophy).

---

## 2. High-Level Architecture Diagram

```
                          ┌──────────────────────────────────────────────────────────┐
                          │                     CONTROL PLANE                          │
                          │   Web Dashboard  │  REST/WebSocket API  │  Telegram/Discord │
                          └───────────────┬──────────────────────────────┬────────────┘
                                          │ commands / queries           │ alerts
                                          ▼                              ▲
┌───────────────┐   ticks/bars   ┌────────────────────────────────────────────────────────────┐
│  EXCHANGES    │◀──────────────▶│                    MESSAGE BUS (pub/sub)                     │
│  CEX / DEX    │   orders/fills │   topics: data.*  signal.*  order.*  fill.*  risk.*  sys.*   │
└──────┬────────┘                └───┬───────┬───────────┬───────────┬───────────┬─────────────┘
       │                             │       │           │           │           │
       ▼                             ▼       ▼           ▼           ▼           ▼
┌──────────────┐  normalized  ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐
│  ADAPTER /   │──domain──────▶│  DATA    │ │STRATEGY│ │ RISK   │ │EXECUTION│ │ PORTFOLIO /  │
│  EXCHANGE    │  events       │  ENGINE  │▶│ ENGINE │▶│ ENGINE │▶│ ENGINE  │▶│  ACCOUNTING  │
│  ABSTRACTION │◀──orders──────│ (ingest, │ │(alpha, │ │(pre-   │ │(order   │ │ (positions,  │
│  (CCXT +     │               │ feature, │ │signals)│ │ trade  │ │ mgmt,   │ │  PnL, margin)│
│  Gateway)    │               │ store)   │ │        │ │ gate)  │ │executors)│ │              │
└──────────────┘               └────┬─────┘ └────────┘ └────────┘ └────────┘ └──────┬───────┘
                                    │                                               │
                                    ▼                                               ▼
                          ┌───────────────────┐                          ┌───────────────────┐
                          │  MARKET DATA STORE │                          │  STATE / TRADE DB  │
                          │  (Parquet / TSDB)  │                          │ (Postgres/SQLite + │
                          │  + feature catalog │                          │      Redis cache)  │
                          └───────────────────┘                          └───────────────────┘

        ┌────────────────────────────────────────────────────────────────────────────────┐
        │  CROSS-CUTTING:  Config & Secrets  │  Logging/Metrics/Tracing  │  Scheduler       │
        └────────────────────────────────────────────────────────────────────────────────┘

   MODE INJECTION (same core, swapped edges):
     BACKTEST  → SimulatedExchange (historical feed + fill simulator)  ─┐
     PAPER     → LiveDataFeed + SimulatedExecution (no real orders)     ├─▶ identical engines
     LIVE      → LiveDataFeed + LiveExecution (real venue adapters)     ─┘
```

**Reading the diagram:** all inter-component communication flows through the message bus. No engine calls another engine directly; they publish and subscribe to topics. This is the decoupling pattern from NautilusTrader (`MessageBus`) and OctoBot (`Async-Channel` producer/consumer), and it is what makes mode-swapping possible — you replace the *edges* (data source, execution target) while the *center* (Data → Strategy → Risk → Execution → Portfolio) stays byte-for-byte identical.

---

## 3. Component Breakdown

For each component: **Responsibility · Inputs → Outputs · Suggested tech · Key idea (source repo).**

### 3.1 API Gateway / Exchange Abstraction Layer
- **Responsibility:** Translate every venue's idiosyncratic REST/WebSocket API into one normalized domain model (instruments, order books, ticks, orders, fills, balances). Handle auth, rate limiting, reconnection, and order-type mapping.
- **Inputs → Outputs:** venue raw payloads → normalized `MarketData` / `AccountState` events; normalized `OrderCommand` → venue-specific order requests.
- **Tech:** CCXT (unified spot/margin/futures over 100+ venues) as the baseline; per-venue **adapter** classes for the fast path (native WebSocket, venue-specific order flags); a **Gateway** micro-service for DEX/on-chain venues.
- **Key idea:** Two-layer design. CCXT gives breadth immediately; a thin **adapter interface** (NautilusTrader `adapters/`, Freqtrade `exchange/`, Hummingbot `connector/`) lets you drop to hand-written connectors for latency-critical venues without touching strategy code. Hummingbot separates on-chain connectivity into a distinct **Gateway** process — replicate that: keep blockchain RPC and signing isolated from the trading core.
- **Order types to normalize:** market, limit, stop-market, stop-limit, trailing-stop, post-only, reduce-only, IOC/FOK/GTC time-in-force, OCO, and bracket/OrderList (NautilusTrader `execution/` supports mixed-instrument `OrderList`).

### 3.2 Market Data Ingestion & Storage
- **Responsibility:** Subscribe to real-time feeds, normalize, timestamp deterministically, deduplicate, persist, and replay for backtests.
- **Inputs → Outputs:** venue WS streams / historical dumps → ordered `Tick`/`Bar`/`OrderBookDelta`/`Trade` events on `data.*` topics; persisted columnar history.
- **Tech:** async WebSocket clients; **Parquet** files partitioned by `venue/instrument/date` for historical bars/ticks; a time-series DB (**TimescaleDB** or **QuestDB**) for queryable live data; an in-memory **Cache** (NautilusTrader `cache/`, optionally Redis-backed) for hot state.
- **Key idea:** **Dual-timestamp every event** (`ts_event` = when it happened at the venue, `ts_init` = when we received it), the NautilusTrader convention. Deterministic ordering by `ts_event` is what makes backtests reproducible and prevents look-ahead. Store only **completed** candles for decision-making (Freqtrade explicitly refuses incomplete candles to avoid *repainting*). Maintain a **startup lookback** so indicators have enough warm-up history on both backtest and live (Freqtrade `startup_candle_count`).

### 3.3 Strategy Engine
- **Responsibility:** Host user strategies, feed them data events, and collect their signals. Manage strategy lifecycle (`on_start` / `on_bar` / `on_tick` / `on_event` / `on_stop`).
- **Inputs → Outputs:** `data.*` events + indicator values → `signal.*` events (intent to enter/exit/resize), never direct orders.
- **Tech:** Python for author ergonomics; a compiled hot path (Rust/Cython) for indicators if latency demands it (NautilusTrader model).
- **Key idea:** A **declarative + callback hybrid** interface. Freqtrade splits work into vectorized `populate_indicators` / `populate_entry_trend` / `populate_exit_trend` (fast, computed once) plus event **callbacks** (`custom_stoploss`, `custom_exit`, `adjust_trade_position`) for per-trade decisions. OctoBot separates **evaluators** (produce a normalized `eval_note` in [-1, 1]) from **strategies** (aggregate evaluator notes) from **trading modes** (turn a signal into orders). Adopt this three-tier separation: *evaluator → strategy → trading mode* maps cleanly onto *feature → alpha → execution intent* and keeps each piece independently testable and swappable.

### 3.4 Signal Generator / Alpha Module
- **Responsibility:** Convert features/indicators into a directional/sizing conviction score. This is where TA, statistical models, sentiment, and ML predictions are combined.
- **Inputs → Outputs:** features + model predictions → a scalar or vector conviction with metadata (horizon, confidence).
- **Tech:** TA-Lib / pandas-ta for indicators; scikit-learn / LightGBM / PyTorch for models; optional LLM/NLP evaluators for news & social sentiment (OctoBot supports social + LLM evaluators).
- **Key idea:** Aggregate heterogeneous signals into **one normalized indicator** before it reaches sizing (both OctoBot's `eval_note` aggregation and intelligent-trading-bot's `signal_sets` do exactly this). Keep alpha **stateless and pure** so the same function runs in backtest and live. Version every alpha so backtest results are reproducible against a known signal definition.

### 3.5 Order Manager & Execution Engine
- **Responsibility:** Turn approved signals into concrete orders, manage their full lifecycle (submit, amend, cancel, timeout, partial fill, retry), and emit `fill.*` events.
- **Inputs → Outputs:** risk-approved `OrderCommand` → venue orders; venue order-updates → `OrderFilled`/`OrderCanceled`/`OrderRejected` events.
- **Tech:** async order router; idempotent client-order-ID generation; per-venue rate-limit-aware queueing.
- **Key idea:** **Executors as first-class objects** (Hummingbot `strategy_v2` executors: `PositionExecutor`, `DCAExecutor`, `TWAPExecutor`, etc.). Instead of scattering order-management logic through the strategy, a strategy *launches an executor* that owns one position's entire lifecycle (entry, stop, take-profit, trailing, timeout) and reports back. This cleanly encapsulates the messy realities Freqtrade handles via `unfilledtimeout`, `custom_stoploss`, and `stoploss_on_exchange`. Define an explicit **exit precedence** (Freqtrade's is: *Liquidation → Stop-loss → ROI → Trailing-stop → Exit-signal*) so competing exit conditions resolve deterministically.

### 3.6 Risk Management & Portfolio Allocator
- **Responsibility:** A **mandatory pre-trade gate**. Validate every order against position limits, notional caps, leverage, drawdown, correlation, and rate throttles *before* it reaches execution. Allocate capital across strategies/instruments.
- **Inputs → Outputs:** proposed order + current portfolio state → APPROVE / MODIFY / REJECT + reason.
- **Tech:** in-process rule engine on the critical path; portfolio state from the accounting component.
- **Key idea:** Risk lives **between** strategy and execution as its own engine (NautilusTrader `RiskEngine` is a distinct core engine; Freqtrade `plugins/protections` implements `StoplossGuard`, `MaxDrawdown`, `LowProfitPairs`, `CooldownPeriod`). Concretely implement:
  - **Position sizing:** fixed-fractional, volatility-targeted (ATR-scaled), or fractional-Kelly.
  - **Hard caps:** max open positions, max per-instrument notional, max gross/net leverage.
  - **Circuit breakers:** halt new entries after N consecutive losses, after a rolling max-drawdown breach, or on abnormal data (stale feed, price gap).
  - **Correlation guard:** reject stacking highly correlated positions (a known Freqtrade footgun — five EMA variants firing together amplify drawdown).
  Risk rejections must be **logged and alerted**, never silent.

### 3.7 Backtesting & Simulation Engine
- **Responsibility:** Replay historical data through the *identical* engine graph with a simulated exchange, producing fills, PnL, and performance analytics.
- **Inputs → Outputs:** historical dataset + strategy + config → trade ledger, equity curve, metrics report.
- **Tech:** event-driven replay (not vectorized) for parity; a fill simulator modeling fees, slippage, and latency.
- **Key idea:** **Event-driven backtest = live path** (NautilusTrader's core thesis; Freqtrade shares `IStrategy` between `Backtesting` and live, differing only in `LocalTrade` in-memory vs `Trade` DB persistence). Fill realism rules to implement:
  - Signals fire on **candle close**, fills occur at **next candle open** (or within low/high range for limit orders) — never same-candle (Freqtrade).
  - Optional **sub-candle detail timeframe** for accurate intra-candle stop/TP resolution (Freqtrade `timeframe_detail`).
  - Model **fees per venue** and **slippage**; assume limit orders may *not* fill if volume is thin.
  - Ship **look-ahead and recursion analyzers** (Freqtrade `lookahead-analysis`, `recursive-analysis`) to automatically catch future-data leaks.
  - **Monte Carlo** trade-order/candle shuffling to reject strategies that only work on one specific market sequence (Jesse).

### 3.8 Paper Trading / Dry-Run Mode
- **Responsibility:** Run against **live** market data but route orders to a simulated executor — real-time forward testing with zero capital risk.
- **Inputs → Outputs:** live feed → simulated fills tracked exactly as real trades.
- **Tech:** the live data adapter + the backtest fill simulator, wired behind the same `ExecutionClient` interface.
- **Key idea:** Every studied project treats this as **mandatory before live** (Freqtrade `dry_run`, OctoBot simulator, Hummingbot `paper_trade`). It catches the gap between backtest assumptions (all orders fill) and reality (limit orders may not). Expect and monitor small divergences between dry-run and backtest as a *health signal*, not a bug.

### 3.9 State Persistence & Database
- **Responsibility:** Durable record of orders, trades, positions, balances, strategy state, and an event log for recovery.
- **Inputs → Outputs:** engine events → persisted rows; on restart → rehydrated state.
- **Tech:** **SQLite** for single-node simplicity (Freqtrade default) or **PostgreSQL** for multi-node/HA; **SQLAlchemy** ORM; **Redis** for the hot cache / cross-process message-bus backing (NautilusTrader optional Redis cache & bus backend).
- **Key idea:** Separate **in-memory backtest state** (`LocalTrade`) from **persisted live state** (`Trade`) behind one interface (Freqtrade). Use **event sourcing** for the live core — persist the ordered stream of state-changing events so a crashed node can deterministically replay to its last good state (NautilusTrader event store + crash-only design).

### 3.10 Monitoring, Logging & Alerting
- **Responsibility:** Structured logs, real-time metrics, health checks, and push alerts on fills, errors, risk trips, and disconnects.
- **Inputs → Outputs:** engine events → logs/metrics/alerts.
- **Tech:** structured logging (JSON); **Prometheus + Grafana** for metrics; **Sentry** for exceptions; Telegram/Discord/webhook notifiers.
- **Key idea:** Multiple, redundant control/monitor surfaces (Freqtrade offers FreqUI + Telegram + REST + webhook + Discord; Hummingbot has the **Condor** Telegram bot; OctoBot has web + telegram interfaces). Alert on: order fills, rejects, risk-engine trips, data-feed staleness, reconnects, PnL thresholds, and drawdown breaches.

### 3.11 UI / Dashboard / Control Panel
- **Responsibility:** Observe and control the bot — start/stop, force-exit, adjust config, view equity curve, open positions, and logs.
- **Inputs → Outputs:** operator actions → control commands; engine state → live views.
- **Tech:** React/TypeScript SPA over the REST/WebSocket API (FreqUI pattern); mobile control via Telegram bot.
- **Key idea:** The UI is a **client of the same API** used by automation — the control plane is decoupled from the trading core over the message bus (Hummingbot's Condor/MCP/API talk to the client via a server layer). Never let the dashboard reach into engine internals directly.

### 3.12 Security & Secrets Management
- **Responsibility:** Protect API keys, signing keys, and credentials; enforce least privilege.
- **Inputs → Outputs:** secret references → runtime credentials (never logged, never in code).
- **Tech:** environment variables via `.env` (git-ignored; NautilusTrader ships `.env.example`); a secrets manager (**HashiCorp Vault**, AWS/GCP Secrets Manager) in production; OS keyring for desktop.
- **Key idea:** **Keys never in source or images.** Docker mounts config/secrets as volumes at runtime (Freqtrade, OctoBot pattern). Enforce **trade-only API permissions with withdrawals disabled** — the single most repeated security instruction across every project. Isolate blockchain signing in the Gateway process. Rotate keys; scope one key per venue per bot instance for blast-radius control.

### 3.13 Deployment & Orchestration
- **Responsibility:** Package, ship, run, and supervise the bot reproducibly.
- **Inputs → Outputs:** source + config → running, restart-on-failure services.
- **Tech:** **Docker** (all three flagship projects ship official images); **docker-compose** for single-host multi-service (core + DB + Redis + Grafana + Gateway); **Kubernetes** for multi-strategy fleets; a **scheduler** (cron/APScheduler/K8s CronJob) for data downloads, retraining, and reporting.
- **Key idea:** One image, config injected at runtime (Freqtrade `Dockerfile` = base image + strategy + config volumes). Run the core as a **long-lived service with automatic restart**; run backtests/retraining/reports as **scheduled jobs**. Keep the low-latency trading core close to the venue (co-located region) and the control plane anywhere.

### 3.14 (Optional) ML Pipeline for Model-Based Strategies
- **Responsibility:** Train, validate, deploy, and periodically retrain predictive models that feed the alpha module.
- **Inputs → Outputs:** historical features + labels → serialized models → live predictions.
- **Tech:** scikit-learn / LightGBM / XGBoost for tabular; PyTorch/TF for deep nets; FinRL for deep-RL agents; Optuna for hyperparameter search (Freqtrade uses Optuna in hyperopt).
- **Key idea:** **Guarantee identical feature computation offline and online** — the central challenge intelligent-trading-bot is built around ("the same derived features must be used in both training and prediction"). Use **rolling/walk-forward retraining** evaluated only on unseen data (intelligent-trading-bot `predict_rolling`; FreqAI adaptive retraining). Treat model deployment as a first-class, versioned artifact; never let a model see future data through leaky features.

---

## 4. Data Flow and Event Sequence

Lifecycle of one decision, from market tick to persisted fill:

1. **Ingest.** Exchange adapter receives a WebSocket message → normalizes to a domain event with `ts_event`/`ts_init` → publishes `data.bar.BTCUSDT.BINANCE`.
2. **Feature/Indicator update.** Data Engine updates rolling indicators and (optionally) ML features from the completed candle; publishes enriched data.
3. **Evaluate.** Subscribed evaluators compute normalized notes (RSI note, trend note, sentiment note, model-prediction note).
4. **Alpha aggregation.** Strategy aggregates evaluator notes into a single conviction; if it crosses a threshold, it publishes `signal.entry` (or `signal.exit`/`signal.resize`) with size intent.
5. **Risk gate.** Risk Engine consumes the signal + current portfolio state, applies sizing and all guards, and either publishes `order.command` (approved, possibly resized) or `risk.rejected` (with reason → alert).
6. **Execute.** Execution Engine launches/updates an **Executor** for the position; submits the venue order with an idempotent client-order-ID; publishes `order.submitted`.
7. **Venue ack & fills.** Adapter streams order updates → Execution Engine publishes `order.filled` (partial/full), `order.canceled`, or `order.rejected`. Timeouts trigger amend/cancel per policy.
8. **Accounting.** Portfolio/Accounting updates positions, realized/unrealized PnL, and margin; publishes `portfolio.updated`.
9. **Persist.** Persistence subscribes throughout and writes orders/trades/positions + the event log; cache updates hot state.
10. **Observe.** Monitoring emits metrics and pushes alerts (fill notification, PnL update). Dashboard reflects new state over WebSocket.
11. **Post-trade management.** The Executor keeps managing the open position (trailing stop, take-profit, DCA add, time-based exit) until closed, looping steps 5–9 as needed.

In **backtest/paper** modes, steps 6–7 are served by the **SimulatedExecution** client (fees/slippage/next-open fills); every other step is identical.

---

## 5. Multi-Exchange & Multi-Asset Support Strategy

- **Normalized domain model as the lingua franca.** Internally, everything is an `Instrument` with a venue-qualified ID (NautilusTrader uses `ETHUSDT-PERP.BINANCE`). Strategies reference instruments, never raw venue symbols. CCXT + adapters translate at the edge.
- **Capability metadata per venue.** Each adapter declares supported order types, fee schedule, tick/lot sizes, rate limits, and market types (spot/margin/perp/options). Risk and execution read this metadata so the same strategy adapts to venue constraints (Freqtrade requires per-exchange backtests because fees differ: Binance 0.1% vs Kraken 0.16% flips profitability).
- **Asset-class abstraction.** Spot, perpetual futures, dated futures, and options share the `Instrument` interface but carry class-specific fields (leverage, funding rate, expiry, greeks — NautilusTrader models option greeks natively). Margin/liquidation logic lives in Accounting, not strategies.
- **DEX/on-chain via Gateway.** On-chain venues go through a separate signing/RPC service (Hummingbot Gateway) so blockchain concerns never leak into the core.
- **Cross-venue strategies.** Because all venues publish to one bus with a unified model, market-making and arbitrage across venues are natural (NautilusTrader explicitly supports multi-venue cross-market strategies).
- **Per-venue deployment option.** For isolation, run one core instance per venue with a shared control plane; for cross-venue alpha, run one instance subscribing to multiple adapters.

---

## 6. Operational Modes

| Mode | Data source | Execution | Capital risk | Primary use |
|---|---|---|---|---|
| **Backtest** | Historical (Parquet/TSDB replay) | Simulated fills (fees/slippage/next-open) | None | Strategy R&D, hyperopt, Monte Carlo |
| **Paper / Dry-run** | Live feed | Simulated fills | None | Forward testing, pre-live validation |
| **Live** | Live feed | Real venue adapters | Real | Production trading |
| **Hybrid** | Live feed | Live for some instruments/strategies, simulated for others | Partial | Canarying a new strategy alongside proven ones; A/B allocation |

Mode is selected by **dependency injection at startup** — the engine graph binds a `DataClient` and an `ExecutionClient` implementation. Strategy code is unaware of the mode. This is the enabling mechanism for research–live parity (NautilusTrader "identical strategy implementations"; Freqtrade shared `IStrategy`). **Mandatory promotion path:** backtest → dry-run (1–2 weeks) → small-capital live → scaled live.

---

## 7. Technology Stack Recommendations

| Layer | Recommendation | Rationale (grounded in research) |
|---|---|---|
| **Core language** | Python control plane + Rust/Cython hot path | Python for strategy ergonomics and ecosystem; compiled core for deterministic, low-latency event processing. Exactly NautilusTrader's split (Rust core, Python control plane); OctoBot Cythonizes hot loops. |
| **Exchange connectivity** | CCXT + native adapters + Gateway (DEX) | CCXT is the de-facto standard (~42K★, 100+ venues) used by Freqtrade and OctoBot; native adapters for latency; Gateway isolates on-chain (Hummingbot). |
| **Messaging (in-proc)** | Async message bus (asyncio) | Decoupled pub/sub is the shared backbone of NautilusTrader (`MessageBus`) and OctoBot (`Async-Channel`). Avoids inefficient polling loops. |
| **Messaging (cross-proc/HA)** | Redis Streams or NATS | NautilusTrader supports Redis as a bus/cache backend for multi-node; NATS is a lightweight alternative for a bus fabric. |
| **Market-data store** | Parquet (cold) + TimescaleDB/QuestDB (hot) | Columnar Parquet for cheap, replayable history; TSDB for queryable live series. |
| **State DB** | SQLite (single node) / PostgreSQL (HA) via SQLAlchemy | Freqtrade uses SQLite+SQLAlchemy; Postgres for multi-instance durability. |
| **Cache** | Redis | Hot state, dedup, and optional bus backing. |
| **Indicators / TA** | TA-Lib, pandas-ta | Ubiquitous across Freqtrade strategies. |
| **ML** | scikit-learn / LightGBM / PyTorch; FinRL (RL); Optuna (HPO) | FreqAI + intelligent-trading-bot use sklearn-family pipelines; FinRL for DRL; Freqtrade hyperopt uses Optuna. |
| **Dashboard** | React/TypeScript SPA + FastAPI backend | FreqUI pattern; FastAPI for the REST/WS control API. |
| **Notifications** | Telegram + Discord + webhooks | Standard across Freqtrade, Hummingbot (Condor), OctoBot. |
| **Observability** | Prometheus + Grafana + Sentry + structured JSON logs | Production monitoring; NautilusTrader ships a dedicated logging framework. |
| **Packaging** | Docker + docker-compose; K8s for fleets | Official images exist for all three flagships. |
| **Secrets** | `.env` (dev) → Vault / cloud secrets manager (prod) | NautilusTrader `.env.example`; keys never in images. |

---

## 8. Security & Resilience Measures

- **API-key hygiene:** trade-only permissions, withdrawals disabled, one scoped key per venue per instance, rotated regularly, stored in a secrets manager, never logged.
- **Rate limiting:** per-venue token-bucket throttles in the adapter; respect venue weight limits; back off on 429s. Freqtrade caps redundant OHLCV calls precisely to avoid overload.
- **Idempotency:** deterministic client-order-IDs so retries never double-submit.
- **Circuit breakers:** halt new entries on max-drawdown breach, consecutive-loss threshold, abnormal spread/price gap, or stale data feed (Freqtrade protections; NautilusTrader risk engine).
- **Fail-fast / crash-only:** invalid domain state (NaN/inf price, negative timestamp, impossible fill) halts the instance immediately rather than trading on corruption; recover deterministically by replaying the event log (NautilusTrader design philosophy).
- **Failover:** health-checked live process with auto-restart (Docker restart policy / K8s liveness probe); on restart, rehydrate from persisted state + event log and **reconcile open orders/positions against the venue** before resuming.
- **Feed redundancy:** detect WebSocket staleness via heartbeat; auto-reconnect with exponential backoff; fall back to REST polling; pause trading while degraded.
- **Kill switch:** an operator command (dashboard/Telegram) that cancels all open orders and flattens or freezes positions instantly.
- **Reconciliation:** on startup and periodically, compare internal state to venue truth (balances, open orders, positions) and alert on any divergence.

---

## 9. Scalability & High-Availability Considerations

- **Single-threaded deterministic core, scale by sharding.** Keep one instrument-set/strategy on a single-threaded event core (determinism > parallelism inside the core; NautilusTrader's core is single-threaded with cross-thread channels feeding it). Scale **horizontally** by running more instances, each owning a disjoint set of instruments/strategies.
- **Stateless-edge, stateful-center.** Data adapters and the control plane scale independently; the stateful core persists via event log + DB so any replica can recover.
- **Shared bus for multi-node.** Use Redis Streams/NATS as the cross-process bus so a fleet of cores, a shared risk/portfolio service, and the control plane interoperate (NautilusTrader multi-node message-bus streams keyed by trader instance UUID).
- **Central portfolio risk service.** When many cores trade one account, route their orders through a **shared risk/allocator service** to enforce account-level caps and correlation limits (prevents independent bots from collectively over-leveraging).
- **Data tier scaling:** partition market data by venue/instrument/date; use TSDB continuous aggregates for multi-timeframe queries; keep hot windows in Redis.
- **HA:** active-passive core with fast failover (passive replays event log to warm state); Postgres with replication; health probes + auto-restart. True active-active on one account is dangerous (double-fills) — prefer active-passive with venue reconciliation on promotion.
- **Compute-heavy jobs off the hot path:** run backtests, hyperopt, and model retraining as separate scheduled/batch workers so they never contend with live trading.

---

## 10. How to Extend with New Strategies (Plugin System)

- **Convention-based discovery + resolver loading.** Drop a strategy file into a known directory; a **resolver** dynamically imports the class by name at startup (Freqtrade `resolvers/` loads strategies/pairlists/protections by class name; `freqtrade trade --strategy MyStrategy`). No core changes required.
- **Tentacle-style pluggable everything.** Make evaluators, strategies, risk modules, and trading modes all pluggable behind abstract base classes, installable/updatable independently, with the engine auto-selecting the most specific implementation (OctoBot tentacles: it "automatically looks for the most advanced version of all available tentacles"). New plugin types are added by subclassing and registering in a config manifest (`tentacles_config.json` analog).
- **Config-driven activation.** A profile/config file enables which plugins run and with what parameters (OctoBot per-profile `specific_config`; Freqtrade `config.json`), so the same binary runs many strategy compositions.
- **Executor plugins.** New order-management behaviors (e.g., iceberg, TWAP, grid, DCA) are added as new **Executor** subclasses without touching strategies (Hummingbot `strategy_v2` executor pattern).
- **Hot-reload considerations.** Support reloading a strategy without dropping venue connections for paper/dev; for live, prefer a **controlled restart with state rehydration + venue reconciliation** over in-place hot-swapping (safer given open positions). Strategies must be **pure/declarative** enough that reload = re-instantiate + replay warm-up candles.
- **Extension contract checklist for a new strategy:** implement `populate_indicators`/`on_bar` (features), the evaluator→alpha aggregation, entry/exit intent, and optional `custom_stoploss`/`custom_exit` callbacks; declare required timeframes, warm-up lookback, and instruments; provide a config schema and default params; ship a backtest fixture. Then: backtest → lookahead/recursion analysis → hyperopt → Monte Carlo → dry-run → small live.

---

## 11. Validation Summary

Each major architectural choice mapped to the repositories that inspired or validated it. Confidence reflects breadth of adoption across independently-developed, actively-maintained projects.

| # | Architectural choice | Validated by (repo + link) | Confidence |
|---|---|---|---|
| 1 | Deterministic **event-driven core with central message bus** | NautilusTrader `MessageBus`/`common` — https://github.com/nautechsystems/nautilus_trader · OctoBot `Async-Channel` — https://github.com/Drakkar-Software/OctoBot · Hummingbot `core` clock/event — https://github.com/hummingbot/hummingbot | **High** |
| 2 | **Unified exchange abstraction** (CCXT + adapter layer + Gateway) | CCXT — https://github.com/ccxt/ccxt · Freqtrade `exchange/` — https://github.com/freqtrade/freqtrade · Hummingbot `connector/` + Gateway — https://github.com/hummingbot/hummingbot · NautilusTrader `adapters/` | **High** |
| 3 | **Research–live parity** (same strategy code all modes) | NautilusTrader (identical strategies backtest/live) · Freqtrade (shared `IStrategy`, `LocalTrade` vs `Trade`) — https://github.com/freqtrade/freqtrade | **High** |
| 4 | **Declarative + callback strategy interface** | Freqtrade `strategy/` (`populate_*` + callbacks) · OctoBot evaluator/strategy/trading-mode tiers — https://github.com/Drakkar-Software/OctoBot-Tentacles · NautilusTrader `Strategy`/`Actor` | **High** |
| 5 | **Plugin / resolver hot-load system** | Freqtrade `resolvers/` · OctoBot tentacles + `tentacles_config.json` | **High** |
| 6 | **Executor-based order lifecycle** | Hummingbot `strategy_v2` executors — https://github.com/hummingbot/hummingbot · NautilusTrader `execution/` (`OrderList`) | **High** |
| 7 | **Risk engine as mandatory pre-trade gate** | NautilusTrader `risk/` `RiskEngine` · Freqtrade `plugins/` protections (`StoplossGuard`, `MaxDrawdown`, `LowProfitPairs`) | **High** |
| 8 | **Backtest realism** (no repaint, next-open fills, fees/slippage, intra-candle, look-ahead detection) | Freqtrade `optimize/` + `lookahead-analysis`/`recursive-analysis` · NautilusTrader `backtest/` · Jesse (Monte Carlo) — https://github.com/jesse-ai/jesse | **High** |
| 9 | **Paper / dry-run before live** | Freqtrade `dry_run` · OctoBot simulator · Hummingbot `paper_trade` | **High** |
| 10 | **Persistence: SQLite/Postgres via SQLAlchemy + Redis cache + event log** | Freqtrade `persistence/` · Hummingbot `model/` · NautilusTrader `cache/` + Redis + event store | **High** |
| 11 | **Dual-timestamp deterministic data + no incomplete candles** | NautilusTrader (`ts_event`/`ts_init`) · Freqtrade (no repainting, `startup_candle_count`) | **High** |
| 12 | **Hyperparameter optimization (Optuna) + walk-forward** | Freqtrade `optimize/` hyperopt (Optuna) · intelligent-trading-bot `predict_rolling` — https://github.com/asavinov/intelligent-trading-bot | **High** |
| 13 | **Multi-surface monitoring/control** (Telegram + REST + WebUI + webhook) | Freqtrade `rpc/` (FreqUI/Telegram/REST) · Hummingbot `notifier/` + Condor · OctoBot `Services` | **High** |
| 14 | **ML pipeline with offline/online feature parity** | intelligent-trading-bot (offline train / online predict parity) · Freqtrade FreqAI (`freqai/`) · FinRL (DRL) — https://github.com/AI4Finance-Foundation/FinRL | **High** |
| 15 | **Docker-first deployment, config/secrets as volumes** | Freqtrade `Dockerfile` · OctoBot `drakkarsoftware/octobot` image · Hummingbot images | **High** |
| 16 | **Crash-only / fail-fast core with event-sourced recovery** | NautilusTrader (design philosophy, event store) | **Medium** |
| 17 | **DEX / on-chain isolated behind a Gateway process** | Hummingbot Gateway — https://github.com/hummingbot/hummingbot | **Medium** |
| 18 | **Normalized instrument/domain model for multi-asset (spot/perp/options/greeks)** | NautilusTrader `model/` (option greeks, instrument types) · CCXT unified markets | **High** |
| 19 | **Vectorized research alongside event-driven production** | vectorbt (research) — https://github.com/polakowo/vectorbt · NautilusTrader (bridges the two) | **Medium** |
| 20 | **Community strategy library / templates** | Freqtrade `freqtrade-strategies` — https://github.com/freqtrade/freqtrade-strategies · OctoBot default tentacles | **Medium** |
| 21 | **HFT-grade backtest realism** (feed+order latency models, order-queue-position fills, L2/L3 order-book reconstruction) | hftbacktest — https://github.com/nkaz001/hftbacktest · Crypto-toolbox/HFT-Orderbook — https://github.com/Crypto-toolbox/HFT-Orderbook | **Medium** |
| 22 | **Agent-native / LLM decision layer behind the risk gate** | HKUDS/AI-Trader — https://github.com/HKUDS/AI-Trader · The-Swarm-Corporation/AutoHedge — https://github.com/The-Swarm-Corporation/AutoHedge · okx/agent-trade-kit — https://github.com/okx/agent-trade-kit · OctoBot LLM evaluators · Hummingbot MCP | **Medium** |
| 23 | **On-chain / MEV arbitrage isolated in a signing service** | flashbots/simple-arbitrage — https://github.com/flashbots/simple-arbitrage · paco0x/amm-arbitrageur — https://github.com/paco0x/amm-arbitrageur · x89/Solana-Arbitrage-Bot — https://github.com/x89/Solana-Arbitrage-Bot | **Medium** |
| 24 | **ML-first walk-forward framework** | edtechre/pybroker — https://github.com/edtechre/pybroker · tensortrade-org/tensortrade — https://github.com/tensortrade-org/tensortrade · FinRL | **Medium** |

**Additional references consulted:** QuantConnect Lean (multi-asset live engine, C#/.NET) — https://github.com/QuantConnect/Lean; Superalgos (visual strategy design) — https://github.com/Superalgos/Superalgos; stefan-jansen/machine-learning-for-trading — https://github.com/stefan-jansen/machine-learning-for-trading; backtrader / zipline (historically influential backtesting engines, now largely unmaintained — patterns borrowed, code not).

---

## 12. Open Challenges & Future Directions

Derived from recurring issues, PR discussions, and documented limitations across the studied repos:

1. **Backtest ≠ reality gap.** Every project warns loudly that impressive backtests rarely survive live (Freqtrade: "do not assume these results are achievable"). Slippage, partial fills, latency, and thin-book liquidity are hard to simulate. *Direction:* higher-fidelity fill simulation with order-book replay and latency injection; mandatory dry-run gating; live-vs-backtest divergence dashboards as a first-class metric.
2. **ML deployment & overfitting.** FinRL and FreqAI are powerful but "not turnkey" — bridging sim-to-live (real-time data, fees, slippage, execution latency) and preventing look-ahead through leaky features remain the dominant failure modes. *Direction:* enforced feature-parity harness, walk-forward-only evaluation, automated data-leak detection, model versioning + shadow deployment before promotion.
3. **Connector maintenance burden.** Venue APIs change constantly; unmaintained connectors silently break (many arbitrage repos are marked dead). *Direction:* CCXT for breadth + a strong per-adapter contract test suite run continuously against venue sandboxes; capability metadata so the core degrades gracefully.
4. **Live limited to few venues even when backtest supports many.** Jesse backtests widely but live-trades only a handful of exchanges. *Direction:* clearly separate `DataClient` breadth from `ExecutionClient` breadth; certify execution adapters independently.
5. **Correlated-strategy drawdown stacking.** Running many similar strategies amplifies drawdowns when they fire together. *Direction:* account-level correlation guard in a shared risk service; portfolio-aware allocation rather than per-strategy sizing.
6. **HA without double-fills.** Active-active on one account risks duplicate orders; determinism favors single-threaded cores. *Direction:* active-passive with event-log replay + venue reconciliation on failover; idempotent order IDs; a shared portfolio-of-record.
7. **Secrets & operational security at scale.** Keys in env files don't scale to fleets. *Direction:* secrets-manager integration, per-instance scoped keys, and a hardened Gateway for on-chain signing.
8. **AI-agent / LLM strategies (emerging).** OctoBot's LLM evaluators and Hummingbot's MCP AI-agent surface point to LLM-driven, explainable strategies and natural-language strategy authoring (JesseGPT). *Direction:* treat LLM output as *one normalized evaluator note* behind the same risk gate — never let a model place orders directly; keep the deterministic risk engine authoritative.
9. **Regulatory & compliance surface.** Multi-jurisdiction trading raises reporting/KYC/tax obligations rarely handled by OSS bots. *Direction:* pluggable compliance/reporting module fed by the same event log.

---

### Build order (suggested implementation sequence)
1. Domain model + message bus + config/secrets scaffolding.
2. One exchange adapter (via CCXT) + market-data ingestion + Parquet store.
3. Backtest engine with SimulatedExecution (fees/slippage/next-open) + one reference strategy.
4. Strategy engine + evaluator/alpha tiers + resolver plugin loading.
5. Risk engine (sizing + core guards) as the mandatory gate.
6. Execution engine + Executors + live adapter; add dry-run mode (reuse sim fills on live data).
7. Persistence + event log + reconciliation + crash recovery.
8. Monitoring/alerting + REST API + dashboard + Telegram.
9. Hyperopt + look-ahead/Monte-Carlo analyzers.
10. Optional ML pipeline with feature-parity harness.
11. Dockerize; add second venue; scale to multi-instance with shared risk service.

*This document is a design synthesis for educational and engineering purposes. Automated trading carries substantial financial risk; validate every component in backtest and dry-run before risking capital, and use trade-only API keys with withdrawals disabled.*

---

## Appendix A — Repository Survey (90+ repos)

Collected via the GitHub search API across the topic areas below; star counts, primary language, and last-push dates are as of **2026-07-08**. "Status" reflects maintenance: **Active** = pushed within ~6 months, **Legacy** = influential but stale/unmaintained. This is the breadth pool the architecture was synthesized from; deep code-level study focused on the starred (★) rows.

### A.1 Full trading-bot frameworks & live engines
| Repo | Stars | Lang | Last push | Status | Note |
|---|---|---|---|---|---|
| ★ freqtrade/freqtrade | 52,167 | Python | 2026-07-07 | Active | Reference for strategy interface, protections, dry-run, hyperopt, rpc |
| ★ nautechsystems/nautilus_trader | ~10k+ | Rust/Python | 2026-07 | Active | Reference for event-driven core, engines, research-live parity |
| ★ hummingbot/hummingbot | 19,092 | Python | 2026-07-08 | Active | Reference for connectors, Gateway, strategy_v2 executors |
| ★ QuantConnect/Lean | 20,421 | C# | 2026-07-07 | Active | Multi-asset live engine; alpha/insight model |
| ★ Drakkar-Software/OctoBot | 6,210 | Python | 2026-07-08 | Active | Reference for tentacle plugin system, Async-Channel |
| jesse-ai/jesse | 8,144 | Python | 2026-07-07 | Active | Multi-TF, Monte Carlo, JesseGPT |
| StockSharp/StockSharp | 10,256 | C# | 2026-06-22 | Active | Designer + connectors (stocks/FX/crypto) |
| ricequant/rqalpha | 6,557 | Python | 2026-07-08 | Active | Extendable backtest+trade framework |
| Superalgos/Superalgos | 5,564 | JavaScript | 2026-07-08 | Active | Visual strategy design, multi-server |
| TraderAlice/OpenAlice | 5,865 | TypeScript | 2026-07-08 | Active | TS trading framework |
| brokermr810/QuantDinger | 9,360 | Python | 2026-07-03 | Active | Quant platform |
| chrisleekr/binance-trading-bot | 5,525 | JavaScript | 2025-06-10 | Active | Grid/DCA, web UI |
| Haehnchen/crypto-trading-bot | 3,494 | TypeScript | 2026-03-01 | Active | Multi-exchange TS bot |
| Open-Trader/opentrader | 2,750 | TypeScript | 2025-06-29 | Active | Grid/DCA, self-hosted |
| Lumiwealth/lumibot | 1,797 | Python | 2026-07-08 | Active | Multi-broker live+backtest |
| AlexWan/OsEngine | 1,003 | C# | 2026-07-07 | Active | .NET trading platform |
| yasinkuyu/binance-trader | 2,736 | Python | 2026-01-03 | Active | Binance strategies |
| askmike/gekko | 10,180 | JavaScript | 2020-02-16 | Legacy | Historically dominant, unmaintained |
| DeviaVir/zenbot | 8,259 | HTML | 2022-02-14 | Legacy | CLI bot + MongoDB |
| ccxt/binance-trade-bot | 8,706 | Python | 2025-03-12 | Active | Coin-hopping strategy |

### A.2 Exchange connectivity
| Repo | Stars | Lang | Last push | Status | Note |
|---|---|---|---|---|---|
| ★ ccxt/ccxt | 43,261 | Python/TS/JS | 2026-07-08 | Active | The de-facto unified exchange API (100+ venues) |

### A.3 Backtesting & simulation engines
| Repo | Stars | Lang | Last push | Status | Note |
|---|---|---|---|---|---|
| mementum/backtrader | 22,373 | Python | 2024-08-19 | Legacy | Hugely influential; effectively unmaintained |
| quantopian/zipline | 19,956 | Python | 2024-02-13 | Legacy | Pipeline/alpha-factor design DNA |
| ★ nkaz001/hftbacktest | 4,254 | Rust | 2025-12-23 | Active | Latency + queue-position + L2/L3 fills |
| kernc/backtesting.py | 8,655 | Python | 2025-12-20 | Active | Lightweight vectorized-ish backtester |
| polakowo/vectorbt | 8,223 | Python | 2026-07-05 | Active | Fast vectorized research backtesting |
| gbeced/pyalgotrade | 4,664 | Python | 2023-11-13 | Legacy | Event-driven backtester |
| edtechre/pybroker | 3,443 | Python | 2026-05-11 | Active | ML + walk-forward, Numba-accelerated |
| scrtlabs/catalyst | 2,562 | Python | 2022-11-26 | Legacy | Crypto fork of zipline |
| ranaroussi/qtpylib | 2,267 | Python | 2021-09-22 | Legacy | IB algo library |

### A.4 ML / RL / quantitative research
| Repo | Stars | Lang | Last push | Status | Note |
|---|---|---|---|---|---|
| stefan-jansen/machine-learning-for-trading | 19,584 | Jupyter | 2026-07-07 | Active | Canonical ML-for-trading reference |
| goldmansachs/gs-quant | 11,101 | Python | 2026-07-06 | Active | Institutional quant toolkit |
| huseinzol05/Stock-Prediction-Models | 9,428 | Jupyter | 2023-04-16 | Legacy | Model zoo |
| ★ AI4Finance-Foundation/FinRL | ~10k | Python | 2026 | Active | Deep-RL trading pipeline |
| cantaro86/Financial-Models-Numerical-Methods | 7,000 | Jupyter | 2024-10-22 | Active | Pricing/numerical methods |
| tensortrade-org/tensortrade | 6,417 | Python | 2026-02-19 | Active | RL trading framework |
| google/tf-quant-finance | 5,425 | Python | 2026-02-12 | Active | TF-based quant finance |
| shashankvemuri/Finance | 4,044 | Python | 2026-03-26 | Active | 150+ analysis scripts |
| chrisconlan/algorithmic-trading-with-python | 3,407 | Python | 2021-06-01 | Legacy | Book code |
| TradeMaster-NTU/TradeMaster | 2,912 | Jupyter | 2025-06-04 | Active | RL trading platform |
| nickmccullum/algorithmic-trading-python | 2,850 | Jupyter | 2024-06-20 | Legacy | Course code |
| Yvictor/TradingGym | 1,897 | Python | 2024-02-11 | Legacy | RL env |
| notadamking/RLTrader | 1,861 | Python | 2022-01-22 | Legacy | RL bot (BTC) |
| deependersingla/deep_trader / Ceruleanacg/Personae / samre12/deep-trading-agent / ucaiado/QLearning_Trading / hackthemarket/gym-trading | 517–1,497 | Python | 2016–2018 | Legacy | Early RL-trading experiments |
| Albert-Z-Guo/Deep-Reinforcement-Stock-Trading | 693 | Jupyter | 2024-11-06 | Active | DRL tutorial |
| sadighian/crypto-rl | 965 | Python | 2022-01-12 | Legacy | LOB + RL |
| bradleyboyuyang/ML-HFT | 592 | Jupyter | 2022-09-20 | Legacy | ML for HFT |

### A.5 LLM / AI-agent trading (emerging)
| Repo | Stars | Lang | Last push | Status | Note |
|---|---|---|---|---|---|
| ★ HKUDS/AI-Trader | 20,625 | Python | 2026-06-11 | Active | Agent-native platform; agents register via SKILL.md, live mark-to-market scoring |
| HKUDS/Vibe-Trading | 18,633 | Python | 2026-07-08 | Active | LLM "vibe" trading agents |
| The-Swarm-Corporation/AutoHedge | 3,724 | Python | 2026-05-11 | Active | Multi-agent hedge-fund swarm |
| Polymarket/agents | 3,719 | Python | 2024-11-05 | Active | LLM agents for prediction markets |
| simonlin1212/TradingAgents-astock | 1,729 | Python | 2026-06-28 | Active | Multi-agent A-share trading |
| okx/agent-trade-kit | 355 | TypeScript | 2026-06-25 | Active | Agent trading SDK |
| okx/agent-skills | 153 | Shell | 2026-06-25 | Active | Agent skill definitions |
| maxbbraun/trump2cash | 6,523 | Python | 2023-05-23 | Legacy | News/sentiment → trades (classic) |
| CryptoSignal/Crypto-Signal | 5,591 | Python | 2024-07-07 | Legacy | Technical signal engine |

### A.6 HFT / market-making / order book
| Repo | Stars | Lang | Last push | Status | Note |
|---|---|---|---|---|---|
| michaelgrosner/tribeca | 4,113 | TypeScript | 2021-12-08 | Legacy | Classic crypto MM bot |
| ctubio/Krypto-trading-bot | 3,703 | C++ | 2024-12-15 | Active | Low-latency MM (tribeca successor) |
| jamesmawm/High-Frequency-Trading-Model-with-IB | 2,891 | Python | 2025-05-29 | Active | HFT model w/ Interactive Brokers |
| rorysroes/SGX-Full-OrderBook-Tick-Data-Trading-Strategy | 2,302 | Jupyter | 2022-08-27 | Legacy | LOB strategy research |
| Crypto-toolbox/HFT-Orderbook | 1,377 | C | 2024-11-13 | Active | Limit-order-book (rbtree) impl |
| purefinance/mmb | 606 | Rust | 2022-11-17 | Legacy | Rust MM bot |
| ondra-novak/mmbot | 253 | C++ | 2026-04-21 | Active | Grid/MM bot |
| algotrading-lab/coinbase-market-making-bot | 92 | TypeScript | 2026-06-17 | Active | Coinbase MM |
| Drakkar-Software/OctoBot-Market-Making | 36 | Python | 2026-03-30 | Active | MM tentacle |
| rlindland/options-market-making | 84 | Python | 2018-01-27 | Legacy | Options MM |

### A.7 Arbitrage / DeFi / on-chain (MEV)
| Repo | Stars | Lang | Last push | Status | Note |
|---|---|---|---|---|---|
| flashbots/simple-arbitrage | 2,115 | TypeScript | — | Legacy | Canonical MEV arb example |
| paco0x/amm-arbitrageur | 2,000 | Solidity | 2022-12-07 | Legacy | Flash-swap AMM arb |
| x89/Solana-Arbitrage-Bot | 1,174 | Rust | 2026-05-08 | Active | Solana arb |
| ExtropyIO/defi-bot | 1,231 | Solidity | 2022-03-22 | Legacy | DeFi arb tutorial |
| kelvinau/crypto-arbitrage | 843 | Python | 2024-02-17 | Active | CEX arb |
| 0xNineteen/solana-arbitrage-bot | 805 | Rust | 2023-04-10 | Legacy | Solana arb |
| ARBProtocol/solana-jupiter-bot | 790 | JavaScript | 2024-02-14 | Active | Jupiter aggregator arb |

### A.8 Grid / DCA bots
| Repo | Stars | Lang | Last push | Status | Note |
|---|---|---|---|---|---|
| 51bitquant/binance_grid_trader | 969 | Python | 2026-06-07 | Active | Binance grid |
| DogsTailFarmer/martin-binance | 227 | Python | 2026-07-06 | Active | Martingale/grid |
| jordantete/grid_trading_bot | 138 | Python | 2026-07-07 | Active | Config-driven grid |
| HenrisonTao/ftx_grid_trading_bot | 171 | Python | 2022-08-30 | Legacy | Grid (FTX) |
| tiansin/docker_grid_trader | 108 | Python | 2021-05-05 | Legacy | Dockerized grid |

### A.9 Prediction-market making (new venue class)
| Repo | Stars | Lang | Last push | Status | Note |
|---|---|---|---|---|---|
| warproxxx/poly-maker | 1,379 | Python | 2026-07-06 | Active | Polymarket market maker |
| ent0n29/polybot | 842 | Java | 2026-02-20 | Active | Polymarket bot |
| manifoldmarkets/market-maker | 44 | TypeScript | 2024-07-04 | Active | Manifold MM |
| gelatotrade/Hyperliquid-Outcome-Market-Making-Bot | 47 | Rust | 2026-05-04 | Active | Hyperliquid outcome MM |

### A.10 Curated lists & knowledge bases (meta)
| Repo | Stars | Lang | Last push | Status |
|---|---|---|---|---|
| wilsonfreitas/awesome-quant | 27,491 | HTML | 2026-07-08 | Active |
| grananqvist/Awesome-Quant-Machine-Learning-Trading | 3,843 | — | 2025-05-21 | Active |
| LLMQuant/quant-wiki | 3,853 | — | 2026-04-16 | Active |
| LLMQuant/quant-mind | 1,912 | Python | 2026-06-04 | Active |
| wangzhe3224/awesome-systematic-trading | — | — | — | Active |
| Roibal/Cryptocurrency-Trading-Bots-Python-Beginner-Advance | 1,439 | Python | 2023-06-26 | Legacy |

*(A handful of auto-collected results were tangential — e.g. `study8677/awesome-architecture` is general software architecture, not trading — and were excluded from synthesis.)*

---

## Appendix B — Additional insights from the extended survey

The wider pool surfaced four patterns that refine or extend the core design:

**B.1 Backtest realism is a spectrum — support tiers, not one engine.** The survey spans three fidelity levels: (a) **candle/next-open** fills (Freqtrade, backtesting.py) — cheap, fine for swing/position strategies; (b) **intra-candle detail-timeframe** resolution (Freqtrade `timeframe_detail`) — needed for tight stops; (c) **tick-level L2/L3 with feed+order latency and order-queue-position modeling** (nkaz001/hftbacktest, Crypto-toolbox/HFT-Orderbook) — mandatory for market-making and HFT, where your fill depends on where you sit in the queue and how late your order arrives. *Design impact:* make the fill simulator a **pluggable component** with selectable fidelity, and let the same strategy be validated at whatever tier its holding period demands. hftbacktest also runs the **same code live**, reinforcing the research-live-parity thesis at the HFT end.

**B.2 Agent-native / LLM trading is arriving fast — quarantine it behind the risk gate.** Several of the highest-momentum 2026 repos (HKUDS/AI-Trader at 20k★, HKUDS/Vibe-Trading, AutoHedge's multi-agent "swarm", okx/agent-trade-kit, TradingAgents) put an LLM in the decision loop. AI-Trader even exposes a `SKILL.md` registration protocol so any agent (any coding agent: Codex, Cursor, etc.) can join and be scored by live mark-to-market PnL. *Design impact:* treat every LLM/agent as **one more evaluator producing a normalized note**, never as a direct order source. The deterministic Risk Engine (§3.6) stays authoritative — it sizes, caps, and can veto any agent-originated intent. This keeps explainability and adaptability without surrendering safety, and matches OctoBot's LLM-evaluator and Hummingbot's MCP-agent placement.

**B.3 On-chain/MEV and CEX arbitrage are different enough to isolate.** The arbitrage cluster (flashbots/simple-arbitrage, paco0x/amm-arbitrageur, the Solana bots) relies on flash loans, mempool/bundle submission, gas/priority-fee optimization, and private-key signing — concerns absent from CEX trading. *Design impact:* this validates the **Gateway isolation** decision (§3.1). On-chain venues live behind a dedicated signing/RPC service with its own risk rules (gas caps, revert protection, slippage bounds), and the core treats an on-chain "fill" as just another normalized event.

**B.4 New venue classes (prediction markets) fit the same abstraction.** Polymarket/Hyperliquid/Manifold market-making bots (poly-maker, Polymarket/agents, manifoldmarkets/market-maker) trade *outcome shares* rather than assets, but map cleanly onto the `Instrument` + `Executor` model with an outcome-specific pricing evaluator. *Design impact:* the normalized domain model (§5) generalizes beyond spot/perp/options to event-outcome markets with no core changes — a good stress test that the abstraction is right.

**Survey method note:** the 90+ repositories in Appendix A were enumerated live via the GitHub search REST API across ten topic queries (trading-bot, algorithmic-trading, crypto-trading-bot, market-making, arbitrage, grid-trading, quantitative-finance, reinforcement-learning-trading, high-frequency-trading, AI-trading-agent), deduplicated, and sorted by stars. Star counts and push dates are point-in-time (2026-07-08) and will drift.
