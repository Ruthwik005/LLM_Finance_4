# UI redesign notes

## Design tokens (src/index.css)
- Dark-first; light via `[data-theme="light"]` or `prefers-color-scheme` when no stored choice. Toggle persisted in localStorage (try/catch).
- Surfaces: bg #0b0f17, surface #111827, surface-2 #151c2b, border rgba(255,255,255,.08). Accent indigo->violet gradient. Semantic green/red/amber/blue with soft and text variants.
- Fonts: Inter (UI), JetBrains Mono (numbers, tabular-nums). 4px spacing scale (--s1..--s10), radii 8/12/14.
- Motion: fade-up, hover lift, KPI count-up, shimmer skeletons; all disabled under prefers-reduced-motion.

## Components (src/components)
Sidebar, Topbar, RunSelect (custom listbox), KpiCard, CountUp, Sparkline, ChartCard, ChartTooltip, EquityChart, Skeleton(+Card/Kpis), EmptyState, StatusChip, DataTable, Segmented, Logo, NoRun, nav.js, chartTheme.js.
Views: OverviewView, PnlView, RiskView, BehaviorView, SignalsView, CompareView.
Lib: format.js, derive.js (unit tested), hooks.js (useAsync/useTheme), runData.js (parallel fetch, tolerant), variant.js.

## Views
1. Overview: 6 KPI cards with sparklines, run metadata, equity preview, data-mode notice.
2. PnL / Equity: stats strip (best/worst, win rate, avg, max drawdown), gradient area chart with zero line, colored episode bars.
3. Inventory & Risk: inventory derived from fills (reset per episode), split long/short area, KPIs incl. PnL/MAP ("—" if MAP is 0/missing).
4. Market-Making Behavior: side/episode filters, sticky zebra table with BID/ASK pills, show-more paging, donut + volume bars, summary KPIs.
5. Sentiment & Signals: three offline-pipeline status cards, SVG as-of timeline, "Not run / insufficient data" empty area. No fabricated numbers.
6. Model Comparison: all-run metrics (allSettled), best-per-column highlight, clickable rows, grouped mean/median bars.

## Honesty
Persistent chips: the Mode B data-mode text and "Research backtest · not financial advice". The API does not store data_mode, so backend was not changed; the red "SOFTWARE SMOKE TEST ONLY - NOT A FINANCIAL PERFORMANCE RESULT" chip is shown heuristically when the selected run id/variant contains demo/synthetic/smoke/fake.
