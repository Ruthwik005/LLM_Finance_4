const isNum = (v) => typeof v === "number" && Number.isFinite(v);

export const HIGHER_BETTER = ["mean_pnl", "median_pnl", "total_pnl", "sharpe_like", "pnl_over_map", "spread_capture_pnl", "inventory_pnl", "n_fills", "fill_rate"];
export const LOWER_BETTER = ["max_drawdown", "std_pnl", "map", "max_abs_inventory", "fees_paid", "adverse_selection_ratio"];

export const METRIC_LABELS = {
  mean_pnl: "Mean PnL",
  median_pnl: "Median PnL",
  total_pnl: "Total PnL",
  sharpe_like: "Sharpe-like",
  pnl_over_map: "PnL / MAP",
  map: "MAP",
  max_drawdown: "Max drawdown",
  max_abs_inventory: "Max |inventory|",
  n_fills: "Fills",
  fill_rate: "Fill rate",
  cancel_ratio: "Cancel ratio",
  adverse_selection_ratio: "Adverse selection",
  avg_quoted_spread: "Avg quoted spread",
  avg_quote_distance: "Avg quote distance",
  spread_capture_pnl: "Spread capture",
  inventory_pnl: "Inventory PnL",
  fees_paid: "Fees paid",
  std_pnl: "Std PnL",
};

export const TABLE_METRICS = ["mean_pnl", "sharpe_like", "pnl_over_map", "map", "max_drawdown", "n_fills", "fill_rate", "cancel_ratio", "adverse_selection_ratio"];

export function metricLabel(m) {
  return METRIC_LABELS[m] || m;
}

/** "higher" | "lower" | null (no preferred direction). */
export function direction(metric) {
  if (HIGHER_BETTER.includes(metric)) return "higher";
  if (LOWER_BETTER.includes(metric)) return "lower";
  return null;
}

/** Index of the best variant for a metric (ties/undefined -> -1). Needs >= 2 variants with the metric. */
export function bestVariantIndex(variants, metric) {
  const dir = direction(metric);
  if (!dir) return -1;
  const vals = (variants || []).map((v) => v?.metrics?.[metric]?.mean);
  if (vals.filter(isNum).length < 2) return -1;
  let best = -1;
  vals.forEach((v, i) => {
    if (!isNum(v)) return;
    if (best < 0 || (dir === "higher" ? v > vals[best] : v < vals[best])) best = i;
  });
  return best;
}

/** Rows for a bar chart with error bars: {variant, mean, std, err (0 when undefined), n_seeds}. */
export function errorBarData(variants, metric) {
  return (variants || [])
    .map((v) => {
      const m = v?.metrics?.[metric];
      if (!m || !isNum(m.mean)) return null;
      return { variant: v.variant, mean: m.mean, std: isNum(m.std) ? m.std : null, err: isNum(m.std) ? m.std : 0, n_seeds: m.n_seeds };
    })
    .filter(Boolean);
}

/** Metrics available in at least one variant, ordered by TABLE_METRICS first. */
export function availableMetrics(variants) {
  const set = new Set();
  (variants || []).forEach((v) => Object.keys(v?.metrics || {}).forEach((m) => set.add(m)));
  const first = TABLE_METRICS.filter((m) => set.has(m));
  const rest = [...set].filter((m) => !first.includes(m)).sort();
  return [...first, ...rest];
}

export const EXPLORATORY_NOTE = "Exploratory — small number of seeds; not a claim of statistical significance.";
