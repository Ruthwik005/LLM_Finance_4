import { describe, it, expect, vi } from "vitest";
import { createRoot } from "react-dom/client";
import { act } from "react";
import App from "./App";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };

const fills = [
  { episode: 0, side: "bid", price: 100, volume: 1 },
  { episode: 0, side: "ask", price: 101, volume: 2 },
];
const routes = {
  "/runs": { run_ids: ["demo_run"] },
  "/runs/demo_run": { run_id: "demo_run", variant: "SAC_LOB_ONLY", seed: 1, model_path: "m.zip", created_at: "2026-01-01T00:00:00Z" },
  "/runs/demo_run/metrics": { mean_pnl: 1.5, median_pnl: 1, std_pnl: 2, map: 0 },
  "/runs/demo_run/episodes": [{ episode: 0, pnl: 1 }, { episode: 1, pnl: -2 }],
  "/runs/demo_run/fills": fills,
  "/runs/demo_run/equity-curve": [{ episode: 0, cumulative_pnl: 1 }, { episode: 1, cumulative_pnl: -1 }],
  "/runs/demo_run/pnl-attribution": { run_id: "demo_run", spread_capture_pnl: -1, inventory_pnl: 2.5, fees_paid: 0.1, rebates: 0, total: 1.4 },
  "/runs/demo_run/training-curve": { run_id: "demo_run", points: [
    { timesteps: 300, train_reward_mean: null, val_reward_mean: -1.4, val_pnl_mean: 0.1, val_map: 0.3, ent_coef: null, critic_loss: null, actor_loss: null, is_best: true },
    { timesteps: 600, train_reward_mean: -1, val_reward_mean: -1.5, val_pnl_mean: 0.2, val_map: 0.4, ent_coef: 0.4, critic_loss: 1, actor_loss: -2, is_best: false }] },
  "/runs/demo_run/episodes/0/replay": {
    run_id: "demo_run", episode: 0, n_steps_total: 2, n_points: 2, downsampled: false, stride: 1,
    steps: [0, 1].map((t) => ({ step: t, timestamp: null, mid: 100 + t, bid_1: 99, bid_2: 98, ask_1: 101, ask_2: 102, bid_vol_1: 1, bid_vol_2: 1, ask_vol_1: 1, ask_vol_2: 1, inventory: t, mtm_pnl: t, placed_volume: 1, canceled_volume: 1 })),
    fills: [{ side: "bid", price: 99, volume: 1, step: 1, timestamp: null }],
    summary: { n_fills: 1, filled_volume: 1, final_mtm_pnl: 1, max_abs_inventory: 1, avg_quoted_spread: 2, cancel_volume: 2, placed_volume: 2 },
  },
  "/experiments": { experiments: [{ experiment_id: "exp1", data_mode: "synthetic_smoke_test", created_utc: null, n_variants: 2, has_summary_file: true }] },
  "/experiments/exp1": {
    experiment_id: "exp1", data_mode: "synthetic_smoke_test", seeds: [0, 1], reference_variant: "SAC_LOB_ONLY", comparison_label: "exploratory, small n",
    comparison_note: "x",
    variants: [
      { variant: "SAC_LOB_ONLY", n_seeds: 2, metrics: { mean_pnl: { mean: 1, std: 0.5, n_seeds: 2 }, max_drawdown: { mean: 3, std: 1, n_seeds: 2 } } },
      { variant: "SAC_FULL", n_seeds: 2, metrics: { mean_pnl: { mean: 2, std: 0.1, n_seeds: 2 }, max_drawdown: { mean: 2, std: 1, n_seeds: 2 } } },
    ],
    comparisons: [{ variant: "SAC_FULL", metric: "mean_pnl", n_a: 2, n_b: 2, mean_diff: 1, t: 1, p_value: 0.4, cohens_d: 0.7, note: "n" }],
  },
  "/signals": {
    available: true, split: "test", n_rows_total: 4, n_points: 3, stride: 1, notes: [],
    timestamps: ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", "2024-01-01T00:02:00Z"], midprice: [1, 2, 3],
    series: { sentiment_ewm_30: [0, 0.1, 0.2], article_count_30: [0, 1, 2], transformer_prob_h1_down: [0.3, 0.3, 0.3], transformer_prob_h1_flat: [0.4, 0.4, 0.4], transformer_prob_h1_up: [0.3, 0.3, 0.3], transformer_signal_valid: [0, 1, 1] },
  },
};

async function mount(fetchImpl) {
  global.fetch = vi.fn(fetchImpl);
  const el = document.createElement("div");
  document.body.appendChild(el);
  const root = createRoot(el);
  await act(async () => { root.render(<App />); });
  await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
  return el;
}
const tabs = (el) => [...el.querySelectorAll('[role="tab"]')];

describe("App smoke", () => {
  it("renders every view with data without throwing", async () => {
    const el = await mount(async (url) => {
      const path = new URL(url).pathname;
      return { ok: path in routes, status: 200, json: async () => routes[path] };
    });
    // synthetic runs must NOT claim real-LOB provenance
    expect(el.textContent).not.toContain("Data mode: Mode B");
    expect(el.textContent).toContain("SOFTWARE SMOKE TEST ONLY");
    for (const t of tabs(el)) {
      await act(async () => { t.click(); });
      await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
      expect(el.querySelector("main").textContent.length).toBeGreaterThan(10);
    }
  });

  it("shows the new screens' real content", async () => {
    const el = await mount(async (url) => {
      const path = new URL(url).pathname;
      return { ok: path in routes, status: 200, json: async () => routes[path] };
    });
    const go = async (label) => {
      const t = tabs(el).find((x) => x.textContent.includes(label));
      await act(async () => { t.click(); });
      await act(async () => { await new Promise((r) => setTimeout(r, 30)); });
      return el.querySelector("main").textContent;
    };
    expect(await go("Episode Replay")).toContain("Filled volume");
    expect(await go("Training")).toContain("Best checkpoint at");
    const behavior = await go("Market-Making Behavior");
    expect(behavior).toContain("Cancel ratio");
    expect(behavior).toContain("PnL attribution");
    const compare = await go("Model Comparison");
    expect(compare).toContain("Multi-seed experiment");
    expect(compare).toContain("Exploratory — small number of seeds; not a claim of statistical significance.");
    expect(await go("Sentiment")).toContain("Feature artifacts");
  });

  it("falls back to honest placeholders when /signals is unavailable", async () => {
    const el = await mount(async (url) => {
      const path = new URL(url).pathname;
      if (path === "/signals") return { ok: true, status: 200, json: async () => ({ available: false, reason: "no features" }) };
      return { ok: path in routes, status: 200, json: async () => routes[path] };
    });
    const t = tabs(el).find((x) => x.textContent.includes("Sentiment"));
    await act(async () => { t.click(); });
    await act(async () => { await new Promise((r) => setTimeout(r, 30)); });
    const txt = el.querySelector("main").textContent;
    expect(txt).toContain("Not run / insufficient data");
    expect(txt).toContain("no features");
  });

  it("renders every view when all requests fail", async () => {
    const el = await mount(async () => ({ ok: false, status: 500, json: async () => ({ detail: "boom" }) }));
    for (const t of tabs(el)) {
      await act(async () => { t.click(); });
      await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
      expect(el.querySelector("main").textContent).toContain("Not run / insufficient data");
    }
  });
});
