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

  it("renders every view when all requests fail", async () => {
    const el = await mount(async () => ({ ok: false, status: 500, json: async () => ({ detail: "boom" }) }));
    for (const t of tabs(el)) {
      await act(async () => { t.click(); });
      await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
      expect(el.querySelector("main").textContent).toContain("Not run / insufficient data");
    }
  });
});
