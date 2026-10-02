import { describe, it, expect } from "vitest";
import { prepareSteps, markerSize, fillMarkers, visibleRows, visibleFills, alignComparison, episodeKpis, deltaPnl, indexAtOrBefore, defaultCompareRun, priceDomain } from "./replay";

const steps = [
  { step: 0, mid: 100, bid_1: 99, bid_2: 98, ask_1: 101, ask_2: 102, inventory: 0, mtm_pnl: 0, canceled_volume: 1 },
  { step: 5, mid: 101, bid_1: 100, bid_2: 99, ask_1: 103, ask_2: 104, inventory: -2, mtm_pnl: 1, canceled_volume: 2 },
  { step: 9, mid: 100, bid_1: 99, bid_2: 98, ask_1: 102, ask_2: 103, inventory: 3, mtm_pnl: -1, canceled_volume: 3 },
];

describe("replay helpers", () => {
  it("prepares rows with spread and split inventory", () => {
    const r = prepareSteps(steps);
    expect(r[1].spread).toBe(3);
    expect(r[1].neg).toBe(-2);
    expect(r[1].pos).toBe(0);
    expect(r[2].pos).toBe(3);
    expect(prepareSteps(null)).toEqual([]);
  });
  it("scales marker size with sqrt(volume) and clamps", () => {
    expect(markerSize(0, 10)).toBe(3);
    expect(markerSize(10, 10)).toBe(8);
    expect(markerSize(2.5, 10)).toBeCloseTo(3 + 5 * Math.sqrt(0.25));
    expect(markerSize(99, 10)).toBe(8);
    expect(markerSize(1, 0)).toBe(3);
  });
  it("keeps only fills with a step", () => {
    const m = fillMarkers([{ side: "bid", price: 1, volume: 2, step: 3 }, { side: "ask", price: 2, volume: 1, step: null }]);
    expect(m).toHaveLength(1);
    expect(m[0].r).toBe(8);
  });
  it("reveals progressively inside the zoom window", () => {
    const r = prepareSteps(steps);
    expect(visibleRows(r, 5).map((x) => x.step)).toEqual([0, 5]);
    expect(visibleRows(r, 100, 4, 6).map((x) => x.step)).toEqual([5]);
    expect(visibleFills([{ step: 1 }, { step: 8 }], 4)).toHaveLength(1);
  });
  it("aligns a comparison run by forward fill", () => {
    const base = prepareSteps(steps);
    const cmp = prepareSteps([{ step: 2, mid: 1, bid_1: 1, bid_2: 1, ask_1: 1, ask_2: 1, inventory: 7, mtm_pnl: 4 }]);
    const a = alignComparison(base, cmp);
    expect(a[0].cmp_inventory).toBeNull();
    expect(a[1].cmp_inventory).toBe(7);
    expect(a[2].cmp_mtm_pnl).toBe(4);
    expect(alignComparison(base, [])).toBe(base);
  });
  it("uses the API summary for KPIs, else derives them", () => {
    const k = episodeKpis({ summary: { n_fills: 3, filled_volume: 4, final_mtm_pnl: 1.5, max_abs_inventory: 9, avg_quoted_spread: 2, cancel_volume: 7 }, fills: [], steps });
    expect(k).toMatchObject({ fills: 3, volume: 4, pnl: 1.5, maxAbsInventory: 9, avgSpread: 2, cancelVolume: 7 });
    const d = episodeKpis({ steps, fills: [{ volume: 1 }, { volume: 2 }] });
    expect(d.fills).toBe(2);
    expect(d.volume).toBe(3);
    expect(d.maxAbsInventory).toBe(3);
    expect(d.cancelVolume).toBe(6);
    expect(d.avgSpread).toBeCloseTo((2 + 3 + 3) / 3);
    expect(episodeKpis(null)).toBeNull();
  });
  it("computes delta, binary search, default baseline and price domain", () => {
    expect(deltaPnl(3, 1)).toBe(2);
    expect(deltaPnl(null, 1)).toBeNull();
    const r = prepareSteps(steps);
    expect(indexAtOrBefore(r, 6)).toBe(1);
    expect(indexAtOrBefore(r, -1)).toBe(-1);
    expect(defaultCompareRun(["a", "heuristic", "avellaneda_stoikov"], "a")).toBe("heuristic");
    expect(defaultCompareRun(["heuristic"], "heuristic")).toBe("");
    const [lo, hi] = priceDomain(r);
    expect(lo).toBeLessThan(98);
    expect(hi).toBeGreaterThan(104);
    expect(priceDomain([])).toEqual(["auto", "auto"]);
  });
});
