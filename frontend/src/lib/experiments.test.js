import { describe, it, expect } from "vitest";
import { bestVariantIndex, errorBarData, availableMetrics, direction } from "./experiments";
import { invalidRanges, probHorizons, stackProbs, signalRows } from "./signals";
import { bestPoint, hasCurve, hasSeries, lastValue } from "./training";
import { attributionStack } from "./attribution";

const variants = [
  { variant: "A", n_seeds: 3, metrics: { mean_pnl: { mean: 1, std: 0.5, n_seeds: 3 }, max_drawdown: { mean: 4, std: null, n_seeds: 3 }, cancel_ratio: { mean: 0.9, std: 0, n_seeds: 3 } } },
  { variant: "B", n_seeds: 1, metrics: { mean_pnl: { mean: 2, std: null, n_seeds: 1 }, max_drawdown: { mean: 2, std: null, n_seeds: 1 }, cancel_ratio: { mean: 0.5, std: null, n_seeds: 1 } } },
];

describe("experiments", () => {
  it("picks the best per metric by direction", () => {
    expect(bestVariantIndex(variants, "mean_pnl")).toBe(1);
    expect(bestVariantIndex(variants, "max_drawdown")).toBe(1);
    expect(bestVariantIndex(variants, "cancel_ratio")).toBe(-1);
    expect(bestVariantIndex([variants[0]], "mean_pnl")).toBe(-1);
    expect(direction("map")).toBe("lower");
  });
  it("builds error-bar data with 0 error for undefined std", () => {
    const d = errorBarData(variants, "mean_pnl");
    expect(d[0]).toMatchObject({ variant: "A", mean: 1, err: 0.5, n_seeds: 3 });
    expect(d[1].err).toBe(0);
    expect(d[1].std).toBeNull();
    expect(errorBarData(variants, "nope")).toEqual([]);
  });
  it("lists metrics with the table metrics first", () => {
    expect(availableMetrics(variants)).toEqual(["mean_pnl", "max_drawdown", "cancel_ratio"]);
  });
});

describe("signals", () => {
  const payload = {
    available: true,
    timestamps: ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", "2024-01-01T00:02:00Z", "2024-01-01T00:03:00Z"],
    midprice: [1, 2, 3, 4],
    series: { transformer_signal_valid: [0, 0, 1, 0], transformer_prob_h1_down: [0.2, 0, 0.1, 0.5], transformer_prob_h1_flat: [0.6, 0, 0.8, 0.5], transformer_prob_h1_up: [0.2, 0, 0.1, 0.5], transformer_prob_h5_up: [1, 1, 1, 1] },
  };
  it("converts payload to rows", () => {
    const r = signalRows(payload);
    expect(r).toHaveLength(4);
    expect(r[0].mid).toBe(1);
    expect(r[1].t - r[0].t).toBe(60000);
    expect(signalRows({ available: false })).toEqual([]);
  });
  it("finds invalid ranges", () => {
    const r = signalRows(payload);
    expect(invalidRanges(r)).toEqual([[r[0].t, r[1].t], [r[3].t, r[3].t]]);
  });
  it("finds horizons and normalises probabilities", () => {
    expect(probHorizons(payload.series)).toEqual([1, 5]);
    const s = stackProbs(signalRows(payload), 1);
    expect(s[0].down + s[0].flat + s[0].up).toBeCloseTo(1);
    expect(s[1].flat).toBe(0); // all-zero row: no division blow-up
    expect(stackProbs(signalRows(payload), 5)[0].down).toBeNull();
  });
});

describe("training + attribution", () => {
  const pts = [{ timesteps: 1, is_best: true, val_pnl_mean: 1 }, { timesteps: 2, is_best: false, val_pnl_mean: null }, { timesteps: 3, is_best: true, val_pnl_mean: 2 }];
  it("finds best point and series presence", () => {
    expect(bestPoint(pts).timesteps).toBe(3);
    expect(bestPoint([])).toBeNull();
    expect(hasCurve([])).toBe(false);
    expect(hasSeries(pts, "val_pnl_mean")).toBe(true);
    expect(hasSeries(pts, "ent_coef")).toBe(false);
    expect(lastValue(pts, "val_pnl_mean")).toBe(2);
  });
  it("builds a signed attribution stack and omits undefined parts", () => {
    const a = attributionStack({ spread_capture_pnl: -1, inventory_pnl: 2.5, fees_paid: 0.1, rebates: undefined, total: 1.4 });
    expect(a.row[0]).toEqual({ name: "PnL", spread_capture_pnl: -1, inventory_pnl: 2.5, fees_paid: -0.1 });
    expect(a.parts.map((p) => p.key)).not.toContain("rebates");
    expect(a.total).toBe(1.4);
    expect(attributionStack({})).toBeNull();
    expect(attributionStack(null)).toBeNull();
  });
});
