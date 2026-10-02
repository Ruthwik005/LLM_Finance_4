import { describe, it, expect } from "vitest";
import {
  inventoryPath, maxAbsInventory, meanAbsInventory, maxDrawdown, winRate,
  episodeStats, pnlPerMap, fillStats, downsample, bestIndex,
} from "./derive";

describe("inventoryPath", () => {
  it("accumulates bid as + and ask as -, resetting per episode", () => {
    const p = inventoryPath([
      { episode: 0, side: "bid", volume: 2 },
      { episode: 0, side: "bid", volume: 1 },
      { episode: 0, side: "ask", volume: 4 },
      { episode: 1, side: "ask", volume: 1 },
    ]);
    expect(p.map((r) => r.inventory)).toEqual([2, 3, -1, -1]);
    expect(p[2].pos).toBe(0);
    expect(p[2].neg).toBe(-1);
  });
  it("handles empty / invalid input", () => {
    expect(inventoryPath([])).toEqual([]);
    expect(inventoryPath(undefined)).toEqual([]);
    expect(maxAbsInventory([])).toBeNull();
    expect(meanAbsInventory([])).toBeNull();
  });
  it("computes max and mean absolute inventory", () => {
    const p = inventoryPath([
      { episode: 0, side: "ask", volume: 3 },
      { episode: 0, side: "bid", volume: 5 },
    ]);
    expect(maxAbsInventory(p)).toBe(3);
    expect(meanAbsInventory(p)).toBe(2.5);
  });
});

describe("maxDrawdown", () => {
  it("returns the largest peak-to-trough drop", () => {
    const eq = [1, 5, 2, 6, 1, 3].map((v, i) => ({ episode: i, cumulative_pnl: v }));
    expect(maxDrawdown(eq)).toBe(5);
  });
  it("is 0 for monotonic gains and null for empty", () => {
    expect(maxDrawdown([{ cumulative_pnl: 1 }, { cumulative_pnl: 2 }])).toBe(0);
    expect(maxDrawdown([])).toBeNull();
  });
  it("counts a drop below the initial zero equity", () => {
    expect(maxDrawdown([{ cumulative_pnl: -3 }])).toBe(3);
  });
});

describe("winRate / episodeStats", () => {
  it("computes win rate", () => {
    expect(winRate([{ pnl: 1 }, { pnl: -1 }, { pnl: 0 }, { pnl: 2 }])).toBe(0.5);
    expect(winRate([])).toBeNull();
  });
  it("finds best, worst and average", () => {
    const s = episodeStats([{ episode: 0, pnl: 1 }, { episode: 1, pnl: -3 }, { episode: 2, pnl: 5 }]);
    expect(s.best.episode).toBe(2);
    expect(s.worst.episode).toBe(1);
    expect(s.avg).toBe(1);
    expect(episodeStats([])).toBeNull();
  });
});

describe("pnlPerMap", () => {
  it("divides pnl by map", () => expect(pnlPerMap(10, 2)).toBe(5));
  it("returns null when MAP is zero, missing or non-finite", () => {
    expect(pnlPerMap(10, 0)).toBeNull();
    expect(pnlPerMap(10, undefined)).toBeNull();
    expect(pnlPerMap(10, Infinity)).toBeNull();
    expect(pnlPerMap(undefined, 2)).toBeNull();
  });
});

describe("fillStats / helpers", () => {
  it("summarises fills", () => {
    const s = fillStats([
      { side: "bid", price: 100, volume: 1 },
      { side: "ask", price: 102, volume: 3 },
    ]);
    expect(s).toMatchObject({ count: 2, bidCount: 1, askCount: 1, bidVolume: 1, askVolume: 3, totalVolume: 4, avgPrice: 101, avgSize: 2 });
    expect(fillStats([])).toBeNull();
  });
  it("downsamples", () => {
    expect(downsample([1, 2, 3], 10)).toEqual([1, 2, 3]);
    expect(downsample(Array.from({ length: 100 }, (_, i) => i), 5)).toHaveLength(5);
  });
  it("bestIndex ignores non-numbers", () => {
    expect(bestIndex([1, null, 5, 3])).toBe(2);
    expect(bestIndex([4, 2, 9], false)).toBe(1);
    expect(bestIndex([null])).toBe(-1);
  });
});
