import { describe, it, expect, vi, afterEach } from "vitest";
import { listRuns, getRunMetrics, getRunDetail, getRunEpisodes, getRunFills, getEquityCurve } from "./api";

describe("api client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("listRuns returns the run_ids array from the response", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ run_ids: ["run_a", "run_b"] }),
    });
    const runs = await listRuns();
    expect(runs).toEqual(["run_a", "run_b"]);
  });

  it("getRunMetrics throws with the server's detail message on 404", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Run 'missing' not found" }),
    });
    await expect(getRunMetrics("missing")).rejects.toThrow("Run 'missing' not found");
  });

  it("getRunDetail returns the run metadata", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ variant: "v1", seed: 42, model_path: "/path/to/model", created_at: "2026-09-25T10:00:00Z" }),
    });
    const detail = await getRunDetail("run_123");
    expect(detail).toEqual({ variant: "v1", seed: 42, model_path: "/path/to/model", created_at: "2026-09-25T10:00:00Z" });
  });

  it("getRunEpisodes returns the episodes array", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ episodes: [{ id: 1, reward: 10 }, { id: 2, reward: 20 }] }),
    });
    const episodes = await getRunEpisodes("run_123");
    expect(episodes).toEqual({ episodes: [{ id: 1, reward: 10 }, { id: 2, reward: 20 }] });
  });

  it("getRunFills returns the fills array", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ fills: [{ episode: 1, side: "BUY", price: 100, volume: 10 }] }),
    });
    const fills = await getRunFills("run_123");
    expect(fills).toEqual({ fills: [{ episode: 1, side: "BUY", price: 100, volume: 10 }] });
  });

  it("getEquityCurve returns the equity curve data", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ cumulative_pnl: [0, 10, 20, 15] }),
    });
    const curve = await getEquityCurve("run_123");
    expect(curve).toEqual({ cumulative_pnl: [0, 10, 20, 15] });
  });
});
