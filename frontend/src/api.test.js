import { describe, it, expect, vi, afterEach } from "vitest";
import { listRuns, getRunMetrics } from "./api";

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
});
