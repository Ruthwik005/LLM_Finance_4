import { useAsync } from "./hooks";
import { getRunDetail, getRunMetrics, getRunEpisodes, getRunFills, getEquityCurve, getPnlAttribution } from "../api";

const LOADERS = {
  detail: getRunDetail,
  metrics: getRunMetrics,
  episodes: getRunEpisodes,
  fills: getRunFills,
  equity: getEquityCurve,
  attribution: getPnlAttribution,
};

/**
 * Fetch several resources for a run in parallel, tolerating individual failures.
 * data[key] is the payload or null if that request failed. error is set only if ALL failed.
 */
export function useRunData(runId, refreshKey, keys) {
  const joined = keys.join(",");
  return useAsync(async () => {
    if (!runId) return {};
    const results = await Promise.allSettled(keys.map((k) => LOADERS[k](runId)));
    const out = {};
    let firstErr = null;
    results.forEach((r, i) => {
      out[keys[i]] = r.status === "fulfilled" ? r.value : null;
      if (r.status === "rejected" && !firstErr) firstErr = r.reason;
    });
    if (results.every((r) => r.status === "rejected")) throw firstErr || new Error("Request failed");
    return out;
  }, [runId, refreshKey, joined]);
}
