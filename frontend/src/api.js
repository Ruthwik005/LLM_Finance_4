const BASE_URL = import.meta.env?.VITE_API_URL ?? "http://localhost:8000";

async function request(path) {
  const resp = await fetch(`${BASE_URL}${path}`);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `Request to ${path} failed with status ${resp.status}`);
  }
  return resp.json();
}

export async function listRuns() {
  const data = await request("/runs");
  return data.run_ids;
}

export function getRunDetail(runId) {
  return request(`/runs/${runId}`);
}

export function getRunMetrics(runId) {
  return request(`/runs/${runId}/metrics`);
}

export function getRunEpisodes(runId) {
  return request(`/runs/${runId}/episodes`);
}

export function getRunFills(runId) {
  return request(`/runs/${runId}/fills`);
}

export function getEquityCurve(runId) {
  return request(`/runs/${runId}/equity-curve`);
}

export function getRunsSummary() {
  return request("/runs/summary").then((d) => d.runs);
}

export function getReplay(runId, episode, maxPoints = 2000) {
  return request(`/runs/${encodeURIComponent(runId)}/episodes/${episode}/replay?max_points=${maxPoints}`);
}

export function getTrainingCurve(runId) {
  return request(`/runs/${encodeURIComponent(runId)}/training-curve`).then((d) => d.points);
}

export function getPnlAttribution(runId) {
  return request(`/runs/${encodeURIComponent(runId)}/pnl-attribution`);
}

export function listExperiments() {
  return request("/experiments").then((d) => d.experiments);
}

export function getExperiment(experimentId) {
  return request(`/experiments/${encodeURIComponent(experimentId)}`);
}

export function getSignals(split = "test", maxPoints = 2000) {
  return request(`/signals?split=${split}&max_points=${maxPoints}`);
}
