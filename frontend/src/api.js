const BASE_URL = "http://localhost:8000";

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
