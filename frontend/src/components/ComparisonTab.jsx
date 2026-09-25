import { useEffect, useState } from "react";
import { getRunMetrics } from "../api";

export default function ComparisonTab({ runId }) {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    getRunMetrics(runId).then(setMetrics).catch((err) => setError(err.message));
  }, [runId]);

  if (!runId) return <p>Select a run to see its metrics.</p>;
  if (error) return <p>Not run / insufficient data</p>;
  if (!metrics) return <p>Loading…</p>;
  if (Object.keys(metrics).length === 0) return <p>Not run / insufficient data</p>;

  return (
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        {Object.entries(metrics).map(([name, value]) => (
          <tr key={name}><td>{name}</td><td>{value}</td></tr>
        ))}
      </tbody>
    </table>
  );
}
