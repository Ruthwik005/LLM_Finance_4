import { useEffect, useState } from "react";
import { listRuns } from "../api";

export default function RunSelector({ selectedRun, onSelect }) {
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    listRuns()
      .then(setRuns)
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <p>Not run / insufficient data</p>;
  if (runs.length === 0) return <p>No runs found yet.</p>;

  return (
    <select value={selectedRun || ""} onChange={(e) => onSelect(e.target.value)}>
      <option value="" disabled>
        Select an experiment run
      </option>
      {runs.map((runId) => (
        <option key={runId} value={runId}>
          {runId}
        </option>
      ))}
    </select>
  );
}
