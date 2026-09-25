import { useEffect, useState } from "react";
import { getRunFills } from "../api";

export default function BehaviorTab({ runId }) {
  const [fills, setFills] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    getRunFills(runId).then(setFills).catch((err) => setError(err.message));
  }, [runId]);

  if (!runId) return <p>Select a run to see market-making behavior.</p>;
  if (error) return <p>Not run / insufficient data</p>;
  if (!fills) return <p>Loading…</p>;
  if (fills.length === 0) return <p>Not run / insufficient data</p>;

  return (
    <table>
      <thead><tr><th>Episode</th><th>Side</th><th>Price</th><th>Volume</th></tr></thead>
      <tbody>
        {fills.map((f, i) => (
          <tr key={i}><td>{f.episode}</td><td>{f.side}</td><td>{f.price}</td><td>{f.volume}</td></tr>
        ))}
      </tbody>
    </table>
  );
}
