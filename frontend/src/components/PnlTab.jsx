import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getEquityCurve } from "../api";

export default function PnlTab({ runId }) {
  const [curve, setCurve] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    getEquityCurve(runId).then(setCurve).catch((err) => setError(err.message));
  }, [runId]);

  if (!runId) return <p>Select a run to see PnL / equity.</p>;
  if (error || (curve && curve.length === 0)) return <p>Not run / insufficient data</p>;
  if (!curve) return <p>Loading…</p>;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={curve}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="episode" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="cumulative_pnl" stroke="#2563eb" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
