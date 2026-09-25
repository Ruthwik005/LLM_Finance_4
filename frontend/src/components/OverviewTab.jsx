import { useEffect, useState } from "react";
import { getRunDetail } from "../api";

export default function OverviewTab({ runId }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    getRunDetail(runId).then(setDetail).catch((err) => setError(err.message));
  }, [runId]);

  if (!runId) return <p>Select a run to see its overview.</p>;
  if (error) return <p>Not run / insufficient data</p>;
  if (!detail) return <p>Loading…</p>;

  return (
    <dl>
      <dt>Variant</dt><dd>{detail.variant}</dd>
      <dt>Seed</dt><dd>{detail.seed}</dd>
      <dt>Model path</dt><dd>{detail.model_path}</dd>
      <dt>Created at</dt><dd>{detail.created_at}</dd>
    </dl>
  );
}
