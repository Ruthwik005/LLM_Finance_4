import { useEffect, useMemo, useState } from "react";
import ResponsiveContainer from "./AutoSize";
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Tooltip, XAxis, YAxis } from "recharts";
import { FlaskConical } from "lucide-react";
import ChartCard from "./ChartCard";
import ChartTooltip from "./ChartTooltip";
import DataTable from "./DataTable";
import Segmented from "./Segmented";
import StatusChip from "./StatusChip";
import { SkeletonCard } from "./Skeleton";
import { AXIS_LINE, AXIS_TICK, GRID } from "./chartTheme";
import { getExperiment, listExperiments } from "../api";
import { useAsync } from "../lib/hooks";
import { DASH, formatNumber } from "../lib/format";
import { EXPLORATORY_NOTE, TABLE_METRICS, availableMetrics, bestVariantIndex, errorBarData, metricLabel } from "../lib/experiments";

const BASELINES = ["HEURISTIC", "AVELLANEDA_STOIKOV"];
const fmt = (v) => (Number.isFinite(v) ? formatNumber(v, Math.abs(v) >= 100 ? 1 : 4) : DASH);

/** Returns null when no experiment aggregates exist (caller falls back to the per-run comparison). */
export default function ExperimentsPanel({ refreshKey }) {
  const list = useAsync(async () => listExperiments(), [refreshKey]);
  const experiments = useMemo(() => list.data || [], [list.data]);
  const [expId, setExpId] = useState(null);
  const [metric, setMetric] = useState("mean_pnl");
  const id = expId && experiments.some((e) => e.experiment_id === expId) ? expId : experiments[0]?.experiment_id;
  const detail = useAsync(async () => (id ? getExperiment(id) : null), [id, refreshKey]);
  const d = detail.data;
  const variants = useMemo(() => d?.variants || [], [d]);
  const metrics = useMemo(() => availableMetrics(variants), [variants]);
  useEffect(() => {
    if (metrics.length && !metrics.includes(metric)) setMetric(metrics[0]);
  }, [metrics, metric]);

  if (list.loading || (id && detail.loading && !d)) return <SkeletonCard h={260} />;
  if (list.error || !experiments.length || detail.error || !d || !variants.length) return null;

  const chartData = errorBarData(variants, metric).map((r) => ({ ...r, name: r.variant }));
  const tableMetrics = TABLE_METRICS.filter((m) => metrics.includes(m));
  const best = Object.fromEntries(tableMetrics.map((m) => [m, bestVariantIndex(variants, m)]));
  const columns = [
    {
      key: "variant",
      header: "Variant",
      render: (r) => (
        <span className="run-cell">
          <span className="mono">{r.variant}</span>
          {BASELINES.includes(r.variant) && <StatusChip tone="info">baseline</StatusChip>}
        </span>
      ),
    },
    { key: "n", header: "Seeds", align: "right", mono: true, render: (r) => r.n_seeds },
    ...tableMetrics.map((m) => ({
      key: m,
      header: metricLabel(m),
      align: "right",
      mono: true,
      render: (r, i) => {
        const x = r.metrics[m];
        if (!x) return DASH;
        const txt = (
          <>
            {fmt(x.mean)}
            <span className="cell-std">{Number.isFinite(x.std) ? `± ${fmt(x.std)}` : ""}</span>
          </>
        );
        return best[m] === i ? <span className="best">{txt}</span> : txt;
      },
    })),
  ];
  const cmpRows = (d.comparisons || []).filter((c) => c.metric === metric);
  const cmpCols = [
    { key: "variant", header: `Variant vs ${d.reference_variant}`, render: (r) => <span className="mono">{r.variant}</span> },
    { key: "n", header: "n (A / B)", align: "right", mono: true, render: (r) => `${r.n_a ?? DASH} / ${r.n_b ?? DASH}` },
    { key: "mean_diff", header: "Mean diff", align: "right", mono: true, render: (r) => fmt(r.mean_diff) },
    { key: "cohens_d", header: "Cohen's d", align: "right", mono: true, render: (r) => fmt(r.cohens_d) },
    { key: "p_value", header: "p (descriptive)", align: "right", mono: true, render: (r) => fmt(r.p_value) },
  ];

  return (
    <>
      <section className="card fade-up" aria-label="Multi-seed experiment">
        <header className="card-head">
          <div>
            <h3 className="card-title">Multi-seed experiment</h3>
            <p className="card-sub">
              Mean ± sample std across seeds per variant (std is undefined for a single seed). Best value per column highlighted where a direction is defined.
            </p>
          </div>
          <div className="card-actions">
            {d.data_mode === "synthetic_smoke_test" && <StatusChip tone="danger">synthetic smoke test</StatusChip>}
            {experiments.length > 1 && (
              <label className="select-wrap">
                <span className="sr-only">Experiment</span>
                <select value={id} onChange={(e) => setExpId(e.target.value)} aria-label="Experiment">
                  {experiments.map((e) => <option key={e.experiment_id} value={e.experiment_id}>{e.experiment_id}</option>)}
                </select>
              </label>
            )}
          </div>
        </header>
        <DataTable caption="Variant aggregates across seeds" columns={columns} rows={variants} rowKey={(r) => r.variant} />
      </section>

      <ChartCard
        title={`${metricLabel(metric)} by variant`}
        subtitle="Bar = mean across seeds, whisker = ± 1 std; n_seeds shown under each variant"
        actions={<Segmented label="Metric" value={metric} onChange={setMetric} options={tableMetrics.slice(0, 5).map((m) => ({ value: m, label: metricLabel(m) }))} />}
        empty={!chartData.length}
        height={300}
      >
        <div role="img" aria-label={`Bar chart with error bars of ${metricLabel(metric)} per variant`} style={{ width: "100%", height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 12, right: 12, left: 0, bottom: 8 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="name" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} interval={0} tickFormatter={(v) => String(v).replace("SAC_", "").replace("_", " ")} />
              <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={64} tickFormatter={(v) => fmt(v)} />
              <Tooltip cursor={{ fill: "var(--hover)" }} isAnimationActive={false}
                content={<ChartTooltip labelFormatter={(l, p) => `${l} · n=${p?.[0]?.payload?.n_seeds ?? "?"}`} formatter={(v, p) => `${fmt(v)}${Number.isFinite(p?.payload?.std) ? ` ± ${fmt(p.payload.std)}` : ""}`} />} />
              <Bar dataKey="mean" name={metricLabel(metric)} radius={[4, 4, 0, 0]} maxBarSize={56} isAnimationActive={false}>
                {chartData.map((r) => <Cell key={r.name} fill={BASELINES.includes(r.name) ? "var(--text-3)" : "var(--accent)"} />)}
                <ErrorBar dataKey="err" width={6} stroke="var(--text-2)" strokeWidth={1.5} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="legend-row">
          {chartData.map((r) => <span key={r.name} className="mono">{r.name}: n={r.n_seeds}</span>)}
        </div>
      </ChartCard>

      <section className="card fade-up" aria-label="Exploratory pairwise comparison">
        <header className="card-head">
          <div>
            <h3 className="card-title">Pairwise comparison vs {d.reference_variant}</h3>
            <p className="card-sub">{metricLabel(metric)} · Welch t-test and Cohen's d, exploratory</p>
          </div>
          <StatusChip tone="warn" icon={FlaskConical}>exploratory, small n</StatusChip>
        </header>
        <p className="note" role="note">{EXPLORATORY_NOTE}</p>
        <div style={{ height: 12 }} />
        {cmpRows.length ? (
          <DataTable caption="Exploratory pairwise comparison" columns={cmpCols} rows={cmpRows} rowKey={(r) => r.variant + r.metric} />
        ) : (
          <p className="card-sub">No pairwise comparison for this metric (needs at least two seeds per variant).</p>
        )}
      </section>
    </>
  );
}
