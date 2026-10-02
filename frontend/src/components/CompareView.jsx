import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import ChartCard from "./ChartCard";
import ChartTooltip from "./ChartTooltip";
import DataTable from "./DataTable";
import EmptyState from "./EmptyState";
import { SkeletonCard } from "./Skeleton";
import StatusChip from "./StatusChip";
import { AXIS_LINE, AXIS_TICK, GRID } from "./chartTheme";
import { useAsync } from "../lib/hooks";
import { getRunMetrics } from "../api";
import { bestIndex, pnlPerMap } from "../lib/derive";
import { DASH, formatCompact, formatCurrency, formatNumber } from "../lib/format";

export default function CompareView({ runs, runsLoading, selectedRun, onSelect, refreshKey }) {
  const key = runs.join("|");
  const { data, loading } = useAsync(async () => {
    const res = await Promise.allSettled(runs.map((r) => getRunMetrics(r)));
    return runs.map((run, i) => {
      const m = res[i].status === "fulfilled" ? res[i].value || {} : null;
      return {
        run,
        failed: m === null,
        mean_pnl: m?.mean_pnl,
        median_pnl: m?.median_pnl,
        std_pnl: m?.std_pnl,
        map: m?.map,
        ratio: m ? pnlPerMap(m.mean_pnl, m.map) : null,
      };
    });
  }, [key, refreshKey]);

  const rows = useMemo(() => data || [], [data]);
  const best = useMemo(
    () => ({
      mean_pnl: bestIndex(rows.map((r) => r.mean_pnl)),
      median_pnl: bestIndex(rows.map((r) => r.median_pnl)),
      std_pnl: bestIndex(rows.map((r) => r.std_pnl), false),
      map: bestIndex(rows.map((r) => r.map), false),
      ratio: bestIndex(rows.map((r) => r.ratio)),
    }),
    [rows],
  );

  if (runsLoading || (loading && !data)) return <SkeletonCard h={300} />;
  if (runs.length < 1) return <div className="card"><EmptyState hint="No runs found yet." /></div>;

  const cell = (field, fmt) => (r, i) => {
    const v = r[field];
    const txt = Number.isFinite(v) ? fmt(v) : DASH;
    return best[field] === i ? <span className="best">{txt}</span> : txt;
  };
  const columns = [
    {
      key: "run",
      header: "Run",
      render: (r) => (
        <span className="run-cell">
          <span className="mono">{r.run}</span>
          {r.run === selectedRun && <StatusChip tone="accent">selected</StatusChip>}
          {r.failed && <StatusChip tone="danger">failed</StatusChip>}
        </span>
      ),
    },
    { key: "mean_pnl", header: "Mean PnL", align: "right", mono: true, render: cell("mean_pnl", (v) => formatCurrency(v)) },
    { key: "median_pnl", header: "Median PnL", align: "right", mono: true, render: cell("median_pnl", (v) => formatCurrency(v)) },
    { key: "std_pnl", header: "Std", align: "right", mono: true, render: cell("std_pnl", (v) => formatCurrency(v)) },
    { key: "map", header: "MAP", align: "right", mono: true, render: cell("map", (v) => formatNumber(v, 3)) },
    { key: "ratio", header: "PnL / MAP", align: "right", mono: true, render: cell("ratio", (v) => formatNumber(v, 2)) },
  ];

  return (
    <>
      <section className="card fade-up" aria-label="Run comparison">
        <header className="card-head">
          <div>
            <h3 className="card-title">Run comparison</h3>
            <p className="card-sub">Best value per column is highlighted (lowest for Std and MAP). Click a row to select it.</p>
          </div>
        </header>
        <DataTable caption="Model comparison table" columns={columns} rows={rows} rowKey={(r) => r.run} onRowClick={(r) => onSelect(r.run)} activeKey={selectedRun} />
      </section>
      <ChartCard title="Mean PnL across runs" subtitle="Mean vs median per episode" empty={rows.every((r) => !Number.isFinite(r.mean_pnl))} height={300}>
        <div role="img" aria-label="Grouped bar chart of mean and median PnL across runs" style={{ width: "100%", height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="run" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} interval="preserveStartEnd" />
              <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={64} tickFormatter={(v) => formatCompact(v)} />
              <Tooltip cursor={{ fill: "var(--hover)" }} content={<ChartTooltip labelFormatter={(l) => l} formatter={(v) => formatCurrency(v)} />} />
              <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-2)" }} />
              <Bar dataKey="mean_pnl" name="Mean PnL" fill="var(--accent)" radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false} />
              <Bar dataKey="median_pnl" name="Median PnL" fill="var(--accent-2)" radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>
    </>
  );
}
