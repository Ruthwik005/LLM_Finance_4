import { Activity, Coins, Hash, Layers, Sigma, TrendingUp, Info, Cpu, Calendar, Dices, Tag } from "lucide-react";
import KpiCard from "./KpiCard";
import ChartCard from "./ChartCard";
import EquityChart from "./EquityChart";
import { SkeletonKpis, SkeletonCard } from "./Skeleton";
import StatusChip from "./StatusChip";
import EmptyState from "./EmptyState";
import NoRun from "./NoRun";
import { useRunData } from "../lib/runData";
import { formatCurrency, formatDate, formatNumber, DASH } from "../lib/format";
import { downsample, inventoryPath, meanAbsInventory } from "../lib/derive";


function Meta({ icon: Icon, label, children }) {
  return (
    <div className="meta-row">
      <span className="meta-label">
        <Icon size={14} aria-hidden="true" /> {label}
      </span>
      <span className="meta-val mono">{children}</span>
    </div>
  );
}

export default function OverviewView({ runId, refreshKey, synthetic }) {
  const { data, loading, error } = useRunData(runId, refreshKey, ["detail", "metrics", "episodes", "fills", "equity"]);
  if (!runId) return <NoRun />;
  if (loading && !data) {
    return (
      <>
        <SkeletonKpis />
        <div className="grid-2">
          <SkeletonCard h={200} />
          <SkeletonCard h={200} />
        </div>
      </>
    );
  }
  if (error || !data) {
    return (
      <div className="card">
        <EmptyState hint={error || undefined} />
      </div>
    );
  }
  const { detail, metrics, episodes, fills, equity } = data;
  const m = metrics || {};
  const eps = episodes || [];
  const pnlSpark = downsample(eps.map((e) => e.pnl), 40);
  const eqSpark = downsample((equity || []).map((e) => e.cumulative_pnl), 40);
  const map = Number.isFinite(m.map) ? m.map : meanAbsInventory(inventoryPath(fills));
  const hasDetail = detail && Object.keys(detail).length > 0;

  return (
    <>
      <div className="kpi-grid kpi-grid-6">
        <KpiCard delay={0} label="Mean PnL" icon={TrendingUp} value={m.mean_pnl} format={(v) => formatCurrency(v)} colorBySign spark={eqSpark} />
        <KpiCard delay={40} label="Median PnL" icon={Sigma} value={m.median_pnl} format={(v) => formatCurrency(v)} colorBySign spark={pnlSpark} />
        <KpiCard delay={80} label="PnL Std Dev" icon={Activity} value={m.std_pnl} format={(v) => formatCurrency(v)} />
        <KpiCard delay={120} label="Mean Abs Position" icon={Layers} value={map} format={(v) => formatNumber(v, 3)} hint={Number.isFinite(m.map) ? undefined : "Derived from fills"} />
        <KpiCard delay={160} label="Episodes" icon={Hash} value={episodes ? eps.length : undefined} format={(v) => formatNumber(Math.round(v), 0)} />
        <KpiCard delay={200} label="Total Fills" icon={Coins} value={fills ? fills.length : undefined} format={(v) => formatNumber(Math.round(v), 0)} />
      </div>

      <div className="grid-2">
        <section className="card fade-up" aria-label="Run metadata">
          <header className="card-head">
            <div>
              <h3 className="card-title">Run metadata</h3>
              <p className="card-sub">Configuration of the selected experiment</p>
            </div>
          </header>
          {hasDetail ? (
            <div className="meta">
              <Meta icon={Hash} label="Run ID">{detail.run_id || runId}</Meta>
              <Meta icon={Tag} label="Variant">
                <StatusChip tone="accent">{detail.variant || DASH}</StatusChip>
              </Meta>
              <Meta icon={Dices} label="Seed">{detail.seed ?? DASH}</Meta>
              <Meta icon={Cpu} label="Model path"><span className="path">{detail.model_path || DASH}</span></Meta>
              <Meta icon={Calendar} label="Created">{formatDate(detail.created_at)}</Meta>
            </div>
          ) : (
            <EmptyState compact />
          )}
        </section>

        <ChartCard title="Equity curve" subtitle="Cumulative PnL across episodes" empty={!equity || equity.length === 0} height={200}>
          <EquityChart data={equity || []} height={200} compact />
        </ChartCard>
      </div>

      <section className="card notice fade-up" aria-label="Data mode notice">
        <span className="notice-icon" aria-hidden="true"><Info size={18} /></span>
        <div>
          <h3 className="card-title">Data mode</h3>
          <p className="notice-text">
            {synthetic
              ? "SOFTWARE SMOKE TEST ONLY - NOT A FINANCIAL PERFORMANCE RESULT. This run uses synthetic demo data."
              : "Data mode: Mode B — real crypto LOB; FNSPID news validated separately, not causally matched to this LOB."}
          </p>
          <p className="card-sub">
            Results shown here come from a research backtest. Metrics are illustrative of software behavior and are not a claim of
            tradable financial performance.
          </p>
        </div>
      </section>
    </>
  );
}
