import { RefreshCw, FlaskConical, Radio } from "lucide-react";
import RunSelect from "./RunSelect";
import StatusChip from "./StatusChip";
import { VIEWS } from "./nav";

export default function Topbar({ view, runs, runsLoading, selectedRun, onSelect, variants, onRefresh, refreshing, synthetic }) {
  const v = VIEWS.find((x) => x.id === view);
  return (
    <header className="topbar">
      <div className="topbar-main">
        <div className="titles">
          <p className="crumb">
            MarketMaker Lab <span aria-hidden="true">/</span> {selectedRun ? <span className="mono">{selectedRun}</span> : "No run selected"}
          </p>
          <h1 className="page-title">{v.label}</h1>
        </div>
        <div className="topbar-actions">
          <RunSelect runs={runs} value={selectedRun} onChange={onSelect} variants={variants} loading={runsLoading} />
          <button type="button" className="icon-btn" onClick={onRefresh} aria-label="Refresh data" title="Refresh">
            <RefreshCw size={16} className={refreshing ? "spin" : ""} />
          </button>
        </div>
      </div>
      <div className="topbar-chips">
        {!synthetic && (
          <StatusChip tone="info" icon={Radio} title="Data mode">
            Data mode: Mode B — real crypto LOB; FNSPID news validated separately, not causally matched to this LOB.
          </StatusChip>
        )}
        <StatusChip tone="warn" icon={FlaskConical}>
          Research backtest · not financial advice
        </StatusChip>
        {synthetic && (
          <StatusChip tone="danger" className="chip-strong">
            SOFTWARE SMOKE TEST ONLY - NOT A FINANCIAL PERFORMANCE RESULT
          </StatusChip>
        )}
      </div>
    </header>
  );
}
