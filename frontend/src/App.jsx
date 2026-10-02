import { useCallback, useEffect, useState } from "react";

const NO_RUNS = [];
const NO_VARIANTS = {};
import Sidebar from "./components/Sidebar";
import Topbar from "./components/Topbar";
import OverviewView from "./components/OverviewView";
import PnlView from "./components/PnlView";
import RiskView from "./components/RiskView";
import BehaviorView from "./components/BehaviorView";
import SignalsView from "./components/SignalsView";
import CompareView from "./components/CompareView";
import { listRuns, getRunDetail } from "./api";
import { useAsync, useTheme } from "./lib/hooks";
import { looksSynthetic } from "./lib/variant";
import "./App.css";

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [view, setView] = useState("overview");
  const [selectedRun, setSelectedRun] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const { data: runData, loading: runsLoading } = useAsync(async () => {
    const ids = await listRuns().catch(() => []);
    const details = await Promise.allSettled(ids.map((id) => getRunDetail(id)));
    const variants = {};
    ids.forEach((id, i) => {
      if (details[i].status === "fulfilled") variants[id] = details[i].value?.variant;
    });
    return { ids: ids || [], variants };
  }, [refreshKey]);

  const runs = runData?.ids ?? NO_RUNS;
  const variants = runData?.variants ?? NO_VARIANTS;

  useEffect(() => {
    if (!runs.length) return;
    setSelectedRun((cur) => (cur && runs.includes(cur) ? cur : runs[0]));
  }, [runs]);

  const refresh = useCallback(() => setRefreshKey((k) => k + 1), []);
  const synthetic = looksSynthetic(selectedRun, variants[selectedRun]);

  let content;
  const common = { runId: selectedRun, refreshKey };
  switch (view) {
    case "pnl": content = <PnlView {...common} />; break;
    case "risk": content = <RiskView {...common} />; break;
    case "behavior": content = <BehaviorView {...common} />; break;
    case "signals": content = <SignalsView />; break;
    case "compare":
      content = <CompareView runs={runs} runsLoading={runsLoading} selectedRun={selectedRun} onSelect={setSelectedRun} refreshKey={refreshKey} />;
      break;
    default: content = <OverviewView {...common} />;
  }

  return (
    <div className="shell">
      <Sidebar active={view} onChange={setView} theme={theme} onToggleTheme={toggleTheme} />
      <div className="main-col">
        <Topbar
          view={view}
          runs={runs}
          runsLoading={runsLoading}
          selectedRun={selectedRun}
          onSelect={setSelectedRun}
          variants={variants}
          onRefresh={refresh}
          refreshing={runsLoading}
          synthetic={synthetic}
        />
        <main className="content" id="main-panel" role="tabpanel" aria-labelledby={`tab-${view}`} tabIndex={-1}>
          <div className="content-inner" key={view}>{content}</div>
        </main>
      </div>
    </div>
  );
}
