import { useState } from "react";
import RunSelector from "./components/RunSelector";
import OverviewTab from "./components/OverviewTab";
import PnlTab from "./components/PnlTab";
import RiskTab from "./components/RiskTab";
import BehaviorTab from "./components/BehaviorTab";
import SentimentTab from "./components/SentimentTab";
import ComparisonTab from "./components/ComparisonTab";

const TABS = [
  ["Overview", OverviewTab],
  ["PnL / Equity", PnlTab],
  ["Inventory / Risk", RiskTab],
  ["Market-Making Behavior", BehaviorTab],
  ["Sentiment / Temporal Signals", SentimentTab],
  ["Model Comparison", ComparisonTab],
];

export default function App() {
  const [selectedRun, setSelectedRun] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const ActiveComponent = TABS[activeTab][1];

  return (
    <div style={{ fontFamily: "sans-serif", padding: "1rem" }}>
      <h1>LLM-Enhanced Market Making — Results Dashboard</h1>
      <p style={{ background: "#fef3c7", padding: "0.5rem" }}>
        Data mode: Mode B — real crypto LOB; FNSPID news validated separately, not causally
        matched to this LOB.
      </p>
      <RunSelector selectedRun={selectedRun} onSelect={setSelectedRun} />
      <nav style={{ margin: "1rem 0" }}>
        {TABS.map(([label], i) => (
          <button key={label} onClick={() => setActiveTab(i)} disabled={i === activeTab}>
            {label}
          </button>
        ))}
      </nav>
      <ActiveComponent runId={selectedRun} />
    </div>
  );
}
