import { LayoutDashboard, LineChart, ShieldAlert, Activity, Newspaper, BarChart3 } from "lucide-react";

export const VIEWS = [
  { id: "overview", label: "Overview", sub: "Run summary & key metrics", icon: LayoutDashboard },
  { id: "pnl", label: "PnL / Equity", sub: "Cumulative and per-episode performance", icon: LineChart },
  { id: "risk", label: "Inventory & Risk", sub: "Position exposure and drawdown", icon: ShieldAlert },
  { id: "behavior", label: "Market-Making Behavior", sub: "Fills, sides and volume", icon: Activity },
  { id: "signals", label: "Sentiment & Signals", sub: "Offline NLP pipeline status", icon: Newspaper },
  { id: "compare", label: "Model Comparison", sub: "All runs side by side", icon: BarChart3 },
];
