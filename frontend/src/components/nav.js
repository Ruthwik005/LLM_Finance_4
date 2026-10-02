import { LayoutDashboard, LineChart, ShieldAlert, Activity, Newspaper, BarChart3, PlayCircle, Brain } from "lucide-react";

export const VIEWS = [
  { id: "overview", label: "Overview", sub: "Run summary & key metrics", icon: LayoutDashboard },
  { id: "replay", label: "Episode Replay", sub: "Quotes, fills and inventory step by step", icon: PlayCircle },
  { id: "pnl", label: "PnL / Equity", sub: "Cumulative and per-episode performance", icon: LineChart },
  { id: "risk", label: "Inventory & Risk", sub: "Position exposure and drawdown", icon: ShieldAlert },
  { id: "behavior", label: "Market-Making Behavior", sub: "Fills, sides and volume", icon: Activity },
  { id: "training", label: "Training", sub: "Learning curves and best checkpoint", icon: Brain },
  { id: "signals", label: "Sentiment & Signals", sub: "Sentiment and transformer features", icon: Newspaper },
  { id: "compare", label: "Model Comparison", sub: "All runs side by side", icon: BarChart3, PlayCircle, Brain },
];
