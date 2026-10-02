import { MousePointerClick } from "lucide-react";
import EmptyState from "./EmptyState";

export default function NoRun() {
  return (
    <div className="card fade-up">
      <EmptyState icon={MousePointerClick} hint="Select a run from the top bar to see its results." />
    </div>
  );
}
