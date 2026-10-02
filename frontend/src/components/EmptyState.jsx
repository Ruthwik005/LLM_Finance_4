import { Inbox } from "lucide-react";

export default function EmptyState({ hint, icon: Icon = Inbox, compact = false }) {
  return (
    <div className={`empty ${compact ? "empty-compact" : ""}`} role="status">
      <span className="empty-icon" aria-hidden="true">
        <Icon size={compact ? 18 : 24} />
      </span>
      <p className="empty-title">Not run / insufficient data</p>
      {hint && <p className="empty-hint">{hint}</p>}
    </div>
  );
}
