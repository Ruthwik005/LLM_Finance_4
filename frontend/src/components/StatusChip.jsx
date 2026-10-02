export default function StatusChip({ tone = "neutral", icon: Icon, children, title, className = "", mono = false }) {
  return (
    <span className={`chip chip-${tone} ${mono ? "mono" : ""} ${className}`} title={title}>
      {Icon && <Icon size={13} aria-hidden="true" />}
      <span>{children}</span>
    </span>
  );
}
