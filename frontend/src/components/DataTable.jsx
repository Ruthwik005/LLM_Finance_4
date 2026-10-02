export default function DataTable({ columns, rows, rowKey, onRowClick, activeKey, caption, maxHeight }) {
  return (
    <div className="table-wrap" style={maxHeight ? { maxHeight } : undefined} tabIndex={0} role="region" aria-label={caption}>
      <table className="table">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} scope="col" className={c.align === "right" ? "num" : ""}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const k = rowKey(r, i);
            const active = activeKey !== undefined && k === activeKey;
            return (
              <tr
                key={k}
                className={`${onRowClick ? "clickable" : ""} ${active ? "active" : ""}`}
                onClick={onRowClick ? () => onRowClick(r) : undefined}
                onKeyDown={onRowClick ? (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onRowClick(r)) : undefined}
                tabIndex={onRowClick ? 0 : undefined}
                aria-selected={onRowClick ? active : undefined}
              >
                {columns.map((c) => (
                  <td key={c.key} className={`${c.align === "right" ? "num" : ""} ${c.mono ? "mono" : ""}`}>
                    {c.render ? c.render(r, i) : r[c.key]}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
