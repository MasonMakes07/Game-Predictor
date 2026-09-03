// Left navigation rail: brand, view switcher, and the model accuracy card.
export default function Sidebar({ view, onSelectView, metrics }) {
  const items = [
    { id: "predictions", icon: "🏀", label: "Predictions" },
    { id: "rankings", icon: "📊", label: "Rankings" },
    { id: "custom", icon: "🎯", label: "Custom Matchup" },
  ];

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">P</span>
        <span className="brand-name">Predictor</span>
      </div>

      <nav className="nav">
        <p className="nav-heading">Menu</p>
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-item${view === item.id ? " nav-item--active" : ""}`}
            onClick={() => onSelectView(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      {metrics?.accuracy != null && (
        <div className="metric-card">
          <p className="metric-label">Model accuracy</p>
          <p className="metric-value">{(metrics.accuracy * 100).toFixed(1)}%</p>
          <p className="metric-sub">
            {metrics.eval_split ?? "held-out season"} ·{" "}
            {metrics.val_samples?.toLocaleString()} games
          </p>
        </div>
      )}
    </aside>
  );
}
