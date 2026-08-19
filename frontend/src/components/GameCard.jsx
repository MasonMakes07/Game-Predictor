// Deterministic badge hue so each team keeps the same colour between renders.
function teamHue(abbr) {
  let hash = 0;
  for (let i = 0; i < abbr.length; i += 1) {
    hash = (hash * 31 + abbr.charCodeAt(i)) % 360;
  }
  return hash;
}

// Splits an ISO tip-off time into a short local date and time label.
function formatTipOff(iso) {
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return { date: "TBD", time: "" };
  return {
    date: when.toLocaleDateString(undefined, { month: "short", day: "numeric" }),
    time: when.toLocaleTimeString(undefined, {
      hour: "numeric",
      minute: "2-digit",
    }),
  };
}

// One upcoming game: tip-off, both teams, the probability split, and verdict.
export default function GameCard({ game }) {
  const { date, time } = formatTipOff(game.commenceTime);

  const homeFavored = game.homeProb >= game.awayProb;
  const favoredName = homeFavored ? game.homeTeam : game.awayTeam;
  const favoredPct = Math.round(
    (homeFavored ? game.homeProb : game.awayProb) * 100
  );

  // Only the away width is rounded; the home segment takes the remainder via
  // flex, so the two can never sum past 100% (Lessons.md #4).
  const awayPct = Math.round(game.awayProb * 100);

  return (
    <article className="game-card">
      <div className="game-when">
        <span className="game-date">{date}</span>
        <span className="game-time">{time}</span>
      </div>

      <span
        className="team-badge"
        style={{ "--badge-hue": teamHue(game.awayAbbr) }}
        title={game.awayTeam}
      >
        {game.awayAbbr}
      </span>

      <div className="game-prob">
        <div className="game-prob-head">
          <span className="prob-figure">{awayPct}%</span>
          <span className="prob-caption">win probability</span>
          <span className="prob-figure">{100 - awayPct}%</span>
        </div>
        <div className="prob-bar">
          <span className="prob-seg prob-seg--away" style={{ width: `${awayPct}%` }} />
          <span className="prob-seg prob-seg--home" />
        </div>
      </div>

      <span
        className="team-badge"
        style={{ "--badge-hue": teamHue(game.homeAbbr) }}
        title={game.homeTeam}
      >
        {game.homeAbbr}
      </span>

      <div className="game-verdict">
        <span className="verdict-pill">
          {favoredName.split(" ").slice(-1)[0]} favored · {favoredPct}%
        </span>
        {game.vegasSpread != null && (
          <span className="vegas-line">
            Vegas {game.vegasSpread > 0 ? "+" : ""}
            {game.vegasSpread.toFixed(1)}
          </span>
        )}
      </div>
    </article>
  );
}
