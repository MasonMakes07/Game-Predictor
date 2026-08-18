import sys
from datetime import datetime, timezone

from odds_store import save_odds_snapshot
from vegas_odds import fetch_vegas_spreads


# ── Fetch current spreads and store them as a new timestamped snapshot ────────
def run_odds_update():
    """
    Fetches live NBA spreads from The Odds API and appends them to
    vegas_odds_snapshots as one new snapshot. Returns the number of rows
    inserted (0 in the offseason, when no upcoming games are listed).
    """
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[UPDATE] Starting odds update at {started_at}")

    games = fetch_vegas_spreads()
    if not games:
        # The Odds API lists nothing in the deep offseason. That is a normal
        # no-op, not a failure, so the scheduled run should still pass.
        print("[UPDATE] No upcoming games listed. Nothing to store.")
        return 0

    inserted = save_odds_snapshot(games)
    print(f"[UPDATE] Stored {inserted} games as a new snapshot.")
    return inserted


# ── Scheduled entry point ─────────────────────────────────────────────────────
def main():
    """Runs the odds update, exiting non-zero so a failed run shows as red."""
    try:
        run_odds_update()
    except Exception as error:
        print(f"[UPDATE] FAILED: {type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
