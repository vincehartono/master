import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

# Path to the play-by-play data and a default gameId.
PBP_PATH = r"C:\Users\Vince\master\nba_data\datasets\nbastatsv3_2025.csv"
GAME_ID = "22500001"  # change this to any gameId you want to view


def draw_nba_court(ax=None):
    """
    Draw a single-basket NBA half court, horizontal orientation (left-right).
    All shots are shown going toward the left basket where scoring occurs.
    We rotate the usual shotchart coordinates so x runs baseline-to-baseline.
    """
    if ax is None:
        ax = plt.gca()

    # After rotation we use: x in [-47.5, 422.5], y in [-250, 250]
    ax.set_xlim(-47.5, 422.5)
    ax.set_ylim(-250, 250)

    # Helper to rotate a (x,y) point from original (cx, cy) to horizontal
    def rot(x, y):
        # Original: x in [-250,250], y in [-47.5,422.5]
        # New: x' = y, y' = x
        return y, x

    # Outer lines (approximate)
    x0, y0 = rot(-250, -47.5)
    outer = Rectangle((x0, y0), 470, 500, linewidth=2, color="black", fill=False)
    ax.add_patch(outer)

    # Hoop at left side (scoring basket)
    hb_x, hb_y = rot(0, -47.5 + 5)
    ax.add_patch(Circle((hb_x, hb_y), radius=7.5, linewidth=2, color="orange", fill=False))

    # Backboard (short line behind hoop)
    bb1_x0, bb1_y0 = rot(-30, -7.5)
    ax.add_patch(Rectangle((bb1_x0, bb1_y0), 60, 0, linewidth=2, color="black"))

    # 3-point arc centered at scoring basket
    ax.add_patch(Arc((hb_x, hb_y), 475, 475, theta1=-70, theta2=70, linewidth=2, color="black"))

    # Paint rectangle near basket
    p1_x0, p1_y0 = rot(-80, -47.5)
    ax.add_patch(Rectangle((p1_x0, p1_y0), 190, 160, linewidth=2, color="black", fill=False))

    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def load_game_events(path: str, game_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["gameId"].astype(str) == str(game_id)].copy()
    if "actionNumber" in df.columns:
        df = df.sort_values(["period", "actionNumber"])
    else:
        df = df.sort_values(["period"])
    return df


def main() -> None:
    df = load_game_events(PBP_PATH, GAME_ID)
    if df.empty:
        print(f"No events found for gameId={GAME_ID}")
        return

    teams = df["teamTricode"].dropna().unique().tolist()
    if len(teams) >= 2:
        home, away = teams[0], teams[1]
    else:
        home, away = "HOME", "AWAY"

    fig = plt.figure(figsize=(18, 7))
    # Grid layout: court wide on left, two team boxscores on right
    gs = fig.add_gridspec(2, 3, width_ratios=[5, 1.5, 1.5], height_ratios=[1, 4])

    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")

    ax_court = fig.add_subplot(gs[1, 0])
    draw_nba_court(ax_court)

    ax_box_home = fig.add_subplot(gs[1, 1])
    ax_box_home.axis("off")
    ax_box_away = fig.add_subplot(gs[1, 2])
    ax_box_away.axis("off")

    # Team colors
    team_colors = {home: "red", away: "blue"}

    shot_scatter = ax_court.scatter([], [], s=40, c=[], vmin=0, vmax=1)

    play_text_artist = ax_header.text(0.01, 0.6, "", transform=ax_header.transAxes, va="center", fontsize=10)
    score_text_artist = ax_header.text(0.5, 0.6, "", transform=ax_header.transAxes, va="center", ha="center", fontsize=14, fontweight="bold")
    info_text_artist = ax_header.text(0.01, 0.1, "Space = pause/resume", transform=ax_header.transAxes, va="center", fontsize=9, color="gray")

    x_data = []
    y_data = []
    c_data = []  # color index by team

    home_score = 0
    away_score = 0

    # Play-by-play panel under boxscores (manual axis spanning right side)
    ax_pbp = fig.add_axes([0.58, 0.05, 0.4, 0.25])
    ax_pbp.axis("off")

    # Track stats per player for boxscore
    box = {}  # (team, player) -> dict with FG, REB, AST, STL, BLK, PTS

    paused = {"value": False}
    last_plays = []

    def on_key(event):
        if event.key == " ":
            paused["value"] = not paused["value"]

    fig.canvas.mpl_connect("key_press_event", on_key)

    for _, row in df.iterrows():
        while paused["value"]:
            plt.pause(0.05)

        team = row.get("teamTricode", "")
        x_raw = row.get("xLegacy", 0.0)
        y_raw = row.get("yLegacy", 0.0)
        # Rotate coords for horizontal court: x' = y, y' = x
        x, y = y_raw, x_raw
        shot_result = row.get("shotResult", "")
        is_fg = int(row.get("isFieldGoal", 0)) == 1
        shot_value = int(row.get("shotValue", 0))
        desc = row.get("description", "")
        period = int(row.get("period", 0))
        clock = row.get("clock", "")

        # Update score from the data if available
        if pd.notna(row.get("scoreHome")) and pd.notna(row.get("scoreAway")):
            home_score = int(row["scoreHome"])
            away_score = int(row["scoreAway"])

        # Ensure player entry for any tracked event
        player = row.get("playerName", "Unknown")
        key = (team, player)
        if key not in box:
            box[key] = {"FGM": 0, "FGA": 0, "PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0}

        # Field goals
        if is_fg and team in team_colors:
            made = shot_result == "Made"
            x_data.append(x)
            y_data.append(y)
            c_data.append(0 if team == home else 1)
            shot_scatter.set_offsets(list(zip(x_data, y_data)))
            shot_scatter.set_color([team_colors[home] if v == 0 else team_colors[away] for v in c_data])

            # Update FG stats
            entry = box[key]
            entry["FGA"] += 1
            if made:
                entry["FGM"] += 1
                entry["PTS"] += shot_value

        # Rebounds, assists, steals, blocks via actionType/description keywords
        action_type = str(row.get("actionType", "")).lower()
        desc_lower = str(desc).lower()
        entry = box[key]

        if "rebound" in action_type or "rebound" in desc_lower:
            entry["REB"] += 1
        if "assist" in action_type or "assist" in desc_lower or "ast" in desc_lower:
            entry["AST"] += 1
        if "steal" in action_type or "steal" in desc_lower:
            entry["STL"] += 1
        if "block" in action_type or "block" in desc_lower:
            entry["BLK"] += 1

        play_text = f"Q{period} {clock} | {team}: {desc}"
        play_text_artist.set_text(play_text)
        score_text_artist.set_text(f"{away}: {away_score}  {home}: {home_score}")

        # Track last N plays for textual PBP
        if desc:
            last_plays.append(play_text)
            last_plays = last_plays[-20:]

        # Update boxscore tables (home and away separately)
        if box:
            def build_rows(team_code):
                rows = []
                for (tm, player_name), stats in box.items():
                    if tm != team_code:
                        continue
                    rows.append(
                        f"{player_name[:16]:16s} "
                        f"{stats['FGM']:2d}/{stats['FGA']:<2d}  "
                        f"{stats['REB']:2d}  {stats['AST']:2d}  "
                        f"{stats['STL']:2d}  {stats['BLK']:2d}   "
                        f"{stats['PTS']:3d}"
                    )
                return rows

            ax_box_home.clear()
            ax_box_home.axis("off")
            ax_box_home.text(
                0.01,
                0.98,
                f"{home} Boxscore",
                transform=ax_box_home.transAxes,
                va="top",
                fontsize=9,
                fontweight="bold",
                color=team_colors.get(home, "black"),
            )
            ax_box_home.text(
                0.01,
                0.90,
                "Player            FG   R   A   S   B   PTS",
                transform=ax_box_home.transAxes,
                va="top",
                fontsize=8,
                family="monospace",
            )
            ax_box_home.text(
                0.01,
                0.83,
                "\n".join(build_rows(home)),
                transform=ax_box_home.transAxes,
                va="top",
                fontsize=8,
                family="monospace",
            )

            ax_box_away.clear()
            ax_box_away.axis("off")
            ax_box_away.text(
                0.01,
                0.98,
                f"{away} Boxscore",
                transform=ax_box_away.transAxes,
                va="top",
                fontsize=9,
                fontweight="bold",
                color=team_colors.get(away, "black"),
            )
            ax_box_away.text(
                0.01,
                0.90,
                "Player            FG   R   A   S   B   PTS",
                transform=ax_box_away.transAxes,
                va="top",
                fontsize=8,
                family="monospace",
            )
            ax_box_away.text(
                0.01,
                0.83,
                "\n".join(build_rows(away)),
                transform=ax_box_away.transAxes,
                va="top",
                fontsize=8,
                family="monospace",
            )

            # Update play-by-play panel with last 20 plays
            ax_pbp.clear()
            ax_pbp.axis("off")
            ax_pbp.text(
                0.01,
                0.98,
                "Last 20 plays",
                transform=ax_pbp.transAxes,
                va="top",
                fontsize=9,
                fontweight="bold",
            )
            ax_pbp.text(
                0.01,
                0.92,
                "\n".join(reversed(last_plays)),
                transform=ax_pbp.transAxes,
                va="top",
                fontsize=7,
                family="monospace",
            )

        plt.pause(0.1)

    plt.show()


if __name__ == "__main__":
    main()
