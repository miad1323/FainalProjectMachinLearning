from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

STATSBOMB_RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DEFAULT_ODDS_URL = "https://www.football-data.co.uk/mmz4281/1516/E0.csv"


def _get_name(value: Any, key: str | None = None, default: str | None = None) -> str | None:
    if isinstance(value, dict):
        if key is not None and key in value:
            return value.get(key, default)
        return value.get("name", default)
    return default if value is None else str(value)


def _get_id(value: Any, key: str | None = None, default: int | None = None) -> int | None:
    if isinstance(value, dict):
        if key is not None and key in value:
            return value.get(key, default)
        return value.get("id", default)
    return default


def download_json(url: str, cache_path: str | Path, timeout: int = 60) -> Any:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return response.json()


def download_optional_json(
        url: str,
        cache_path: str | Path,
        timeout: int = 60,
) -> Any | None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return response.json()


def load_competitions(cache_dir: str | Path = "cache") -> pd.DataFrame:
    data = download_json(
        f"{STATSBOMB_RAW_BASE}/competitions.json",
        Path(cache_dir) / "competitions.json",
    )
    return pd.json_normalize(data)


def select_competition(
        competitions: pd.DataFrame,
        competition_name: str = "Premier League",
        season_name: str = "2015/2016",
) -> tuple[int, int, pd.Series]:
    mask = (
            competitions["competition_name"].str.casefold().eq(
                competition_name.casefold()
            )
            & competitions["season_name"].str.casefold().eq(season_name.casefold())
    )
    selected = competitions.loc[mask]
    if selected.empty:
        available = competitions.loc[
            competitions["competition_name"].str.contains(
                competition_name, case=False, na=False
            ),
            ["competition_name", "season_name", "competition_id", "season_id"],
        ]
        raise ValueError(
            f"Competition/season not found: {competition_name} {season_name}. "
            f"Closest entries:\n{available.to_string(index=False)}"
        )
    row = selected.iloc[0]
    return int(row["competition_id"]), int(row["season_id"]), row


def load_matches(
        competition_id: int,
        season_id: int,
        cache_dir: str | Path = "cache",
) -> pd.DataFrame:
    data = download_json(
        f"{STATSBOMB_RAW_BASE}/matches/{competition_id}/{season_id}.json",
        Path(cache_dir) / "matches" / f"{competition_id}_{season_id}.json",
    )
    rows = []
    for m in data:
        kick_off_raw = f"{m.get('match_date')} {m.get('kick_off', '00:00:00')}"
        rows.append(
            {
                "match_id": int(m["match_id"]),
                "competition_id": competition_id,
                "season_id": season_id,
                "match_date": pd.to_datetime(m.get("match_date"), errors="coerce"),
                "kick_off": pd.to_datetime(kick_off_raw, errors="coerce"),
                "home_team_id": _get_id(m.get("home_team"), key="home_team_id"),
                "home_team": _get_name(m.get("home_team"), key="home_team_name"),
                "away_team_id": _get_id(m.get("away_team"), key="away_team_id"),
                "away_team": _get_name(m.get("away_team"), key="away_team_name"),
                "home_score": int(m.get("home_score", 0)),
                "away_score": int(m.get("away_score", 0)),
                "competition_stage": _get_name(m.get("competition_stage")),
                "stadium": _get_name(m.get("stadium")),
                "referee": _get_name(m.get("referee")),
            }
        )
    matches = (
        pd.DataFrame(rows)
        .sort_values(["kick_off", "match_id"])
        .reset_index(drop=True)
    )
    matches["outcome"] = np.select(
        [
            matches.home_score > matches.away_score,
            matches.home_score == matches.away_score
        ],
        ["H", "D"],
        default="A",
    )
    matches["goal_margin"] = (matches.home_score - matches.away_score).clip(-5, 5)
    return matches


def load_events(match_id: int, cache_dir: str | Path = "cache") -> list[dict]:
    return download_json(
        f"{STATSBOMB_RAW_BASE}/events/{match_id}.json",
        Path(cache_dir) / "events" / f"{match_id}.json",
    )


def load_lineups(match_id: int, cache_dir: str | Path = "cache") -> list[dict]:
    return download_json(
        f"{STATSBOMB_RAW_BASE}/lineups/{match_id}.json",
        Path(cache_dir) / "lineups" / f"{match_id}.json",
    )


def load_three_sixty(match_id: int, cache_dir: str | Path = "cache") -> list[dict] | None:
    return download_optional_json(
        f"{STATSBOMB_RAW_BASE}/three-sixty/{match_id}.json",
        Path(cache_dir) / "three-sixty" / f"{match_id}.json",
    )


def flatten_three_sixty(frames: Sequence[dict] | None, match_id: int) -> pd.DataFrame:
    columns = [
        "match_id",
        "event_id",
        "frame_player_index",
        "x",
        "y",
        "teammate",
        "actor",
        "keeper",
        "visible_area",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    rows = []
    for frame in frames:
        event_id = frame.get("event_uuid")
        visible_area = json.dumps(frame.get("visible_area", []), ensure_ascii=False)
        for player_index, player in enumerate(frame.get("freeze_frame", []) or []):
            location = player.get("location") or [np.nan, np.nan]
            rows.append(
                {
                    "match_id": int(match_id),
                    "event_id": event_id,
                    "frame_player_index": int(player_index),
                    "x": location[0] if len(location) > 0 else np.nan,
                    "y": location[1] if len(location) > 1 else np.nan,
                    "teammate": bool(player.get("teammate", False)),
                    "actor": bool(player.get("actor", False)),
                    "keeper": bool(player.get("keeper", False)),
                    "visible_area": visible_area,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def flatten_events(events: Sequence[dict], match_id: int) -> pd.DataFrame:
    rows = []
    for e in events:
        pass_info = e.get("pass") or {}
        shot_info = e.get("shot") or {}
        foul_info = e.get("foul_committed") or {}
        behaviour_info = e.get("bad_behaviour") or {}
        location = e.get("location") or [np.nan, np.nan]
        card = _get_name(foul_info.get("card")) or _get_name(behaviour_info.get("card"))
        rows.append(
            {
                "match_id": int(match_id),
                "event_id": e.get("id"),
                "index": int(e.get("index", 0)),
                "period": int(e.get("period", 0)),
                "minute": int(e.get("minute", 0)),
                "second": int(e.get("second", 0)),
                "event_time_seconds": int(e.get("minute", 0)) * 60 + int(e.get("second", 0)),
                "timestamp": e.get("timestamp"),
                "team_id": _get_id(e.get("team")),
                "team": _get_name(e.get("team")),
                "player_id": _get_id(e.get("player")),
                "player": _get_name(e.get("player")),
                "event_type": _get_name(e.get("type")),
                "play_pattern": _get_name(e.get("play_pattern")),
                "x": location[0] if len(location) > 0 else np.nan,
                "y": location[1] if len(location) > 1 else np.nan,
                "pass_outcome": _get_name(pass_info.get("outcome")),
                "pass_type": _get_name(pass_info.get("type")),
                "shot_outcome": _get_name(shot_info.get("outcome")),
                "shot_xg": float(shot_info.get("statsbomb_xg", 0.0) or 0.0),
                "card": card,
                "possession": int(e.get("possession", 0) or 0),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["period", "minute", "second", "index"])
        .reset_index(drop=True)
    )


def flatten_lineups(lineups: Sequence[dict], match_id: int) -> pd.DataFrame:
    rows = []
    for team_block in lineups:
        team_id = team_block.get("team_id")
        team_name = team_block.get("team_name")
        for p in team_block.get("lineup", []):
            rows.append(
                {
                    "match_id": int(match_id),
                    "team_id": team_id,
                    "team": team_name,
                    "player_id": p.get("player_id"),
                    "player": p.get("player_name"),
                    "player_nickname": p.get("player_nickname"),
                    "jersey_number": p.get("jersey_number"),
                    "country": _get_name(p.get("country")),
                    "positions": json.dumps(p.get("positions", []), ensure_ascii=False),
                    "cards": json.dumps(p.get("cards", []), ensure_ascii=False),
                }
            )
    return pd.DataFrame(rows)


def ingest_season(
        competition_name: str = "Premier League",
        season_name: str = "2015/2016",
        *,
        cache_dir: str | Path = "cache",
        max_matches: int | None = None,
        include_lineups: bool = True,
        include_three_sixty: bool = True,
        verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    competitions = load_competitions(cache_dir)
    competition_id, season_id, selected = select_competition(
        competitions, competition_name, season_name
    )
    matches = load_matches(competition_id, season_id, cache_dir)
    if max_matches is not None:
        matches = matches.head(int(max_matches)).copy()

    event_frames: list[pd.DataFrame] = []
    lineup_frames: list[pd.DataFrame] = []
    three_sixty_frames: list[pd.DataFrame] = []
    event_failures = []
    lineup_failures = []
    three_sixty_missing = []
    three_sixty_failures = []

    total = len(matches)
    for pos, match_id in enumerate(matches.match_id, start=1):
        match_id = int(match_id)
        try:
            event_frames.append(flatten_events(load_events(match_id, cache_dir), match_id))
        except Exception as exc:
            event_failures.append({"match_id": match_id, "error": repr(exc)})
            continue

        if include_lineups:
            try:
                lineup_frames.append(
                    flatten_lineups(load_lineups(match_id, cache_dir), match_id)
                )
            except Exception as exc:
                lineup_failures.append({"match_id": match_id, "error": repr(exc)})

        if include_three_sixty:
            try:
                raw_360 = load_three_sixty(match_id, cache_dir)
                if raw_360 is None:
                    three_sixty_missing.append(match_id)
                else:
                    three_sixty_frames.append(flatten_three_sixty(raw_360, match_id))
            except Exception as exc:
                three_sixty_failures.append({"match_id": match_id, "error": repr(exc)})

        if verbose and (pos == 1 or pos % 20 == 0 or pos == total):
            print(f"Downloaded/loaded {pos}/{total} matches")

    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    lineups = pd.concat(lineup_frames, ignore_index=True) if lineup_frames else pd.DataFrame()
    if three_sixty_frames:
        three_sixty = pd.concat(three_sixty_frames, ignore_index=True)
    else:
        three_sixty = flatten_three_sixty(None, 0)

    loaded_match_ids = set(events.match_id.unique()) if not events.empty else set()
    matches = matches.loc[matches.match_id.isin(loaded_match_ids)].copy()

    metadata = {
        "competition_id": competition_id,
        "season_id": season_id,
        "competition_name": selected["competition_name"],
        "season_name": selected["season_name"],
        "n_matches_requested": total,
        "n_matches_loaded": int(events.match_id.nunique()) if not events.empty else 0,
        "n_lineup_matches_loaded": int(lineups.match_id.nunique()) if not lineups.empty else 0,
        "n_three_sixty_matches_loaded": (
            int(three_sixty.match_id.nunique()) if not three_sixty.empty else 0
        ),
        "event_failures": event_failures,
        "lineup_failures": lineup_failures,
        "three_sixty_missing_count": len(three_sixty_missing),
        "three_sixty_missing_match_ids": three_sixty_missing,
        "three_sixty_failures": three_sixty_failures,
    }
    return matches, events, lineups, three_sixty, metadata


DEFENSIVE_TYPES = {
    "Interception",
    "Duel",
    "Block",
    "Clearance",
    "Ball Recovery",
    "Pressure",
}
RED_CARDS = {"Red Card", "Second Yellow"}


def _summarize_one_team(events: pd.DataFrame, team: str) -> dict:
    own = events.loc[events.team.eq(team)]
    shots = own.event_type.eq("Shot")
    passes = own.event_type.eq("Pass")
    set_piece = own.play_pattern.fillna("").str.startswith("From ")
    return {
        "shots": int(shots.sum()),
        "shot_goals": int((shots & own.shot_outcome.eq("Goal")).sum()),
        "xg": float(own.loc[shots, "shot_xg"].sum()),
        "passes": int(passes.sum()),
        "completed_passes": int((passes & own.pass_outcome.isna()).sum()),
        "pressures": int(own.event_type.eq("Pressure").sum()),
        "carries_final_third": int((own.event_type.eq("Carry") & own.x.ge(80)).sum()),
        "set_pieces": int(set_piece.sum()),
        "defensive_actions": int(own.event_type.isin(DEFENSIVE_TYPES).sum()),
        "yellow_cards": int(own.card.eq("Yellow Card").sum()),
        "red_cards": int(own.card.isin(RED_CARDS).sum()),
        "event_share": float(len(own) / max(1, len(events))),
    }


def build_match_team_table(matches: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match in matches.itertuples(index=False):
        match_events = events.loc[events.match_id.eq(match.match_id)]
        if match_events.empty:
            continue
        home = _summarize_one_team(match_events, match.home_team)
        away = _summarize_one_team(match_events, match.away_team)
        home_row = {
            "match_id": match.match_id,
            "kick_off": match.kick_off,
            "team": match.home_team,
            "opponent": match.away_team,
            "venue": "home",
            "goals_for": match.home_score,
            "goals_against": match.away_score,
            "points": (3 if match.home_score > match.away_score
                       else 1 if match.home_score == match.away_score else 0),
        }
        away_row = {
            "match_id": match.match_id,
            "kick_off": match.kick_off,
            "team": match.away_team,
            "opponent": match.home_team,
            "venue": "away",
            "goals_for": match.away_score,
            "goals_against": match.home_score,
            "points": (3 if match.away_score > match.home_score
                       else 1 if match.home_score == match.away_score else 0),
        }
        for key, value in home.items():
            home_row[f"{key}_for"] = value
            home_row[f"{key}_against"] = away[key]
        for key, value in away.items():
            away_row[f"{key}_for"] = value
            away_row[f"{key}_against"] = home[key]
        rows.extend([home_row, away_row])
    return (
        pd.DataFrame(rows)
        .sort_values(["team", "kick_off", "match_id"])
        .reset_index(drop=True)
    )


def build_pre_match_features(
        matches: pd.DataFrame,
        team_matches: pd.DataFrame,
        *,
        rolling_window: int = 5,
        min_history: int = 3,
) -> pd.DataFrame:
    team_matches = team_matches.copy().sort_values(["team", "kick_off", "match_id"])
    base_cols = [
        c
        for c in team_matches.columns
        if c
           not in {
               "match_id",
               "kick_off",
               "team",
               "opponent",
               "venue",
           }
           and pd.api.types.is_numeric_dtype(team_matches[c])
    ]

    feature_frames = []
    for team, group in team_matches.groupby("team", sort=False):
        group = group.sort_values(["kick_off", "match_id"]).copy()
        group["matches_played_before"] = np.arange(len(group))
        group["history_max_kickoff"] = group["kick_off"].shift(1)
        group["rest_days"] = group["kick_off"].diff().dt.total_seconds().div(86400)

        for col in ["points", "goals_for", "goals_against", "shots_for", "xg_for"]:
            group[f"form5_{col}"] = (
                group[col].shift(1).rolling(5, min_periods=1).mean()
            )

        for col in ["points", "goals_for", "goals_against"]:
            group[f"form3_{col}"] = (
                group[col].shift(1).rolling(3, min_periods=1).mean()
            )

        for col in ["points", "goals_for", "xg_for"]:
            last5 = group[col].shift(1).rolling(5, min_periods=1).mean()
            prev5 = group[col].shift(6).rolling(5, min_periods=1).mean()
            group[f"momentum_{col}"] = last5 - prev5

        group["shot_conv_5"] = (
                group["shot_goals_for"].shift(1).rolling(5, min_periods=1).mean()
                / (group["shots_for"].shift(1).rolling(5, min_periods=1).mean() + 1e-6)
        )

        pass_for = group["passes_for"].shift(1).rolling(5, min_periods=1).mean()
        pass_against = group["passes_against"].shift(1).rolling(5, min_periods=1).mean()
        group["possession_share_5"] = pass_for / (pass_for + pass_against + 1e-6)

        group["pressure_rate_5"] = (
                group["pressures_for"].shift(1).rolling(5, min_periods=1).mean() / 90.0
        )

        sp_for = group["set_pieces_for"].shift(1).rolling(5, min_periods=1).mean()
        sp_against = group["set_pieces_against"].shift(1).rolling(5, min_periods=1).mean()
        group["set_piece_share_5"] = sp_for / (sp_for + sp_against + 1e-6)

        home_mask = group.venue == "home"
        away_mask = group.venue == "away"
        group["home_win_rate_5"] = np.nan
        group["away_win_rate_5"] = np.nan
        if home_mask.any():
            home_group = group[home_mask].copy()
            home_group["home_win_rate_5"] = (
                home_group["points"]
                .shift(1)
                .rolling(5, min_periods=1)
                .apply(lambda x: (x == 3).mean(), raw=False)
            )
            group.loc[home_mask, "home_win_rate_5"] = home_group["home_win_rate_5"]
        if away_mask.any():
            away_group = group[away_mask].copy()
            away_group["away_win_rate_5"] = (
                away_group["points"]
                .shift(1)
                .rolling(5, min_periods=1)
                .apply(lambda x: (x == 3).mean(), raw=False)
            )
            group.loc[away_mask, "away_win_rate_5"] = away_group["away_win_rate_5"]
        group["home_win_rate_5"] = group["home_win_rate_5"].ffill()
        group["away_win_rate_5"] = group["away_win_rate_5"].ffill()

        for col in base_cols:
            group[f"form_{rolling_window}_{col}"] = (
                group[col].shift(1).rolling(
                    rolling_window,
                    min_periods=min(min_history, rolling_window),
                ).mean()
            )
        feature_frames.append(group)

    team_features = pd.concat(feature_frames, ignore_index=True)

    feature_cols = (
            [c for c in team_features if c.startswith(f"form_{rolling_window}_")]
            + [
                "matches_played_before",
                "history_max_kickoff",
                "rest_days",
                "form5_points",
                "form5_goals_for",
                "form5_goals_against",
                "form5_shots_for",
                "form5_xg_for",
                "form3_points",
                "form3_goals_for",
                "form3_goals_against",
                "momentum_points",
                "momentum_goals_for",
                "momentum_xg_for",
                "shot_conv_5",
                "possession_share_5",
                "pressure_rate_5",
                "set_piece_share_5",
                "home_win_rate_5",
                "away_win_rate_5",
            ]
    )
    feature_cols = list(dict.fromkeys(feature_cols))

    home = (
        team_features.loc[team_features.venue.eq("home"), ["match_id", "team", *feature_cols]]
        .copy()
    )
    away = (
        team_features.loc[team_features.venue.eq("away"), ["match_id", "team", *feature_cols]]
        .copy()
    )
    home = home.rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in feature_cols}}
    )
    away = away.rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in feature_cols}}
    )

    result = matches.merge(home, on=["match_id", "home_team"], how="inner").merge(
        away, on=["match_id", "away_team"], how="inner"
    )
    result = result.loc[
        result.home_matches_played_before.ge(min_history)
        & result.away_matches_played_before.ge(min_history)
    ].copy()

    paired = [c.removeprefix("home_") for c in result.columns if c.startswith("home_form_")]
    for suffix in paired:
        away_col = f"away_{suffix}"
        if away_col in result:
            result[f"diff_{suffix}"] = result[f"home_{suffix}"] - result[away_col]

    assert_pre_match_leakage_free(result)
    return result.sort_values(["kick_off", "match_id"]).reset_index(drop=True)


def assert_pre_match_leakage_free(pre_match: pd.DataFrame) -> None:
    for side in ("home", "away"):
        hist = pd.to_datetime(pre_match[f"{side}_history_max_kickoff"], errors="coerce")
        target = pd.to_datetime(pre_match["kick_off"], errors="coerce")
        bad = hist.notna() & (hist >= target)
        if bad.any():
            raise AssertionError(
                f"Pre-match leakage detected for {side}: {int(bad.sum())} rows use "
                f"target/future history."
            )


def _snapshot_team_features(
        events_cut: pd.DataFrame,
        team: str,
        snapshot_seconds: int,
        window_seconds: int,
) -> dict:
    own = events_cut.loc[events_cut.team.eq(team)]
    recent_start = max(0, snapshot_seconds - window_seconds)
    recent = own.loc[own.event_time_seconds.ge(recent_start)]
    shots = own.event_type.eq("Shot")
    passes = own.event_type.eq("Pass")
    set_piece = own.play_pattern.fillna("").str.startswith("From ")
    window_minutes = max(window_seconds / 60.0, 1e-9)

    recent_shots = recent.event_type.eq("Shot").sum()
    recent_pressures = recent.event_type.eq("Pressure").sum()
    recent_passes = recent.event_type.eq("Pass").sum()

    return {
        "shots": int(shots.sum()),
        "xg": float(own.loc[shots, "shot_xg"].sum()),
        "passes": int(passes.sum()),
        "completed_passes": int((passes & own.pass_outcome.isna()).sum()),
        "pressures": int(own.event_type.eq("Pressure").sum()),
        "defensive_actions": int(own.event_type.isin(DEFENSIVE_TYPES).sum()),
        "final_third_carries": int((own.event_type.eq("Carry") & own.x.ge(80)).sum()),
        "set_pieces": int(set_piece.sum()),
        "recent_shots": recent_shots,
        "recent_pressures": recent_pressures,
        "recent_events": int(len(recent)),
        "recent_event_rate": float(len(recent) / window_minutes),
        "red_cards": int(own.card.isin(RED_CARDS).sum()),
        "event_share": float(len(own) / max(1, len(events_cut))),
        "recent_shots_per_min": recent_shots / window_minutes,
        "recent_pressures_per_min": recent_pressures / window_minutes,
        "recent_passes_per_min": recent_passes / window_minutes,
    }


def _score_at_snapshot(events_cut: pd.DataFrame, home_team: str, away_team: str) -> tuple[int, int]:
    shot_goals = events_cut.event_type.eq("Shot") & events_cut.shot_outcome.eq("Goal")
    home_goals = int((shot_goals & events_cut.team.eq(home_team)).sum())
    away_goals = int((shot_goals & events_cut.team.eq(away_team)).sum())
    own_goals = events_cut.event_type.eq("Own Goal Against")
    home_goals += int((own_goals & events_cut.team.eq(away_team)).sum())
    away_goals += int((own_goals & events_cut.team.eq(home_team)).sum())
    return home_goals, away_goals


def build_in_play_snapshots(
        matches: pd.DataFrame,
        events: pd.DataFrame,
        pre_match: pd.DataFrame,
        *,
        snapshot_minutes: Sequence[int] = tuple(range(5, 91, 5)),
        recent_window: int = 5,
) -> pd.DataFrame:
    pre_map = pre_match.set_index("match_id")
    feature_exclude = {
        "match_id",
        "competition_id",
        "season_id",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "outcome",
        "goal_margin",
        "history_max_kickoff",
    }
    pre_cols = [
        c
        for c in pre_match.columns
        if c not in feature_exclude
           and not c.endswith("history_max_kickoff")
           and pd.api.types.is_numeric_dtype(pre_match[c])
    ]
    rows = []
    window_seconds = int(recent_window * 60)
    for match in matches.itertuples(index=False):
        if match.match_id not in pre_map.index:
            continue
        match_events = events.loc[events.match_id.eq(match.match_id)].sort_values(
            ["period", "event_time_seconds", "index"]
        )
        if match_events.empty:
            continue
        pre_row = pre_map.loc[match.match_id]
        for t in snapshot_minutes:
            snapshot_seconds = int(t * 60)
            cut = match_events.loc[match_events.event_time_seconds.le(snapshot_seconds)]
            home = _snapshot_team_features(
                cut, match.home_team, snapshot_seconds, window_seconds
            )
            away = _snapshot_team_features(
                cut, match.away_team, snapshot_seconds, window_seconds
            )
            home_goals, away_goals = _score_at_snapshot(
                cut, match.home_team, match.away_team
            )
            max_seconds = int(cut.event_time_seconds.max()) if not cut.empty else -1
            row = {
                "match_id": match.match_id,
                "kick_off": match.kick_off,
                "snapshot_minute": int(t),
                "snapshot_time_seconds": snapshot_seconds,
                "max_event_minute_used": int(max_seconds // 60) if max_seconds >= 0 else -1,
                "max_event_time_seconds_used": max_seconds,
                "current_home_goals": home_goals,
                "current_away_goals": away_goals,
                "current_score_diff": home_goals - away_goals,
                "man_advantage_home": away["red_cards"] - home["red_cards"],
                "outcome": match.outcome,
                "goal_margin": match.goal_margin,
            }
            for key, value in home.items():
                row[f"live_home_{key}"] = value
            for key, value in away.items():
                row[f"live_away_{key}"] = value
            for key in home:
                if isinstance(home[key], (int, float, np.integer, np.floating)):
                    row[f"live_diff_{key}"] = home[key] - away[key]

            snapshot_minutes_elapsed = max(1, snapshot_seconds / 60.0)
            row["live_momentum_shots"] = (
                    home["recent_shots_per_min"] - (home["shots"] / snapshot_minutes_elapsed)
            )
            row["live_momentum_pressures"] = (
                    home["recent_pressures_per_min"] - (home["pressures"] / snapshot_minutes_elapsed)
            )

            for col in pre_cols:
                row[f"prematch_{col}"] = pre_row[col]
            rows.append(row)

    snapshots = pd.DataFrame(rows)
    assert_time_t_cut(snapshots)
    return snapshots.sort_values(
        ["kick_off", "match_id", "snapshot_minute"]
    ).reset_index(drop=True)


def assert_time_t_cut(snapshots: pd.DataFrame) -> None:
    if snapshots.empty:
        return
    if {
        "max_event_time_seconds_used",
        "snapshot_time_seconds",
    }.issubset(snapshots.columns):
        bad = (
                snapshots.max_event_time_seconds_used
                > snapshots.snapshot_time_seconds
        )
    else:
        bad = snapshots.max_event_minute_used > snapshots.snapshot_minute
    if bad.any():
        raise AssertionError(
            f"Time-t leakage detected in {int(bad.sum())} snapshots: "
            "an event after t was used."
        )


def add_chronological_split(
        pre_match: pd.DataFrame,
        snapshots: pd.DataFrame | None = None,
        *,
        train_fraction: float = 0.70,
        validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[int, str]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1."
        )

    ordered = (
        pre_match[["match_id", "kick_off"]]
        .drop_duplicates()
        .sort_values(["kick_off", "match_id"])
        .reset_index(drop=True)
    )
    n = len(ordered)
    if n < 3:
        raise ValueError(
            "At least three matches are required for train/validation/test."
        )
    train_end = max(1, int(np.floor(n * train_fraction)))
    val_end = max(
        train_end + 1,
        int(np.floor(n * (train_fraction + validation_fraction)))
    )
    val_end = min(val_end, n - 1)

    split_map = {
        int(mid): "train" if i < train_end else "validation" if i < val_end else "test"
        for i, mid in enumerate(ordered.match_id.tolist())
    }
    pre_out = pre_match.copy()
    pre_out["split"] = pre_out.match_id.map(split_map)

    snapshots_out = None
    if snapshots is not None:
        snapshots_out = snapshots.copy()
        snapshots_out["split"] = snapshots_out.match_id.map(split_map)
        if snapshots_out["split"].isna().any():
            raise AssertionError(
                "A snapshot has no parent match in the split map."
            )
        if snapshots_out.groupby("match_id").split.nunique().max() != 1:
            raise AssertionError(
                "Snapshots from one match crossed split boundaries."
            )

    split_sets = {
        name: set(pre_out.loc[pre_out.split.eq(name), "match_id"])
        for name in ("train", "validation", "test")
    }
    if split_sets["train"] & split_sets["validation"]:
        raise AssertionError("Train and validation share match_id values.")
    if split_sets["train"] & split_sets["test"]:
        raise AssertionError("Train and test share match_id values.")
    if split_sets["validation"] & split_sets["test"]:
        raise AssertionError("Validation and test share match_id values.")

    return pre_out, snapshots_out, split_map


def normalize_team_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    text = text.casefold().replace("&", "and")
    text = re.sub(r"\b(fc|afc|cf|football club)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


DEFAULT_TEAM_ALIASES = {
    "manunited": "manchesterunited",
    "mancity": "manchestercity",
    "spurs": "tottenhamhotspur",
    "tottenham": "tottenhamhotspur",
    "westbrom": "westbromwichalbion",
    "westham": "westhamunited",
    "leicester": "leicestercity",
    "newcastle": "newcastleunited",
    "stoke": "stokecity",
    "norwich": "norwichcity",
    "swansea": "swanseacity",
    "athleticbilbao": "athleticclub",
    "athbilbao": "athleticclub",
    "atleticobilbao": "athleticclub",
    "athletic": "athleticclub",
    "atleticomadrid": "atleticomadrid",
    "athmadrid": "atleticomadrid",
    "atletico": "atleticomadrid",
    "barcelona": "barcelona",
    "realmadrid": "realmadrid",
    "real": "realmadrid",
    "valencia": "valencia",
    "sevilla": "sevilla",
    "sevillafc": "sevilla",
    "villareal": "villareal",
    "villarreal": "villareal",
    "espanyol": "espanyol",
    "realsociedad": "realsociedad",
    "sociedad": "realsociedad",
    "realbetis": "betis",
    "betis": "betis",
    "malaga": "malaga",
    "granada": "granada",
    "getafe": "getafe",
    "celta": "celta",
    "celtadevigo": "celta",
    "deportivolacoruna": "deportivo",
    "deportivo": "deportivo",
    "lacoruna": "deportivo",
    "rcdeportivolacoruna": "deportivo",
    "sportinggijon": "sportinggijon",
    "sporting": "sportinggijon",
    "eibar": "eibar",
    "rayovallecano": "rayo",
    "rayo": "rayo",
    "leganes": "leganes",
    "osasuna": "osasuna",
    "alaves": "alaves",
    "deportivoalaves": "alaves",
    "alavés": "alaves",
}


def tag_odds_to_matches(
        matches: pd.DataFrame,
        odds: pd.DataFrame,
        *,
        aliases: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import difflib

    aliases = {**DEFAULT_TEAM_ALIASES, **(aliases or {})}
    left = matches.copy()
    right = odds.copy()

    # Football-Data changed from two-digit to four-digit years across seasons.
    # Parse day-first flexibly so both 21/01/18 and 21/01/2018 are supported.
    right["match_date"] = pd.to_datetime(
        right["Date"], format="mixed", dayfirst=True, errors="coerce"
    ).dt.normalize()
    left["match_date"] = pd.to_datetime(
        left["match_date"], errors="coerce"
    ).dt.normalize()

    for frame, home_col, away_col in [
        (left, "home_team", "away_team"),
        (right, "HomeTeam", "AwayTeam"),
    ]:
        frame["home_key"] = frame[home_col].map(normalize_team_name).replace(aliases)
        frame["away_key"] = frame[away_col].map(normalize_team_name).replace(aliases)

    odds_cols = [c for c in ["B365H", "B365D", "B365A"] if c in right.columns]
    if len(odds_cols) != 3:
        alternatives = [c for c in ["PSH", "PSD", "PSA"] if c in right.columns]
        if len(alternatives) != 3:
            raise ValueError(
                "Expected B365H/B365D/B365A or PSH/PSD/PSA in odds CSV."
            )
        odds_cols = alternatives

    join_keys = ["match_date", "home_key", "away_key"]

    duplicate_odds_keys = right.duplicated(join_keys, keep=False)
    duplicate_key_frame = right.loc[duplicate_odds_keys, join_keys].drop_duplicates()
    right_unique = right.loc[~duplicate_odds_keys, [*join_keys, *odds_cols]].copy()

    tagged = left.merge(
        right_unique,
        on=join_keys,
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    unmatched = tagged[tagged["_merge"] == "left_only"].copy()
    if not unmatched.empty:
        print(f"  Attempting fuzzy fallback for {len(unmatched)} unmatched matches...")
        right_keys = right_unique[["match_date", "home_key", "away_key"]].drop_duplicates()
        fuzzy_matches = []
        for idx, row in unmatched.iterrows():
            date = row["match_date"]
            home_k = row["home_key"]
            away_k = row["away_key"]
            candidates = right_keys[right_keys["match_date"] == date]
            if candidates.empty:
                continue
            home_scores = [
                difflib.SequenceMatcher(None, home_k, k).ratio()
                for k in candidates["home_key"]
            ]
            away_scores = [
                difflib.SequenceMatcher(None, away_k, k).ratio()
                for k in candidates["away_key"]
            ]
            avg_scores = [(h + a) / 2 for h, a in zip(home_scores, away_scores)]
            best_idx = np.argmax(avg_scores)
            if avg_scores[best_idx] > 0.85:
                best_candidate = candidates.iloc[best_idx]
                fuzzy_matches.append({
                    "match_id": row["match_id"],
                    "match_date": date,
                    "home_key": home_k,
                    "away_key": away_k,
                    "matched_home_key": best_candidate["home_key"],
                    "matched_away_key": best_candidate["away_key"],
                })
        if fuzzy_matches:
            fuzzy_df = pd.DataFrame(fuzzy_matches).drop_duplicates("match_id")
            # Keep match_id from the left-side match while attaching the selected
            # Football-Data identity key; this avoids the former fuzzy-join bug
            # where match_id was lost before assignment.
            fuzzy_odds = fuzzy_df.merge(
                right_unique,
                left_on=["match_date", "matched_home_key", "matched_away_key"],
                right_on=["match_date", "home_key", "away_key"],
                how="left",
                validate="many_to_one",
                suffixes=("_left", "_odds"),
            )
            for _, row in fuzzy_odds.iterrows():
                mask = tagged["match_id"].eq(row["match_id"])
                if mask.any() and row[odds_cols].notna().all():
                    tagged.loc[mask, odds_cols] = row[odds_cols].to_numpy()
                    tagged.loc[mask, "_merge"] = "both"

    numeric_odds = tagged[odds_cols].apply(pd.to_numeric, errors="coerce")
    valid_odds = numeric_odds.notna().all(axis=1) & numeric_odds.gt(1.0).all(axis=1)
    raw = 1.0 / numeric_odds
    fair = raw.div(raw.sum(axis=1), axis=0)
    tagged[["market_p_H", "market_p_D", "market_p_A"]] = fair.to_numpy()
    tagged["odds_tagged"] = tagged["_merge"].eq("both") & valid_odds

    if tagged.odds_tagged.any():
        sums = tagged.loc[
            tagged.odds_tagged,
            ["market_p_H", "market_p_D", "market_p_A"],
        ].sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-10):
            raise AssertionError("De-vigged market probabilities do not sum to one.")

    coverage = (
        tagged.groupby(tagged.match_date.dt.year, dropna=False)["odds_tagged"]
        .agg(["count", "sum"])
        .rename(columns={"count": "matches", "sum": "tagged"})
        .reset_index(names="year")
    )
    coverage["coverage_rate"] = coverage.tagged / coverage.matches

    excluded = tagged.loc[
        ~tagged.odds_tagged,
        ["match_id", "match_date", "home_team", "away_team", "home_key",
         "away_key", "_merge"],
    ].copy()
    excluded["reason"] = np.where(
        excluded["_merge"].eq("left_only"),
        "No pre-match identity match in odds source (including fuzzy)",
        "Matched row has missing or invalid 1X2 odds",
    )

    if not duplicate_key_frame.empty:
        duplicate_excluded = left.merge(
            duplicate_key_frame,
            on=join_keys,
            how="inner",
        )
        if not duplicate_excluded.empty:
            duplicate_excluded = duplicate_excluded[
                ["match_id", "match_date", "home_team", "away_team", "home_key",
                 "away_key"]
            ].copy()
            duplicate_excluded["reason"] = "Duplicate identity key in odds source"
            excluded = pd.concat(
                [excluded.drop(columns="_merge"), duplicate_excluded],
                ignore_index=True,
            ).drop_duplicates("match_id", keep="last")
        else:
            excluded = excluded.drop(columns="_merge")
    else:
        excluded = excluded.drop(columns="_merge")

    return tagged.drop(columns="_merge"), coverage, excluded


def download_odds_csv(
        url: str = DEFAULT_ODDS_URL,
        cache_path: str | Path = "cache/odds/E0_1516.csv",
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    return pd.read_csv(cache_path)


def make_demo_football_data(random_state: int = 42, n_matches: int = 420):
    from sklearn.datasets import make_classification

    X, y_int = make_classification(
        n_samples=n_matches,
        n_features=16,
        n_informative=10,
        n_redundant=3,
        n_clusters_per_class=2,
        n_classes=3,
        weights=[0.46, 0.27, 0.27],
        class_sep=0.9,
        flip_y=0.04,
        random_state=random_state,
    )
    labels = np.array(["H", "D", "A"])[y_int]
    feature_names = [
        "diff_form_points",
        "diff_form_xg",
        "diff_form_shots",
        "diff_form_passes",
        "diff_form_pressures",
        "diff_form_def_actions",
        "home_rest_days",
        "away_rest_days",
        "home_form_xg",
        "away_form_xg",
        "home_form_goals",
        "away_form_goals",
        "home_event_share",
        "away_event_share",
        "home_red_cards",
        "away_red_cards",
    ]
    frame = pd.DataFrame(X, columns=feature_names)
    frame["match_id"] = np.arange(100000, 100000 + n_matches)
    frame["kick_off"] = pd.date_range("2015-08-01", periods=n_matches, freq="D")
    frame["outcome"] = labels
    frame["goal_margin"] = np.clip(
        np.round(X[:, 0] + 0.5 * X[:, 1] + np.where(labels == "H", 1,
                                                    np.where(labels == "A", -1, 0))),
        -5,
        5,
    ).astype(int)
    frame, _, _ = add_chronological_split(frame)
    return frame