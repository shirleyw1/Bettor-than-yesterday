"""Weekly NFL betting model, version 3: anytime TD, five yardage/reception props, and game cheat codes.

Setup (once):   pip install nflreadpy pandas scikit-learn pyarrow
Run (weekly):   python td_model.py     (Wednesday, Friday and Sunday morning, so injury news is included)
Output:         site/index.html (open in any browser), site/td_picks.csv and history.csv

TD counts rushing and receiving TDs only (passing TDs and return TDs are not included).
"""
import datetime as dt
import html
import math
import os
import sys

import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

POSITIONS = ["QB", "RB", "WR", "TE"]
WINDOWS = (4, 12)
BASE = ["carries", "targets", "rz_carries", "gl_carries", "rz_targets", "td", "carry_share", "tgt_share",
        "rz_share", "gl_share", "attempts", "passing_yards", "passing_tds", "rushing_yards",
        "receiving_yards", "receptions"]
OPP = ["td", "passing_yards", "passing_tds", "rushing_yards", "receiving_yards", "receptions"]
CTX = ["implied", "spread", "total", "is_home"]
POS = [f"pos_{p}" for p in POSITIONS]
INJ = ["vac_carry", "vac_tgt", "vac_rz", "vac_gl", "adj_carry_share", "adj_tgt_share", "adj_rz_share", "adj_gl_share"]
TD_STATS = ["carries", "targets", "rz_carries", "gl_carries", "rz_targets", "td", "carry_share",
            "tgt_share", "rz_share", "gl_share"]
RECV = ["receiving_yards", "targets", "receptions", "tgt_share", "rz_targets"]
PASS = ["passing_yards", "attempts", "passing_tds"]
# key: (title, unit, feature stats, who qualifies, distribution)
PROPS = {
    "passing_yards": ("Passing yards", "yds", PASS, lambda d: (d["position"] == "QB") & (d["attempts_l4"] >= 10), "normal"),
    "passing_tds": ("Passing TDs", "TDs", PASS, lambda d: (d["position"] == "QB") & (d["attempts_l4"] >= 10), "pois"),
    "rushing_yards": ("Rushing yards", "yds", ["rushing_yards", "carries", "rz_carries", "gl_carries", "carry_share"],
                      lambda d: d["carries_adj"] >= 4, "normal"),
    "receiving_yards": ("Receiving yards", "yds", RECV, lambda d: d["targets_adj"] >= 2, "normal"),
    "receptions": ("Receptions", "rec", RECV, lambda d: d["targets_adj"] >= 2, "pois"),
}
TOP_TD, TOP_PROP, TOP_LONG = 60, 40, 50
OUT_DIR = "site"          # the web page is written here
HISTORY = "history.csv"   # every week's picks and results, kept so the Track record tab can grade them
HCOLS = ["season", "week", "market", "player_id", "player", "team", "position", "value", "score", "naive", "actual", "void"]


def lag(cols):
    return [f"{c}_l{w}" for c in cols for w in WINDOWS]


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))


def _gb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250,
                                          l2_regularization=2.0, min_samples_leaf=60, random_state=0)


MODELS = {"logistic": _lr, "boosted": _gb,
          "blend": lambda: VotingClassifier([("lr", _lr()), ("gb", _gb())], voting="soft")}


def nth(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def current_season():
    t = dt.date.today()
    return t.year if t.month >= 3 else t.year - 1


# ---------------------------------------------------------------- data
def load_data(season):
    seasons = list(range(season - 3, season + 1))
    print(f"Downloading {seasons[0]}-{season} data (the first run takes a few minutes)...")
    keys = ["season", "week"]
    ps = nfl.load_player_stats(seasons).to_pandas()
    ps = ps[(ps["season_type"] == "REG") & ps["position"].isin(POSITIONS)].copy()
    if "team" not in ps.columns:
        ps["team"] = ps["recent_team"]
    ps["player"] = ps["player_display_name"] if "player_display_name" in ps.columns else ps["player_name"]
    ps["td"] = ((ps["rushing_tds"].fillna(0) + ps["receiving_tds"].fillna(0)) > 0).astype(int)
    raw = ["carries", "targets", "attempts", "passing_yards", "passing_tds", "rushing_yards", "receiving_yards", "receptions"]
    ps[raw] = ps[raw].fillna(0)
    ps = ps[["player_id", "player", "position", "team", "td"] + keys + raw]

    cols = keys + ["season_type", "play_type", "yardline_100", "posteam", "rusher_player_id", "receiver_player_id"]
    pbp = nfl.load_pbp(seasons).select(cols).to_pandas()
    pbp = pbp[pbp["season_type"] == "REG"]
    run, air = pbp[pbp["play_type"] == "run"], pbp[pbp["play_type"] == "pass"]
    rush = (run.assign(rz_carries=(run["yardline_100"] <= 20).astype(int), gl_carries=(run["yardline_100"] <= 5).astype(int))
            .groupby(keys + ["rusher_player_id"])[["rz_carries", "gl_carries"]].sum()
            .reset_index().rename(columns={"rusher_player_id": "player_id"}))
    rec = (air.assign(rz_targets=(air["yardline_100"] <= 20).astype(int))
           .groupby(keys + ["receiver_player_id"])[["rz_targets"]].sum()
           .reset_index().rename(columns={"receiver_player_id": "player_id"}))
    ps = ps.merge(rush, on=keys + ["player_id"], how="left").merge(rec, on=keys + ["player_id"], how="left")
    for c in ["rz_carries", "gl_carries", "rz_targets"]:
        ps[c] = ps[c].fillna(0)
    ps["rz"] = ps["rz_carries"] + ps["rz_targets"]
    tm = ps.groupby(["team"] + keys)
    for share, col in [("carry_share", "carries"), ("tgt_share", "targets"), ("rz_share", "rz"), ("gl_share", "gl_carries")]:
        ps[share] = ps[col] / tm[col].transform("sum").clip(lower=1)

    pl = pbp[pbp["play_type"].isin(["run", "pass"])]
    off = (pl.assign(is_pass=(pl["play_type"] == "pass").astype(int)).groupby(["posteam"] + keys)
           .agg(plays=("is_pass", "size"), pass_rate=("is_pass", "mean")).reset_index().rename(columns={"posteam": "team"}))

    sch = nfl.load_schedules(seasons).to_pandas()
    sch = sch[sch["game_type"] == "REG"]
    total, spread = sch["total_line"].fillna(45.0), sch["spread_line"].fillna(0.0)  # spread > 0: home favored
    played = sch["result"].notna()
    home = sch[keys].assign(team=sch["home_team"], opp=sch["away_team"], is_home=1, spread=spread, total=total,
                            implied=total / 2 + spread / 2, played=played, pts_for=sch["home_score"], pts_against=sch["away_score"])
    away = sch[keys].assign(team=sch["away_team"], opp=sch["home_team"], is_home=0, spread=-spread, total=total,
                            implied=total / 2 - spread / 2, played=played, pts_for=sch["away_score"], pts_against=sch["home_score"])
    return ps, pd.concat([home, away], ignore_index=True), off


def allowed_history(ps, ctx):
    """Per defense, position and game: total yards, receptions, TD scorers allowed."""
    d = ps.merge(ctx[["season", "week", "team", "opp"]], on=["season", "week", "team"], how="left")
    return d.groupby(["opp", "position", "season", "week"], as_index=False)[OPP].sum()


def opp_features(a, unplayed):
    """Each defense's last 8 games allowed at a position, relative to the league average (1.0 = normal)."""
    up = pd.DataFrame([(o, p, s, w) for o, s, w in unplayed[["opp", "season", "week"]].itertuples(index=False)
                       for p in POSITIONS], columns=["opp", "position", "season", "week"])
    a = pd.concat([a, up], ignore_index=True).sort_values(["opp", "position", "season", "week"]).reset_index(drop=True)
    out = a[["opp", "position", "season", "week"]].copy()
    for c in OPP:
        s = a.groupby(["opp", "position"])[c].shift(1)
        r = s.groupby([a["opp"], a["position"]]).rolling(8, min_periods=3).mean().reset_index(level=[0, 1], drop=True)
        out[f"opp_{c}"] = (r / a.groupby("position")[c].transform("mean")).replace([np.inf, -np.inf], np.nan)
    return out


def absences(ps, ctx, season, week, status):
    """Share of each team's carries, targets, red-zone touches and goal-line carries held by regulars (players from
    the last 4 team games) who missed the game. For this week's games, 'missed' means ruled Out or Doubtful."""
    sh = ["carry_share", "tgt_share", "rz_share", "gl_share"]
    games = ctx[["team", "season", "week"]].drop_duplicates().sort_values(["team", "season", "week"])
    games["gi"] = games.groupby("team").cumcount()
    p = ps.merge(games, on=["team", "season", "week"])
    by_game = {k: g for k, g in p.groupby(["team", "gi"])}
    rows = []
    for t, s, w, gi in games.itertuples(index=False):
        prev = [by_game[(t, k)] for k in range(gi - 4, gi) if (t, k) in by_game]
        if not prev:
            continue
        reg = pd.concat(prev).groupby("player_id")[sh].mean()
        now = by_game.get((t, gi))
        if now is None:
            if (s, w) != (season, week):
                continue
            absent = [pid for pid in reg.index if status and status.get(pid) in ("Out", "Doubtful")]
        else:
            absent = reg.index.difference(now["player_id"])
        rows.append((t, s, w, *reg.loc[absent].sum().values))
    return pd.DataFrame(rows, columns=["team", "season", "week", "vac_carry", "vac_tgt", "vac_rz", "vac_gl"])


def next_week(ctx, season):
    unplayed = ctx[(ctx["season"] == season) & (~ctx["played"])]
    if unplayed.empty:
        sys.exit("No upcoming games found in the schedule. Try again closer to the next game week.")
    week = int(unplayed["week"].min())
    return week, unplayed[unplayed["week"] == week]


def build_features(ps, ctx, a, season, week, unplayed, status):
    # One blank row per recently active player on a team playing this week. Rolling features use
    # shift(1), so a blank row only sees past games (no leakage).
    latest = ps.sort_values(["season", "week"]).groupby("player_id").tail(1)
    latest = latest[(latest["season"] == season) & (latest["week"] >= week - 3)]
    blank = latest[["player_id", "player", "position", "team"]].assign(season=season, week=week)
    blank = blank.merge(unplayed[["season", "week", "team"]], on=["season", "week", "team"])
    df = pd.concat([ps, blank], ignore_index=True)
    df = df.merge(ctx[["season", "week", "team", "opp", "is_home", "spread", "total", "implied"]],
                  on=["season", "week", "team"], how="left")
    df = df.merge(opp_features(a, unplayed), on=["opp", "position", "season", "week"], how="left")
    df = df.merge(absences(ps, ctx, season, week, status), on=["team", "season", "week"], how="left")
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for c in BASE:
        s = df.groupby("player_id")[c].shift(1)
        for w in WINDOWS:
            df[f"{c}_l{w}"] = s.groupby(df["player_id"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    df["n_prior"] = df.groupby("player_id").cumcount()
    for c, v in [("spread", 0.0), ("total", 45.0), ("implied", 22.5), ("is_home", 0)] + [(f"opp_{o}", 1.0) for o in OPP] + [(c, 0.0) for c in INJ[:4]]:
        df[c] = df[c].fillna(v)
    for p in POSITIONS:
        df[f"pos_{p}"] = (df["position"] == p).astype(int)
    # A teammate's absence shifts his usage to the others: scale each player's share up by what was vacated
    for share, vac in [("carry_share", "vac_carry"), ("tgt_share", "vac_tgt"), ("rz_share", "vac_rz"), ("gl_share", "vac_gl")]:
        df[f"adj_{share}"] = df[f"{share}_l4"] / (1 - df[vac]).clip(lower=0.3)
    df["carries_adj"] = df["carries_l4"] / (1 - df["vac_carry"]).clip(lower=0.3)
    df["targets_adj"] = df["targets_l4"] / (1 - df["vac_tgt"]).clip(lower=0.3)
    df["touches_adj"] = df["carries_adj"] + df["targets_adj"]
    df["touches_l4"] = df["carries_l4"] + df["targets_l4"]
    df["rz_touches_l4"] = df["rz_carries_l4"] + df["rz_targets_l4"]
    return df[df["n_prior"] >= 3]


# ---------------------------------------------------------------- models
def score_of(ref, x):
    return np.searchsorted(np.sort(ref), x) / len(ref) * 100  # percentile vs. last season's predictions


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def td_model(df, season):
    base_feats = lag(TD_STATS) + CTX + ["opp_td"] + POS
    feats = base_feats + INJ
    d = df[df["touches_adj"] >= 2.0]
    known = d[d["td"].notna()]
    tr, va = known[known["season"] < season - 1], known[known["season"] == season - 1]
    base = log_loss(va["td"], np.full(len(va), tr["td"].mean()))
    scores = {}
    for name, make in MODELS.items():
        p = make().fit(tr[feats], tr["td"]).predict_proba(va[feats])[:, 1]
        scores[name] = (log_loss(va["td"], p), brier_score_loss(va["td"], p), p)
        print(f"TD {name:9s} log loss {scores[name][0]:.4f}  brier {scores[name][1]:.4f}  (average-rate baseline {base:.4f})")
    best = min(scores, key=lambda k: scores[k][0])
    p0 = _lr().fit(tr[base_feats], tr["td"]).predict_proba(va[base_feats])[:, 1]
    inj_note = (f" Adding teammate-injury features moved the simple model's log loss from {log_loss(va['td'], p0):.4f} "
                f"to {scores['logistic'][0]:.4f} (lower is better).")

    # Calibration: shrink overconfident probabilities using last season's out-of-sample predictions
    raw = scores[best][2]
    cal = LogisticRegression(C=1e6).fit(logit(raw), va["td"])

    def fix(p):
        return cal.predict_proba(logit(p))[:, 1]
    adj = fix(raw)
    v = va.assign(p=raw, pc=adj)
    top = v.sort_values("p", ascending=False).groupby(["season", "week"]).head(20)
    print(f"Top 20 each week: model said {top['p'].mean():.1%}, hit {top['td'].mean():.1%}; adjusted {top['pc'].mean():.1%}")

    model = MODELS[best]().fit(known[feats], known["td"])
    cand = d[d["td"].isna()].copy()
    cand["prob"] = fix(model.predict_proba(cand[feats])[:, 1])
    cand["score"] = score_of(adj, cand["prob"])
    note = (f"TD backtest on {season - 1}: log loss {log_loss(va['td'], adj):.3f} (raw {scores[best][0]:.3f}) vs {base:.3f} for "
            f"guessing the average rate. Before adjusting, the top 20 picks each week were given {top['p'].mean():.0%} and hit "
            f"{top['td'].mean():.0%}, so probabilities are now shrunk by that gap (top 20 now {top['pc'].mean():.0%}). "
            f"That adjustment was fit on the same season, so the Track record tab is the real test." + inj_note)
    return cand, note


def prop_model(df, season, key):
    title, unit, stats, who, dist = PROPS[key]
    feats = lag(stats) + CTX + [f"opp_{key}"] + POS + INJ
    d = df[who(df)]
    known = d[d[key].notna()]
    tr, va = known[known["season"] < season - 1], known[known["season"] == season - 1]

    def make():
        return HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.05, max_iter=200,
                                             l2_regularization=2.0, min_samples_leaf=60, random_state=0)
    pv = make().fit(tr[feats], tr[key].clip(lower=0)).predict(va[feats])
    mae, naive = np.abs(va[key] - pv).mean(), np.abs(va[key] - va[f"{key}_l4"]).mean()
    floor = 0.2 * pv.mean()
    cv = float(np.sqrt((((va[key] - pv) / np.maximum(pv, floor)) ** 2).mean()))
    print(f"{title}: average miss {mae:.2f} vs {naive:.2f} using just the last-4 average")
    cand = d[d[key].isna()].copy()
    cand["proj"] = make().fit(known[feats], known[key].clip(lower=0)).predict(cand[feats])
    cand["score"] = score_of(pv, cand["proj"])
    cand["sd"] = cv * np.maximum(cand["proj"], floor)
    return cand, f"{title} backtest on {season - 1}: average miss {mae:.1f} {unit} vs {naive:.1f} using just the last-4 average."


def injury_status(season, week):
    try:
        inj = nfl.load_injuries([season]).to_pandas()
        status = dict(zip(inj[inj["week"] == week]["gsis_id"], inj[inj["week"] == week]["report_status"].fillna("")))
        if status:
            return status
        print("No injury report posted yet for this week. Check injuries yourself.")
    except Exception as e:  # keep going if the injury feed is down or its format changed
        print(f"Could not load the injury report ({e}). Check injuries yourself.")
    return None


def prep(cand, status, sort_col):
    cand = cand.copy()
    cand["status"] = cand["player_id"].map(status).fillna("") if status else ""
    return cand[~cand["status"].isin(["Out", "Doubtful"])].sort_values(sort_col, ascending=False)


# ---------------------------------------------------------------- cards
def american(p):
    p = min(0.99, max(0.01, p))
    return f"-{round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


def need_odds(p):
    """Price at which a bet has +10% expected return if the model's probability is right."""
    dec = 1.10 / min(max(p, 0.01), 0.99)
    return f"+{round((dec - 1) * 100)}" if dec >= 2 else f"-{round(100 / (dec - 1))}"


def side_text(spread):
    return "Pick'em" if spread == 0 else (f"Fav by {spread:g}" if spread > 0 else f"Dog by {-spread:g}")


def head(i, r, big, sub):
    return (f"<header><span class='rk'>{i}</span><div class='who'><b>{html.escape(r.player)}</b>"
            f"<span class='sub'>{r.position} {r.team} {'vs' if r.is_home else '@'} {r.opp}</span></div>"
            f"<div class='pct'><b>{big}</b><span>{sub}</span></div></header>")


def attrs(r, kind, extra=""):
    return f"data-kind='{kind}' data-pos='{r.position}' data-name='{html.escape(r.player.lower(), quote=True)}' {extra}"


def boost(r):
    vac = r.vac_carry if r.position == "RB" else (r.vac_tgt if r.position in ("WR", "TE") else 0)
    return vac >= 0.15


TD_BARS = [("Red zone role", "rz_share_l4", lambda r: f"{r.rz_share_l4 * 100:.0f}% of team red zone touches"),
           ("Volume", "touches_l4", lambda r: f"{r.touches_l4:.1f} touches a game"),
           ("Goal line", "gl_carries_l4", lambda r: f"{r.gl_carries_l4:.1f} goal line carries a game"),
           ("Target share", "tgt_share_l4", lambda r: f"{r.tgt_share_l4 * 100:.0f}% of team targets"),
           ("TD rate", "td_l12", lambda r: f"scored in {r.td_l12 * 100:.0f}% of last 12 games"),
           ("Matchup", "opp_td", lambda r: f"defense allows {r.opp_td:.2f}x the usual TD scorers to {r.position}s")]


def td_card(i, r):
    chips = []
    if r.status == "Questionable":
        chips.append("<span class='chip warn'>Questionable</span>")
    if boost(r):
        chips.append("<span class='chip'>Role boost: teammate out</span>")
    if r.rz_rank <= 10:
        chips.append("<span class='chip'>Top 10 red zone share</span>")
    if r.gl_carries_l4 >= 1.0:
        chips.append("<span class='chip'>Goal line work</span>")
    if r.spread >= 3:
        chips.append("<span class='chip'>Favored</span>")
    if r.opp_td >= 1.25:
        chips.append(f"<span class='chip'>Soft matchup vs {r.position}</span>")
    if r.opp_td <= 0.75:
        chips.append("<span class='chip dim'>Tough matchup</span>")
    if r.total >= 48:
        chips.append("<span class='chip'>High scoring game</span>")
    bars = "".join(
        f"<div class='bar'><div class='bl'><span>{lab}</span><i>{fmt(r)}</i></div><div class='track'><span style='width:"
        f"{0 if getattr(r, col) <= 0 else getattr(r, 'pct_' + col) * 100:.0f}%'></span></div></div>" for lab, col, fmt in TD_BARS)
    return (f"<article class='card' {attrs(r, 'td', f'data-p={r.prob:.4f}')}>" + head(i, r, f"{r.score:.0f}", f"score &middot; {r.prob * 100:.0f}% &middot; fair {american(r.prob)}")
            + f"<div class='chips'>{''.join(chips)}</div>"
            f"<p class='script'>Spread <b>{side_text(r.spread)}</b> Total <b>{r.total:g}</b> Implied <b>{r.implied:.1f} pts</b></p>"
            f"<p class='script'>Worth a bet at <b>{need_odds(r.prob)}</b> or better (fair price plus a 10% cushion)</p>"
            "<label class='book'><input class='odds' placeholder='Sportsbook odds, e.g. +150' aria-label='Sportsbook odds'><span class='edge'></span></label>"
            f"<details><summary>Why this player</summary>{bars}</details></article>")


def prop_card(i, r, key):
    title, unit, stats, who, dist = PROPS[key]
    chips = ["<span class='chip warn'>Questionable</span>"] if r.status == "Questionable" else []
    if boost(r) and key in ("rushing_yards", "receiving_yards", "receptions"):
        chips.append("<span class='chip'>Role boost: teammate out</span>")
    opp = getattr(r, f"opp_{key}")
    if opp >= 1.2:
        chips.append("<span class='chip'>Soft matchup</span>")
    if opp <= 0.8:
        chips.append("<span class='chip dim'>Tough matchup</span>")
    if r.spread >= 3:
        chips.append("<span class='chip'>Favored</span>")
    l4, l12 = getattr(r, f"{key}_l4"), getattr(r, f"{key}_l12")
    if l12 > 0 and r.proj / l12 >= 1.15:
        chips.append("<span class='chip'>Model above recent form</span>")
    why = (f"<p class='script'>Last 4 games average <b>{l4:.1f}</b> Last 12 average <b>{l12:.1f}</b></p>"
           f"<p class='script'>Defense allows <b>{opp:.2f}x</b> the usual {title.lower()} to {r.position}s</p>"
           f"<p class='script'>Spread <b>{side_text(r.spread)}</b> Total <b>{r.total:g}</b> Implied <b>{r.implied:.1f} pts</b></p>")
    return (f"<article class='card' {attrs(r, dist, f'data-mu={r.proj:.3f} data-sd={r.sd:.3f}')}>" + head(i, r, f"{r.score:.0f}", f"score &middot; projects {r.proj:.1f} {unit}")
            + f"<div class='chips'>{''.join(chips)}</div>"
            "<label class='book'><input class='line' placeholder='Sportsbook line, e.g. 249.5' aria-label='Sportsbook line'><span class='edge'></span></label>"
            f"<details><summary>Why this player</summary>{why}</details></article>")


# ---------------------------------------------------------------- game cheat codes
DEF_FACTS = [("QB", "passing_yards", "passing yards", "QBs"), ("QB", "passing_tds", "passing TDs", "QBs"),
             ("RB", "rushing_yards", "rushing yards", "RBs"), ("RB", "td", "TD scorers", "RBs"),
             ("RB", "receptions", "receptions", "RBs"), ("WR", "receiving_yards", "receiving yards", "WRs"),
             ("WR", "td", "TD scorers", "WRs"), ("TE", "receiving_yards", "receiving yards", "TEs"),
             ("TE", "td", "TD scorers", "TEs")]


def team_tables(ctx, off, a):
    played = ctx[ctx["played"]].sort_values(["season", "week"])
    pts = played.groupby("team").tail(8).groupby("team")[["pts_for", "pts_against"]].mean()
    pace = off.sort_values(["season", "week"]).groupby("team").tail(8).groupby("team")[["plays", "pass_rate"]].mean()
    t = pts.join(pace)
    ranks = t.rank(ascending=False, method="min")
    a8 = a.sort_values(["season", "week"]).groupby(["opp", "position"]).tail(8).groupby(["opp", "position"])[OPP].mean()
    return t, ranks, a8, a8.groupby(level="position").rank(ascending=False, method="min")


def team_block(team, t, ranks, a8, rk, pool, allc):
    n = len(t)
    items = []
    if team in t.index:
        r, x = ranks.loc[team], t.loc[team]
        items.append(f"Offense: scores {x.pts_for:.1f} a game ({nth(r.pts_for)}), {x.plays:.0f} plays a game ({nth(r.plays)}), "
                     f"passes {x.pass_rate:.0%} of the time ({nth(r.pass_rate)} most).")
        items.append(f"Defense: allows {x.pts_against:.1f} points a game ({nth(r.pts_against)} most).")
    facts = []
    for pos, col, label, who in DEF_FACTS:
        try:
            v, k = a8.loc[(team, pos), col], rk.loc[(team, pos), col]
        except KeyError:
            continue
        if k <= 6:
            facts.append((k, f"Allows {v:.1f} {label} a game to {who} over its last 8 games, {nth(k)} most."))
        elif k >= n - 5:
            facts.append((n + 1 - k, f"Allows {v:.1f} {label} a game to {who} over its last 8 games, {nth(n + 1 - k)} fewest."))
    items += [s for _, s in sorted(facts)[:4]]
    tp = pool[pool["team"] == team]
    if len(tp):
        rz, gl, tg = tp.nlargest(1, "rz_share_l4").iloc[0], tp.nlargest(1, "gl_carries_l4").iloc[0], tp.nlargest(1, "tgt_share_l4").iloc[0]
        items.append(f"Red zone leader: {rz.player} ({rz.rz_share_l4 * 100:.0f}% of the team's red zone touches).")
        if gl.gl_carries_l4 >= 0.8:
            items.append(f"Goal line back: {gl.player} ({gl.gl_carries_l4:.1f} goal line carries a game).")
        items.append(f"Target leader: {tg.player} ({tg.tgt_share_l4 * 100:.0f}% of the team's targets).")
    hurt = allc[(allc["team"] == team) & allc["status"].isin(["Out", "Doubtful", "Questionable"]) & (allc["touches_l4"] >= 3)]
    if len(hurt):
        items.append("Injury report: " + ", ".join(f"{r.player} ({r.status}; {r.carry_share_l4 * 100:.0f}% of carries, {r.tgt_share_l4 * 100:.0f}% of targets)" for r in hurt.itertuples()) + ".")
    return f"<h3>{team}</h3><ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in items) + "</ul>"


def td_why(r):
    w = []
    if r.rz_rank <= 10:
        w.append(f"top 10 red zone share ({r.rz_share_l4 * 100:.0f}% of team red zone touches)")
    if r.gl_carries_l4 >= 1.0:
        w.append(f"{r.gl_carries_l4:.1f} goal line carries a game")
    w.append(f"{r.touches_l4:.1f} touches a game")
    if boost(r):
        w.append("teammate out, bigger role")
    if r.opp_td >= 1.25:
        w.append(f"defense allows {r.opp_td:.2f}x the usual TD scorers to {r.position}s")
    if r.spread >= 3:
        w.append(f"team favored by {r.spread:g}")
    w.append(f"scored in {r.td_l12 * 100:.0f}% of last 12 games")
    if r.status == "Questionable":
        w.append("listed Questionable")
    return "; ".join(w)


def prop_why(r, key):
    opp = getattr(r, f"opp_{key}")
    w = [f"last 4 games average {getattr(r, key + '_l4'):.1f}"]
    if opp >= 1.2:
        w.append(f"soft matchup, defense allows {opp:.2f}x the usual")
    elif opp <= 0.8:
        w.append(f"tough matchup, defense allows {opp:.2f}x the usual")
    if boost(r) and key in ("rushing_yards", "receiving_yards", "receptions"):
        w.append("teammate out, bigger role")
    if r.spread >= 3:
        w.append(f"team favored by {r.spread:g}")
    if r.status == "Questionable":
        w.append("listed Questionable")
    return "; ".join(w)


def game_cards(unplayed, tabs, pool, allc, propc):
    t, ranks, a8, rk = tabs
    out = []
    for g in unplayed[unplayed["is_home"] == 1].sort_values("total", ascending=False).itertuples():
        teams = [g.opp, g.team]
        fav = g.team if g.spread > 0 else g.opp
        line = "Pick'em" if g.spread == 0 else f"{fav} by {abs(g.spread):g}"
        tds = pool[pool["team"].isin(teams)].nlargest(5, "score")
        lead = f"{tds.iloc[0].player} ({tds.iloc[0].score:.0f})" if len(tds) else "none"
        td_items = "".join(f"<li><b>{html.escape(r.player)}</b> ({r.team} {r.position}) score {r.score:.0f}, {r.prob * 100:.0f}% chance, "
                           f"fair {american(r.prob)}. {html.escape(td_why(r))}.</li>" for r in tds.itertuples())
        prop_items = ""
        for key, (title, unit, *_rest) in PROPS.items():
            pr = propc[key]
            for r in pr[pr["team"].isin(teams)].nlargest(2, "score").itertuples():
                prop_items += (f"<li><b>{title}:</b> {html.escape(r.player)} ({r.team}) score {r.score:.0f}, projects {r.proj:.1f} {unit}. "
                               f"{html.escape(prop_why(r, key))}.</li>")
        out.append(f"<details class='card'><summary class='game'>{g.opp} @ {g.team}<span class='sub'>{line}, total {g.total:g}. "
                   f"Top TD score: {html.escape(lead)}</span></summary>"
                   f"<h3>Best touchdown scorers</h3><ul>{td_items or '<li>No qualifying players.</li>'}</ul>"
                   f"<h3>Best props</h3><ul>{prop_items or '<li>No qualifying props.</li>'}</ul>"
                   + team_block(g.opp, t, ranks, a8, rk, pool, allc) + team_block(g.team, t, ranks, a8, rk, pool, allc) + "</details>")
    return "\n".join(out)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Week __WEEK__ NFL picks</title>
<style>
:root{--bg:#070b0a;--card:#0f1714;--ink:#e8f1ec;--mute:#8aa196;--line:#1f2d27;--g:#5be37d;--warn:#f0b429;--bad:#ff6b6b;
padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;padding:20px 14px 40px}
main{max-width:720px;margin:0 auto}
h1{font-size:1.4rem;margin:0 0 4px}h3{margin:14px 0 4px;font-size:1rem;color:var(--g)}
.meta{color:var(--mute);margin:0 0 14px}
.note{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--g);padding:10px 14px;margin:0 0 14px;border-radius:8px;font-size:.92rem}
nav{display:flex;gap:6px;overflow-x:auto;margin:0 0 12px;padding-bottom:4px}
nav button{font:inherit;font-size:.88rem;white-space:nowrap;background:var(--card);color:var(--mute);border:1px solid var(--line);border-radius:999px;padding:6px 14px}
nav button.on{color:#06210f;background:var(--g);border-color:var(--g);font-weight:600}
.tools{display:flex;gap:8px;margin:0 0 12px}
select,input{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);min-width:0}
.tools input{flex:1}
.card{display:block;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:0 0 12px}
.card[hidden],section[hidden]{display:none}
header{display:flex;align-items:center;gap:12px}
.rk{color:var(--mute);font-variant-numeric:tabular-nums;min-width:1.6em}
.who{flex:1;min-width:0}.who b{display:block;font-size:1.05rem}
.sub{display:block;color:var(--mute);font-size:.85rem;font-weight:400}
.pct{text-align:right}.pct b{display:block;font-size:1.7rem;line-height:1.1;color:var(--g)}.pct span{color:var(--mute);font-size:.78rem}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 0}
.chip{border:1px solid var(--g);color:var(--g);border-radius:999px;padding:2px 10px;font-size:.78rem}
.chip.warn{border-color:var(--warn);color:var(--warn)}.chip.dim{border-color:var(--line);color:var(--mute)}
.script{color:var(--mute);margin:8px 0;font-size:.88rem}.script b{color:var(--ink);font-weight:600;margin-right:10px}
.book{display:flex;align-items:center;gap:10px;margin:8px 0 6px}.book input{flex:1}
.edge{font-weight:600;font-size:.88rem}.edge.good{color:var(--g)}.edge.ok{color:var(--warn)}.edge.bad{color:var(--bad)}
summary{cursor:pointer;color:var(--g);font-size:.9rem;padding:6px 0}summary.game{color:var(--ink);font-weight:600;font-size:1.05rem}
ul{margin:0;padding-left:18px;font-size:.9rem}li{margin:4px 0}
.bar{margin:8px 0}.bl{display:flex;justify-content:space-between;gap:8px;font-size:.88rem}.bl i{color:var(--mute);font-style:normal;text-align:right}
.track{height:6px;background:var(--line);border-radius:3px;margin-top:4px;overflow:hidden}.track span{display:block;height:100%;background:var(--g)}
table.rec{width:100%;border-collapse:collapse;font-size:.88rem}table.rec th,table.rec td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}table.rec th:first-child,table.rec td:first-child{text-align:left}table.rec th{color:var(--mute);font-weight:600}
footer{color:var(--mute);font-size:.85rem;margin-top:14px}footer p{margin:6px 0}
:focus-visible{outline:2px solid var(--g);outline-offset:2px}
</style></head><body><main>
<h1>Week __WEEK__ NFL picks</h1>
<p class="meta">Season __SEASON__, built __BUILT__</p>
<p class="note">Score runs 0 to 100 and shows how a player ranks against every player-game the model scored last season. Type a sportsbook's odds or line into a card to compare it with the model. On a touchdown card, expected return is what a $1 bet earns on average if the model is right. The model has error, so I'd want +10% or more, and extra caution on longshots. The model adjusts for teammates ruled Out or Doubtful on the official injury report, but it can't see game-day inactives announced later, so check them before kickoff.</p>
<nav>__NAV__</nav>
<div class="tools">
<select id="pos" aria-label="Position"><option value="">All</option><option>RB</option><option>WR</option><option>TE</option><option>QB</option></select>
<input id="q" type="search" placeholder="Search player" aria-label="Search player">
</div>
__SECTIONS__
<footer>__INFO__</footer>
<script>
const cards=[...document.querySelectorAll('.card')],$=s=>document.querySelector(s);
const am=p=>{p=Math.min(.99,Math.max(.01,p));return p>=.5?'-'+Math.round(100*p/(1-p)):'+'+Math.round(100*(1-p)/p)};
const erf=x=>{const t=1/(1+.3275911*Math.abs(x)),y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x);return x>=0?y:-y};
const Phi=z=>.5*(1+erf(z/Math.SQRT2));
const pois=(k,m)=>{let s=0,t=Math.exp(-m);for(let i=0;i<=k;i++){s+=t;t*=m/(i+1)}return s};
function f(){const p=$('#pos').value,s=$('#q').value.toLowerCase();cards.forEach(c=>{c.hidden=!!(c.dataset.pos&&((p&&c.dataset.pos!==p)||(s&&!c.dataset.name.includes(s))))})}
$('#pos').onchange=f;$('#q').oninput=f;
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('section').forEach(s=>s.hidden=s.id!==b.dataset.t);$('.tools').hidden=['games','record'].includes(b.dataset.t)});
cards.forEach(c=>{const i=c.querySelector('.odds,.line'),e=c.querySelector('.edge');if(!i)return;
i.oninput=()=>{const k=c.dataset.kind;
if(k==='td'){const o=parseInt(i.value.replace(/[^\d+-]/g,''),10);
if(!o||Math.abs(o)<100){e.textContent='';e.className='edge';return}
const p=+c.dataset.p,dec=o>0?o/100+1:100/-o+1,ev=(p*dec-1)*100;
e.textContent='Expected return '+(ev>=0?'+':'')+ev.toFixed(0)+'% per $1';e.className='edge '+(ev>=10?'good':ev>=0?'ok':'bad');return}
const L=parseFloat(i.value),mu=+c.dataset.mu,sd=+c.dataset.sd;
if(isNaN(L)){e.textContent='';return}
const po=k==='pois'?1-pois(Math.floor(L),mu):1-Phi((L-mu)/sd);
e.textContent='Over '+Math.round(po*100)+'% (fair '+am(po)+') / Under '+Math.round((1-po)*100)+'%'}});
</script>
</main></body></html>"""


def write_outputs(sections, td, week, season, notes, have_injuries):
    nav = "".join(f"<button data-t='{k}' class='{'on' if i == 0 else ''}'>{label}</button>" for i, (k, label, _) in enumerate(sections))
    body = "".join(f"<section id='{k}'{'' if i == 0 else ' hidden'}>{content}</section>" for i, (k, _, content) in enumerate(sections))
    info = "".join(f"<p>{html.escape(n)}</p>" for n in notes)
    info += "<p>Bars compare a player to others at his position. Usage numbers are averages over the last 4 games unless noted.</p>"
    if not have_injuries:
        info += "<p><b>No injury report was available, so injured players are NOT filtered out.</b></p>"
    page = (TEMPLATE.replace("__WEEK__", str(week)).replace("__SEASON__", str(season))
            .replace("__BUILT__", dt.date.today().strftime("%B %d, %Y").replace(" 0", " "))
            .replace("__NAV__", nav).replace("__SECTIONS__", body).replace("__INFO__", info))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    out = td[["player", "position", "team", "opp", "is_home", "score", "prob", "status", "touches_l4", "rz_touches_l4",
              "gl_carries_l4", "tgt_share_l4", "rz_share_l4", "opp_td", "spread", "total", "implied"]].copy()
    out["fair_odds"] = out["prob"].map(american)
    out.to_csv(os.path.join(OUT_DIR, "td_picks.csv"), index=False)


def pick_rows(season, week, market, frame, value_col, naive_col):
    return pd.DataFrame({"season": season, "week": week, "market": market, "player_id": frame["player_id"].values,
                         "player": frame["player"].values, "team": frame["team"].values, "position": frame["position"].values,
                         "value": frame[value_col].values, "score": frame["score"].values, "naive": frame[naive_col].values,
                         "actual": np.nan, "void": False})


def update_history(new, season, teams, ps, ctx):
    """Save this run's picks (replacing earlier runs for games not yet played; finished games stay frozen),
    then fill in actual results for every pick whose game is over."""
    try:
        h = pd.read_csv(HISTORY)
    except FileNotFoundError:
        h = pd.DataFrame(columns=HCOLS)
    drop = (h["season"] == season) & (h["week"] == new["week"].iloc[0]) & h["team"].isin(teams)
    h = pd.concat([h[~drop], new], ignore_index=True)
    h["actual"] = pd.to_numeric(h["actual"], errors="coerce")
    act = ps.set_index(["season", "week", "player_id"])
    act = act[~act.index.duplicated()]
    for m in h["market"].unique():
        sel = (h["market"] == m) & h["actual"].isna()
        if sel.any():
            keys = pd.MultiIndex.from_frame(h.loc[sel, ["season", "week", "player_id"]])
            h.loc[sel, "actual"] = act["td" if m == "td_long" else m].reindex(keys).values
    played = ctx[ctx["played"]].set_index(["season", "week", "team"]).index
    h["void"] = pd.MultiIndex.from_frame(h[["season", "week", "team"]]).isin(played) & h["actual"].isna()  # game over, player didn't play
    h[HCOLS].to_csv(HISTORY, index=False)
    return h


def record_html(h):
    h = h.copy()
    h["rank"] = h.groupby(["season", "week", "market"])["score"].rank(ascending=False, method="first")
    g = h[h["actual"].notna()]
    if g.empty:
        return ("<div class='card'><p class='script'>No graded picks yet. After this week's games finish, the next run grades every pick "
                "here: how often the top TD picks scored, and how far off the projections were. Results build up week by week.</p></div>")
    td = g[g["market"] == "td"]
    rows = "".join(
        f"<tr><td>{s} W{w}</td><td>{x[x['rank'] <= 10]['value'].mean():.0%}</td><td>{x[x['rank'] <= 10]['actual'].mean():.0%}</td>"
        f"<td>{x['value'].mean():.0%}</td><td>{x['actual'].mean():.0%}</td></tr>" for (s, w), x in td.groupby(["season", "week"]))
    t10 = td[td["rank"] <= 10]
    rows += (f"<tr><td><b>All weeks</b></td><td><b>{t10['value'].mean():.0%}</b></td><td><b>{t10['actual'].mean():.0%}</b></td>"
             f"<td><b>{td['value'].mean():.0%}</b></td><td><b>{td['actual'].mean():.0%}</b></td></tr>") if len(td) else ""
    ls = g[g["market"] == "td_long"]
    ls_txt = (f"Longshot TD picks: the model said {ls['value'].mean():.0%}, {ls['actual'].mean():.0%} scored ({len(ls)} picks). " if len(ls) else "")
    props = ""
    for key, (title, unit, *_rest) in PROPS.items():
        x = g[g["market"] == key]
        if len(x):
            props += (f"<tr><td>{title}</td><td>{len(x)}</td><td>{(x['actual'] - x['value']).abs().mean():.1f}</td>"
                      f"<td>{(x['actual'] - x['naive']).abs().mean():.1f}</td></tr>")
    return (f"<div class='card'><h3>Touchdown picks: what the model said vs what happened</h3>"
            f"<table class='rec'><tr><th>Week</th><th>Top 10 said</th><th>Top 10 hit</th><th>All picks said</th><th>All picks hit</th></tr>{rows}</table>"
            f"<p class='script'>Ten picks a week is a small sample, so one extra TD moves the hit rate 10 points. Look at the All weeks row once several weeks are in. "
            f"If 'hit' keeps landing below 'said', trust the percentages less.</p></div>"
            f"<div class='card'><h3>Props: average miss of the model vs just using the last 4 games</h3>"
            f"<table class='rec'><tr><th>Prop</th><th>Picks</th><th>Model miss</th><th>Last 4 miss</th></tr>{props}</table></div>"
            f"<p class='script'>{ls_txt}{int(h['void'].sum())} picks were voided because the player did not play. Picks are frozen once a game starts, using the last run before it.</p>")


def main(season=None):
    season = season or current_season()
    ps, ctx, off = load_data(season)
    a = allowed_history(ps, ctx)
    week, unplayed = next_week(ctx, season)
    status = injury_status(season, week)
    df = build_features(ps, ctx, a, season, week, unplayed, status)

    cand, note = td_model(df, season)
    notes = [note]
    allc = cand.assign(status=cand["player_id"].map(status).fillna("") if status else "")
    pool = prep(cand, status, "score")
    pool["rz_rank"] = pool["rz_share_l4"].rank(ascending=False, method="min")
    for c in ["rz_share_l4", "touches_l4", "gl_carries_l4", "tgt_share_l4", "td_l12", "opp_td"]:
        pool["pct_" + c] = pool.groupby("position")[c].rank(pct=True, method="min")
    pool["rank_all"] = np.arange(1, len(pool) + 1)
    boosted = np.where(pool["position"] == "RB", pool["vac_carry"], np.where(pool["position"].isin(["WR", "TE"]), pool["vac_tgt"], 0)) >= 0.15
    pool["signals"] = ((pool["rz_rank"] <= 10).astype(int) + (pool["gl_carries_l4"] >= 1.0) + (pool["spread"] >= 3)
                       + (pool["opp_td"] >= 1.25) + boosted).astype(int)
    td = pool.head(TOP_TD)
    rest = pool.iloc[TOP_TD:]
    long_td = rest[(rest["signals"] >= 1) & (rest["prob"] >= 0.04)].sort_values(["signals", "prob"], ascending=False).head(TOP_LONG)
    td_section = ("td", "Touchdowns", "".join(td_card(i, r) for i, r in enumerate(td.itertuples(), 1)))
    picks = [pick_rows(season, week, "td", td, "prob", "td_l12"), pick_rows(season, week, "td_long", long_td, "prob", "td_l12")]
    long_html = ("<h3>Touchdown longshots</h3><p class='script'>Players outside the main list, ranked by how many positive signals they have "
                 "(red zone share, goal line work, favored team, soft matchup, teammate out), then by model chance. The model can't see sportsbook odds, "
                 "so use the price shown on each card: a longshot only has value if the book pays at least that much. Books usually keep a bigger margin "
                 "on longshots, so real value here is rarer, and the model is least reliable for low-probability players.</p>"
                 + "".join(td_card(r.rank_all, r) for r in long_td.itertuples()))

    propc, prop_sections = {}, []
    for key, (title, *_rest) in PROPS.items():
        pc, pnote = prop_model(df, season, key)
        propc[key] = prep(pc, status, "score")
        propc[key]["rank_all"] = np.arange(1, len(propc[key]) + 1)
        top = propc[key].head(TOP_PROP)
        extra = propc[key].iloc[TOP_PROP:]
        lift = extra["proj"] / extra[f"{key}_l12"].replace(0, np.nan)
        up = extra.assign(lift=lift)[lift >= 1.15].sort_values("lift", ascending=False).head(6)
        if len(up):
            long_html += (f"<h3>{title}: trending up</h3><p class='script'>Projected at least 15% above their last-12-game average. "
                          f"Lines are often set off recent form, so compare the book's line with the projection.</p>"
                          + "".join(prop_card(r.rank_all, r, key) for r in up.itertuples()))
        notes.append(pnote)
        prop_sections.append((key, title, "".join(prop_card(i, r, key) for i, r in enumerate(top.itertuples(), 1))))
        picks.append(pick_rows(season, week, key, top, "proj", f"{key}_l4"))

    hist = update_history(pd.concat(picks, ignore_index=True), season, set(unplayed["team"]), ps, ctx)
    games = ("games", "Games", game_cards(unplayed, team_tables(ctx, off, a), pool, allc, propc))
    sections = [td_section, ("long", "Longshots", long_html), games] + prop_sections + [("record", "Track record", record_html(hist))]
    write_outputs(sections, td, week, season, notes, status is not None)
    print(f"\nDone. Week {week} picks saved to {OUT_DIR}/index.html. Open that file in your browser.")


if __name__ == "__main__":
    main()
