"""College football picks. Runs on GitHub Actions and writes cfb/data.json + cfb/history.json."""
import os, json, math, datetime as dt, requests

KEY = os.environ["CFBD_API_KEY"]
OUT = "cfb"
NOW = dt.datetime.now(dt.timezone.utc)
YEAR = NOW.year if NOW.month >= 8 else NOW.year - 1
HFA, SD, SHRINK, CAP = 2.5, 13.5, 0.35, 28   # home edge, game spread of outcomes, trust-in-model, blowout cap

def api(path, **p):
    r = requests.get("https://api.collegefootballdata.com" + path, params=p,
                     headers={"Authorization": "Bearer " + KEY}, timeout=90)
    r.raise_for_status()
    return r.json()

def load(f, d):
    try:
        with open(f"{OUT}/{f}") as fh: return json.load(fh)
    except Exception: return d

def save(f, o):
    with open(f"{OUT}/{f}", "w") as fh: json.dump(o, fh, indent=1)

def cdf(x): return 0.5 * (1 + math.erf(x / (SD * math.sqrt(2))))
def implied(ml): return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)
def payout(ml): return ml / 100 if ml > 0 else 100 / -ml
def fair(p): return round(-100 * p / (1 - p)) if p >= .5 else round(100 * (1 - p) / p)
def amer(n): return f"+{n}" if n > 0 else str(n)
def fbs(g): return all(g.get(k) in (None, "fbs") for k in ("homeClassification", "awayClassification"))
def started(g): return dt.datetime.fromisoformat(g["startDate"].replace("Z", "+00:00")) <= NOW

def ratings(games):
    """Opponent-adjusted scoring margin (higher = better team)."""
    done = [g for g in games if g.get("completed") and g.get("homePoints") is not None and g.get("awayPoints") is not None]
    r = {}
    for _ in range(25):
        tot, n = {}, {}
        for g in done:
            h, a = g["homeTeam"], g["awayTeam"]
            m = g["homePoints"] - g["awayPoints"] - (0 if g.get("neutralSite") else HFA)
            m = max(-CAP, min(CAP, m))
            tot[h] = tot.get(h, 0) + m + r.get(a, 0); n[h] = n.get(h, 0) + 1
            tot[a] = tot.get(a, 0) - m + r.get(h, 0); n[a] = n.get(a, 0) + 1
        r = {t: tot[t] / n[t] for t in tot}
    return r

def td_table():
    by = {}
    for cat in ("rushing", "receiving"):
        for x in api("/stats/player/season", year=YEAR, category=cat, seasonType="regular"):
            if x.get("statType") == "TD":
                k = (x["player"], x["team"]); by[k] = by.get(k, 0) + float(x["stat"])
    teams = {}
    for (p, t), v in by.items(): teams.setdefault(t, []).append((p, v))
    return teams

def scorers(week):
    s = set()
    for cat in ("rushing", "receiving"):
        for g in api("/games/players", year=YEAR, week=week, seasonType="regular", category=cat):
            for t in g.get("teams", []):
                for c in t.get("categories", []):
                    for ty in c.get("types", []):
                        if ty.get("name") == "TD":
                            for at in ty.get("athletes", []):
                                if float(at.get("stat") or 0) > 0: s.add((g["id"], at["name"]))
    return s

def grade(hist, games):
    G = {g["id"]: g for g in games if g.get("completed") and g.get("homePoints") is not None}
    sc = {}
    for p in hist:
        g = G.get(p["gid"])
        if p.get("result") or not g: continue
        h, a = g["homePoints"], g["awayPoints"]
        mine, opp = (h, a) if p["team"] == g["homeTeam"] else (a, h)
        if p["type"] == "spread":
            m = mine - opp + p["line"]
            p["result"] = "W" if m > 0 else "L" if m < 0 else "P"
        elif p["type"] == "ml":
            p["result"] = "W" if mine > opp else "L"
        else:
            try:
                if p["week"] not in sc: sc[p["week"]] = scorers(p["week"])
                p["result"] = "W" if (p["gid"], p["pick"]) in sc[p["week"]] else "L"
            except Exception as e:
                print("TD grading skipped:", e)

def main():
    os.makedirs(OUT, exist_ok=True)
    games = [g for g in api("/games", year=YEAR, seasonType="regular") if fbs(g)]
    hist = load("history.json", [])
    grade(hist, games)
    up = [g for g in games if not g.get("completed") and not started(g)]
    if not up:
        save("history.json", hist); print("No upcoming games."); return
    week = min(g["week"] for g in up)
    up = [g for g in up if g["week"] == week]
    R = ratings(games)
    lines = {l["id"]: l for l in api("/lines", year=YEAR, week=week, seasonType="regular")}
    teams = td_table()
    outs = set()
    if os.path.exists(f"{OUT}/out.txt"):
        outs = {x.strip().lower() for x in open(f"{OUT}/out.txt") if x.strip() and not x.startswith("#")}

    sp_l, ml_l, td_l, cheats = [], [], [], []
    for g in up:
        L = next((l for l in lines.get(g["id"], {}).get("lines", []) if l.get("spread") is not None), None)
        h, a = g["homeTeam"], g["awayTeam"]
        if not L or h not in R or a not in R: continue
        sp, tot = float(L["spread"]), L.get("overUnder")
        tot = float(tot) if tot else None
        mkt = -sp                                   # market's home margin
        mdl = R[h] - R[a] + (0 if g.get("neutralSite") else HFA)
        fin = mkt + SHRINK * (mdl - mkt)            # model, pulled most of the way back to the market
        name = f"{a} @ {h}"

        edge = fin - mkt
        side, tl = (h, sp) if edge > 0 else (a, -sp)
        sp_l.append(dict(game=name, pick=f"{side} {tl:+g}", edge=round(abs(edge), 1), start=g["startDate"], gid=g["id"], team=side, line=tl))

        hm, am = L.get("homeMoneyline"), L.get("awayMoneyline")
        if hm and am:
            ph, best = cdf(fin), None
            for team, ml, p in ((h, hm, ph), (a, am, 1 - ph)):
                ev = p * payout(ml) - (1 - p)
                if ml > -400 and (best is None or ev > best[0]): best = (ev, team, ml, p)
            if best and best[0] > 0:
                ev, team, ml, p = best
                ml_l.append(dict(game=name, pick=f"{team} {amer(ml)}", ev=round(ev * 100, 1), p=round(p * 100), start=g["startDate"], gid=g["id"], team=team, ml=ml))

        codes = []
        if tot and tot >= 62: codes.append("Shootout total: both offenses get a TD bump.")
        if tot and tot <= 45: codes.append("Grinder: points are scarce, rushing TDs beat receiving TDs.")
        if abs(sp) >= 21: codes.append("Blowout risk: backups get garbage-time carries, and big spreads are coin flips.")
        if abs(sp) <= 3: codes.append("Toss-up: the moneyline price matters more than the side.")
        if abs(mdl - mkt) >= 6: codes.append(f"Model and market split by {abs(mdl - mkt):.0f} pts: trust the market first.")
        if codes: cheats.append(dict(game=name, start=g["startDate"], line=L.get("formattedSpread"), total=tot, codes=codes))

        if tot:
            for team, pts in ((h, (tot + fin) / 2), (a, (tot - fin) / 2)):
                pl = [(p, v) for p, v in teams.get(team, []) if p.lower() not in outs]
                pool = sum(v for _, v in pl)
                if not pool: continue
                lam = 0.8 * pts / 7                 # expected rushing+receiving TDs for the team
                for p, v in sorted(pl, key=lambda x: -x[1])[:3]:
                    if v < 2: continue
                    pr = 1 - math.exp(-lam * v / pool)
                    td_l.append(dict(player=p, team=team, game=name, p=round(pr * 100), fair=amer(fair(pr)), gid=g["id"]))

    sp_l = sorted([x for x in sp_l if x["edge"] >= 1.5], key=lambda x: -x["edge"])[:8]
    ml_l = sorted(ml_l, key=lambda x: -x["ev"])[:5]
    td_l = sorted(td_l, key=lambda x: -x["p"])[:15]

    have = {p["k"] for p in hist}
    for typ, items in (("spread", sp_l[:5]), ("ml", ml_l[:3]), ("td", td_l[:10])):
        for x in items:
            pick = x["player"] if typ == "td" else x["pick"]
            k = f"{week}|{typ}|{x['gid']}" + (f"|{pick}" if typ == "td" else "")
            if k not in have:
                hist.append(dict(k=k, week=week, type=typ, gid=x["gid"], pick=pick, team=x["team"], line=x.get("line"), ml=x.get("ml"), result=None))

    save("data.json", dict(updated=NOW.isoformat(timespec="minutes"), week=week, spread=sp_l, ml=ml_l, td=td_l, cheats=cheats))
    save("history.json", hist)
    print(f"Week {week}: {len(sp_l)} spread, {len(ml_l)} moneyline, {len(td_l)} TD picks.")

if __name__ == "__main__":
    main()
