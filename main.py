# part_1_2_main_line_and_connections_gurobi.py
# Part 1: ILP path (within-area arcs + Achilles links)
# Part 2: LP connections from non-path cities to hubs (Warszawa/Kielce/Opole)

from math import pi
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# ---------- Load city data ----------
df = pd.read_csv("cities.csv")
df["demand"] = df["demand"].apply(lambda x: None if pd.isna(x) else int(x))

# list of tuples (name, x, y, area, demand)
CITIES = list(df.itertuples(index=False, name=None))

IDX = {name: i for i, (name, *_rest) in enumerate(CITIES)}
n = len(CITIES)

# Distances (all pairs, we’ll filter for allowed arcs in Part 1)
DIST_ALL = {}
for i, (ni, xi, yi, ai, di) in enumerate(CITIES):
    for j, (nj, xj, yj, aj, dj) in enumerate(CITIES):
        if i != j:
            DIST_ALL[(i, j)] = ((xi - xj)**2 + (yi - yj)**2) ** 0.5

# Areas
V1 = [i for i, (_, _, _, a, _) in enumerate(CITIES) if a == 1]
V2 = [i for i, (_, _, _, a, _) in enumerate(CITIES) if a == 2]
V3 = [i for i, (_, _, _, a, _) in enumerate(CITIES) if a == 3]

# Key indices
WAR, RAD, KIE, GLI, OPO, JEL = (
    IDX["Warszawa"], IDX["Radom"], IDX["Kielce"],
    IDX["Gliwice"], IDX["Opole"], IDX["Jelenia Góra"]
)

# Policy sets
KON, KAL = IDX["Konin"], IDX["Kalisz"]
BYT, SOS, KAT = IDX["Bytom"], IDX["Sosnowiec"], IDX["Katowice"]
WOD, JAS = IDX["Wodzisław Śl."], IDX["Jastrzębie-Zdrój"]
POZ, ZIE, LES = IDX["Poznań"], IDX["Zielona Góra"], IDX["Leszno"]

def build_allowed_arcs():
    """Part 1: only within-area arcs + Achilles links (both directions)."""
    E = []
    DIST = {}
    for (i, j), d in DIST_ALL.items():
        ai = CITIES[i][3]; aj = CITIES[j][3]
        same_area = (ai == aj)
        achilles = (i, j) in [(RAD, KIE), (KIE, RAD), (GLI, OPO), (OPO, GLI)]
        if same_area or achilles:
            E.append((i, j))
            DIST[(i, j)] = d
    return E, DIST

def solve_part1(verbose: bool = False):
    """Optimal main line path from Warszawa to Jelenia Góra (Part 1)."""
    E, DIST = build_allowed_arcs()

    # adjacency
    in_arcs = {j: [] for j in range(n)}
    out_arcs = {i: [] for i in range(n)}
    for (i, j) in E:
        out_arcs[i].append(j)
        in_arcs[j].append(i)

    m = gp.Model("main_line")
    m.Params.OutputFlag = 1 if verbose else 0

    x = m.addVars(E, vtype=GRB.BINARY, name="x")
    y = m.addVars(range(n), vtype=GRB.BINARY, name="y")

    m.setObjective(gp.quicksum(DIST[i, j] * x[i, j] for (i, j) in E), GRB.MINIMIZE)

    for j in range(n):
        if j == WAR:
            m.addConstr(gp.quicksum(x[i, j] for i in in_arcs[j]) == 0)
            m.addConstr(gp.quicksum(x[j, k] for k in out_arcs[j]) == 1)
            m.addConstr(y[j] == 1)
        elif j == JEL:
            m.addConstr(gp.quicksum(x[i, j] for i in in_arcs[j]) == 1)
            m.addConstr(gp.quicksum(x[j, k] for k in out_arcs[j]) == 0)
            m.addConstr(y[j] == 1)
        else:
            m.addConstr(gp.quicksum(x[i, j] for i in in_arcs[j]) == y[j])
            m.addConstr(gp.quicksum(x[j, k] for k in out_arcs[j]) == y[j])

    # don't allow opposite arcs together
    for (i, j) in E:
        if (j, i) in E:
            m.addConstr(x[i, j] + x[j, i] <= 1)

    # inclusion counts
    m.addConstr(gp.quicksum(y[i] for i in V1 if i not in {WAR, RAD}) >= 2)
    m.addConstr(gp.quicksum(y[i] for i in V2 if i not in {KIE, GLI}) >= 3)
    m.addConstr(gp.quicksum(y[i] for i in V3 if i not in {OPO, JEL}) >= 3)

    # anchors present
    for k in [RAD, KIE, GLI, OPO]:
        m.addConstr(y[k] == 1)

    # policy constraints
    m.addConstr(y[KON] + y[KAL] >= 1)
    m.addConstr(y[BYT] + y[SOS] + y[KAT] <= 1)
    m.addConstr(y[WOD] + y[JAS] <= 1)
    m.addConstr(y[POZ] + y[ZIE] + y[LES] >= 1)

    # Achilles edges (exactly one direction each)
    m.addConstr(x[RAD, KIE] + x[KIE, RAD] == 1)
    m.addConstr(x[GLI, OPO] + x[OPO, GLI] == 1)

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status: {m.Status}")

    X = {(i, j): int(round(x[i, j].X)) for (i, j) in E}
    total_len = m.ObjVal

    out_used = defaultdict(list)
    for (i, j), v in X.items():
        if v == 1:
            out_used[i].append(j)

    path_idx = [WAR]
    cur = WAR
    seen = {WAR}
    while cur != JEL:
        nxts = out_used.get(cur, [])
        if len(nxts) != 1:
            raise RuntimeError(f"{CITIES[cur][0]} has {len(nxts)} outgoing used arcs; expected 1.")
        cur = nxts[0]
        if cur in seen:
            raise RuntimeError("Cycle detected during path reconstruction.")
        seen.add(cur)
        path_idx.append(cur)

    path_names = [CITIES[i][0] for i in path_idx]
    return total_len, path_names, set(path_idx)

# ---------- Part 2 (LP): connect remaining cities to hubs ----------

HUBS = ["Warszawa", "Kielce", "Opole"]
HUB_CAP = {"Warszawa": 800, "Kielce": 1200, "Opole": 400}

def distance(i, j):
    (ni, xi, yi, _, _), (nj, xj, yj, _, _) = CITIES[i], CITIES[j]
    return ((xi - xj)**2 + (yi - yj)**2) ** 0.5

def max_cable_capacity(dist_ij):
    """U = min(200, max(0, 2*pi*min(35, 70-d)))"""
    atten = min(35, 70 - dist_ij)
    atten = max(0.0, atten)
    cap = 2 * pi * atten
    return min(200.0, cap)

def solve_part2(included_indices: set, verbose: bool = False):
    """LP: connect each city not on the main line to at least one hub, minimum cost."""
    # Remaining cities R (exclude hubs with demand None and any on main line)
    hub_idx = {h: IDX[h] for h in HUBS}
    R = []
    for i, (name, x, y, area, demand) in enumerate(CITIES):
        if i in included_indices:
            continue
        if demand is None:  # hubs have None demand
            continue
        R.append(i)

    if not R:
        return 0.0, {}, 0.0

    # Precompute distances and per-cable upper bounds U_{i,s}
    dist_is = {}
    U = {}
    for i in R:
        for h in HUBS:
            s = hub_idx[h]
            d = distance(i, s)
            dist_is[(i, h)] = d
            U[(i, h)] = max_cable_capacity(d)

    m = gp.Model("connect_remaining_cities")
    m.Params.OutputFlag = 1 if verbose else 0

    # Decision variables f_{i,h} >= 0
    f = m.addVars([(i, h) for i in R for h in HUBS], lb=0.0, name="f")

    # Objective: min sum 1000 * distance * flow
    m.setObjective(gp.quicksum(1000.0 * dist_is[(i, h)] * f[i, h] for i in R for h in HUBS), GRB.MINIMIZE)

    # Demand satisfaction
    for i in R:
        Di = CITIES[i][4]
        m.addConstr(gp.quicksum(f[i, h] for h in HUBS) == Di)

    # Per-cable upper bounds (200 TB/s and attenuation)
    for i in R:
        for h in HUBS:
            m.addConstr(f[i, h] <= U[(i, h)])

    # Station capacities
    for h in HUBS:
        m.addConstr(gp.quicksum(f[i, h] for i in R) <= HUB_CAP[h])

    # Łódź max 40% per cable (only if Łódź in R)
    if "Łódź" in IDX and IDX["Łódź"] in R:
        Dlodz = CITIES[IDX["Łódź"]][4]
        for h in HUBS:
            m.addConstr(f[IDX["Łódź"], h] <= 0.4 * Dlodz)

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status (Part 2): {m.Status}")

    # Outputs
    total_cost = m.ObjVal
    flows = {(CITIES[i][0], h): f[i, h].X for i in R for h in HUBS}

    # Specific questions:
    poz = IDX["Poznań"]
    wal = IDX["Wałbrzych"]
    poz_flows = {h: flows.get((CITIES[poz][0], h), 0.0) for h in HUBS}
    wal_flows = {h: flows.get((CITIES[wal][0], h), 0.0) for h in HUBS}

    kielce_util = sum(flows.get((CITIES[i][0], "Kielce"), 0.0) for i in R)

    return total_cost, {"Poznań": poz_flows, "Wałbrzych": wal_flows}, kielce_util

# ---------- Run both parts ----------
if __name__ == "__main__":
    length, path, included_idx = solve_part1(verbose=False)
    print(f"Optimal main line length: {length:.2f} coordinate units")
    print("Main line path (in order): " + " → ".join(path))

    # Print excluded cities
    included_names = set(path)
    excluded = [name for (name, *_rest) in CITIES if name not in included_names and name not in HUBS]
    print("\nCities not included in the main line:")
    for name in sorted(excluded):
        print(f"- {name}")

    # Part 2
    total_cost, city_flows, kielce_util = solve_part2(included_idx, verbose=False)
    print("\n=== Part 2 results ===")
    print(f"Total connection cost (EUR): {total_cost:,.2f}")

    # Flows for Poznań and Wałbrzych
    print("\nFlows for Poznań (TB/s):")
    for h, val in city_flows["Poznań"].items():
        print(f"  Poznań → {h}: {val:.2f}")

    print("\nFlows for Wałbrzych (TB/s):")
    for h, val in city_flows["Wałbrzych"].items():
        print(f"  Wałbrzych → {h}: {val:.2f}")

    print(f"\nKielce station utilization (TB/s): {kielce_util:.2f} / {HUB_CAP['Kielce']}")
