# part_1_main_line_gurobi_fixed.py
# Gurobi-only ILP for Part 1 (a–c), with within-area arcs + Achilles links only,
# and adjacency-aware constraints to avoid KeyError.

from math import hypot, sqrt
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

df = pd.read_csv("cities.csv")
df["demand"] = df["demand"].apply(lambda x: None if pd.isna(x) else int(x))

CITIES = list(df.itertuples(index=False, name=None))

IDX = {name: i for i, (name, *_rest) in enumerate(CITIES)}
n = len(CITIES)

# Cartesian distances for all i != j (we'll later filter to allowed arcs)
DIST_ALL = {}
for i, (ni, xi, yi, ai, di) in enumerate(CITIES):
    for j, (nj, xj, yj, aj, dj) in enumerate(CITIES):
        if i != j:
            DIST_ALL[(i, j)] = hypot(xi - xj, yi - yj)

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
    """Allow only within-area arcs + two area connection links (Radom->Kielce and Gliwice->Opole)."""
    E = []
    DIST = {}
    for (i, j), d in DIST_ALL.items():
        ai = CITIES[i][3]; aj = CITIES[j][3]
        same_area = (ai == aj)
        is_area_connection = (i, j) in [(RAD, KIE), (KIE, RAD), (GLI, OPO), (OPO, GLI)]
        if same_area or is_area_connection:
            E.append((i, j))
            DIST[(i, j)] = d
    return E, DIST

def solve_main_line_gurobi(verbose: bool = False):
    E, DIST = build_allowed_arcs()

    # Build adjacency to avoid KeyError: only sum over existing arcs
    in_arcs = {j: [] for j in range(n)}
    out_arcs = {i: [] for i in range(n)}
    for (i, j) in E:
        out_arcs[i].append(j)
        in_arcs[j].append(i)

    m = gp.Model("main_line")
    m.Params.OutputFlag = 1 if verbose else 0

    # Decision variables only on allowed arcs
    x = m.addVars(E, vtype=GRB.BINARY, name="x")
    y = m.addVars(range(n), vtype=GRB.BINARY, name="y")

    # Objective: total distance
    m.setObjective(gp.quicksum(DIST[i, j] * x[i, j] for (i, j) in E), GRB.MINIMIZE)

    # Degree/flow constraints using adjacency lists
    for j in range(n):
        if j == WAR:
            m.addConstr(gp.quicksum(x[i, j] for i in in_arcs[j]) == 0)  # no incoming
            m.addConstr(gp.quicksum(x[j, k] for k in out_arcs[j]) == 1) # exactly 1 outgoing
            m.addConstr(y[j] == 1)
        elif j == JEL:
            m.addConstr(gp.quicksum(x[i, j] for i in in_arcs[j]) == 1)  # exactly 1 incoming
            m.addConstr(gp.quicksum(x[j, k] for k in out_arcs[j]) == 0) # no outgoing
            m.addConstr(y[j] == 1)
        else:
            m.addConstr(gp.quicksum(x[i, j] for i in in_arcs[j]) == y[j])
            m.addConstr(gp.quicksum(x[j, k] for k in out_arcs[j]) == y[j])

    # No opposite arcs simultaneously (only if both directions exist)
    for (i, j) in E:
        if (j, i) in E:
            m.addConstr(x[i, j] + x[j, i] <= 1)

    # Area inclusion counts (exclude anchors in each area)
    m.addConstr(gp.quicksum(y[i] for i in V1 if i not in {WAR, RAD}) >= 2)
    m.addConstr(gp.quicksum(y[i] for i in V2 if i not in {KIE, GLI}) >= 3)
    m.addConstr(gp.quicksum(y[i] for i in V3 if i not in {OPO, JEL}) >= 3)

    # Force anchor presence
    for k in [RAD, KIE, GLI, OPO]:
      m.addConstr(y[k] == 1)

    # Policy constraints
    m.addConstr(y[KON] + y[KAL] >= 1)
    m.addConstr(y[BYT] + y[SOS] + y[KAT] <= 1)
    m.addConstr(y[WOD] + y[JAS] <= 1)
    m.addConstr(y[POZ] + y[ZIE] + y[LES] >= 1)

    # Achilles edges must be used (exactly one direction)
    # These arcs are guaranteed to exist because we added them explicitly.
    m.addConstr(x[RAD, KIE] + x[KIE, RAD] == 1)
    m.addConstr(x[GLI, OPO] + x[OPO, GLI] == 1)

    # Optimize
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status: {m.Status}")

    # Extract solution and reconstruct the path
    X = {(i, j): int(round(x[i, j].X)) for (i, j) in E}
    total_len = m.ObjVal

    out_used = defaultdict(list)
    for (i, j), v in X.items():
        if v == 1:
            out_used[i].append(j)

    # Follow path from WAR to JEL
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
    return total_len, path_names

if __name__ == "__main__":
    length, path = solve_main_line_gurobi(verbose=False)
    print(f"Optimal main line length: {length:.2f} coordinate units")
    print("Main line path (in order): " + " → ".join(path))
