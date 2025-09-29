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
    '''LP: connect each city not on the main line to at least one hub, minimum cost.'''
    hub_idx = {h: IDX[h] for h in HUBS}
    # Collect demand cities not already served by the main line or hubs.
    R = []
    for i, (name, x, y, area, demand) in enumerate(CITIES):
        if i in included_indices:
            continue
        if demand is None:
            continue
        R.append(i)

    # No remaining demand nodes -> nothing to connect.
    if not R:
        return 0.0, {}, 0.0, [], None

    # Precompute geometry and attenuation terms once.
    dist_is = {}
    atten_caps = {}
    atten_values = {}
    for i in R:
        for h in HUBS:
            s = hub_idx[h]
            d = distance(i, s)
            dist_is[(i, h)] = d
            atten = min(35, 70 - d)
            atten = max(0.0, atten)
            atten_caps[(i, h)] = 2 * pi * atten
            atten_values[(i, h)] = atten

    m = gp.Model("connect_remaining_cities")
    m.Params.OutputFlag = 1 if verbose else 0

    # Flow decision variables f[i, h] represent capacity installed on city-hub cables.
    f = m.addVars([(i, h) for i in R for h in HUBS], lb=0.0, name="f")

    # Minimize total installation cost (distance-weighted capacity).
    m.setObjective(
        gp.quicksum(1000.0 * dist_is[(i, h)] * f[i, h] for i in R for h in HUBS),
        GRB.MINIMIZE,
    )

    # Each city must ship its entire demand to the hubs.
    demand_constrs = {}
    for i in R:
        Di = CITIES[i][4]
        constr = m.addConstr(gp.quicksum(f[i, h] for h in HUBS) == Di)
        demand_constrs[i] = constr

    # Cable-specific capacity: attenuation limit and absolute 200 TB/s cap.
    atten_cap_constrs = {}
    for i in R:
        for h in HUBS:
            cap = atten_caps[(i, h)]
            if cap < 200.0:
                constr = m.addConstr(f[i, h] <= cap)
                atten_cap_constrs[(i, h)] = constr
            else:
                m.addConstr(f[i, h] <= 200.0)

    # Switching-station capacity at each hub.
    for h in HUBS:
        m.addConstr(gp.quicksum(f[i, h] for i in R) <= HUB_CAP[h])

    # Lodz policy: any single cable can carry at most 40% of the city's demand.
    if "Łódź" in IDX and IDX["Łódź"] in R:
        lodz_idx = IDX["Łódź"]
        Dlodz = CITIES[lodz_idx][4]
        for h in HUBS:
            m.addConstr(f[lodz_idx, h] <= 0.4 * Dlodz)

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status (Part 2): {m.Status}")

    # Read out primal solution and derived summaries.
    total_cost = m.ObjVal
    flows = {(CITIES[i][0], h): f[i, h].X for i in R for h in HUBS}

    # Repackage flows by city for easier reporting.
    city_flows = {}
    for i in R:
        city_name = CITIES[i][0]
        city_flows[city_name] = {h: flows.get((city_name, h), 0.0) for h in HUBS}

    # Total flow into Kielce for utilization statistics.
    kielce_util = sum(flows.get((CITIES[i][0], "Kielce"), 0.0) for i in R)

    pi_rhs_ranges = []
    # Store attenuation-limited RHS data so we can back out the admissible pi interval later.
    for (i, h), constr in atten_cap_constrs.items():
        atten = atten_values[(i, h)]
        pi_rhs_ranges.append(
            {
                "city_idx": i,
                "city_name": CITIES[i][0],
                "hub": h,
                "atten": atten,
                "rhs": constr.RHS,
                "rhs_low": constr.SARHSLow,
                "rhs_up": constr.SARHSUp,
            }
        )

    # Capture Konin demand dual information for sensitivity analysis.
    konin_info = None
    if "Konin" in IDX and IDX["Konin"] in R:
        konin_idx = IDX["Konin"]
        constr = demand_constrs[konin_idx]
        konin_info = {
            "demand": CITIES[konin_idx][4],
            "dual": constr.Pi,
            "rhs_low": constr.SARHSLow,
            "rhs_up": constr.SARHSUp,
        }

    return total_cost, city_flows, kielce_util, pi_rhs_ranges, konin_info


def analyze_konin_sensitivity(konin_info):
    '''Analyze Konin demand sensitivity.'''
    if not konin_info:
        print("   Konin is not part of the connection LP; no sensitivity information available.")
        return

    konin_demand = konin_info["demand"]
    konin_dual = konin_info["dual"]
    rhs_low = konin_info["rhs_low"]
    rhs_up = konin_info["rhs_up"]

    print(f"   Konin current demand: {konin_demand} TB/s")
    print(f"   Konin shadow price: {konin_dual:.4f} EUR per TB/s")

    # Gurobi's SARHSLow/Up give the RHS window where the dual remains valid.
    def within_bounds(value):
        low_ok = True if rhs_low == -GRB.INFINITY else value >= rhs_low
        up_ok = True if rhs_up == GRB.INFINITY else value <= rhs_up
        return low_ok and up_ok

    for pct in [10, 20]:
        demand_change = konin_demand * (pct / 100.0)
        lower = konin_demand - demand_change
        upper = konin_demand + demand_change
        in_range = within_bounds(lower) and within_bounds(upper)
        cost_change = konin_dual * demand_change
        note = "" if in_range else " (outside valid sensitivity range)"
        print(
            f"   +/-{pct}% demand change (+/-{demand_change:.1f} TB/s): "
            f"Total cost shifts by +/-{abs(cost_change):,.2f} EUR{note}"
        )


def analyze_pi_interval(pi_rhs_ranges):
    '''Calculate pi interval for current basis optimality.'''
    if not pi_rhs_ranges:
        print("   No pi-dependent capacity constraints were active; pi can vary freely (pi > 0).")
        return

    pi_lower = 0.0
    pi_upper = float("inf")

    for info in pi_rhs_ranges:
        atten = info["atten"]
        if atten <= 0:
            continue
        # Each binding constraint is f <= 2*pi*atten, so divide the allowed RHS range by 2*atten.
        denom = 2.0 * atten
        rhs_low = info["rhs_low"]
        rhs_up = info["rhs_up"]

        if rhs_low != -GRB.INFINITY and denom > 0:
            candidate = rhs_low / denom
            if candidate > 0:
                pi_lower = max(pi_lower, candidate)
        if rhs_up != GRB.INFINITY and denom > 0:
            candidate = rhs_up / denom
            pi_upper = min(pi_upper, candidate)

    print(f"   Current pi: {pi:.6f}")
    print(f"   pi can vary within [{pi_lower:.6f}, {pi_upper:.6f}] without changing the basis.")


if __name__ == "__main__":
    print("=" * 60)
    print("PART 1: MAIN LINE DESIGN")
    print("=" * 60)

    length, path, included_idx = solve_part1(verbose=False)
    print(f"Optimal main line length: {length:.2f} coordinate units")
    print("Main line path (in order): " + " -> ".join(path))

    # Print excluded cities
    included_names = set(path)
    excluded = [name for (name, *_rest) in CITIES if name not in included_names and name not in HUBS]
    print("\nCities not included in the main line:")
    for name in sorted(excluded):
        print(f"- {name}")

    # Part 2
    total_cost, city_flows, kielce_util, pi_rhs_ranges, konin_info = solve_part2(included_idx, verbose=False)
    print("\n" + "="*60)
    print("PART 2: CONNECTION OF REMAINING CITIES TO SWITCHING STATIONS")
    print("="*60)
    
    print(f"\n1. TOTAL CONNECTION COSTS")
    print(f"   Total cost to connect all remaining cities: {total_cost:,.2f} EUR")
    
    # Individual city connection costs
    print(f"\n   Breakdown by city:")
    hub_idx = {h: IDX[h] for h in HUBS}
    R = []
    for i, (name, x, y, area, demand) in enumerate(CITIES):
        if i in included_idx or demand is None:
            continue
        R.append(i)
    
    for i in R:
        city_name = CITIES[i][0]
        city_cost = 0.0
        for h in HUBS:
            flow_val = city_flows.get(city_name, {}).get(h, 0.0)
            if flow_val > 0:
                dist = distance(i, hub_idx[h])
                city_cost += 1000.0 * dist * flow_val
        print(f"   - {city_name}: {city_cost:,.2f} EUR")

    print(f"\n2. CAPACITY INSTALLATIONS")
    print(f"   Required capacities between cities and switching stations:")
    
    print(f"\n   Poznań connections:")
    for h, val in city_flows["Poznań"].items():
        if val > 0:
            print(f"   - Poznań -> {h}: {val:.2f} TB/s")
        else:
            print(f"   - Poznań -> {h}: 0.00 TB/s (no connection)")

    print(f"\n   Wałbrzych connections:")
    for h, val in city_flows["Wałbrzych"].items():
        if val > 0:
            print(f"   - Wałbrzych -> {h}: {val:.2f} TB/s")
        else:
            print(f"   - Wałbrzych -> {h}: 0.00 TB/s (no connection)")

    print(f"\n3. SWITCHING STATION UTILIZATION")
    kielce_capacity = HUB_CAP['Kielce']
    kielce_percentage = (kielce_util / kielce_capacity) * 100
    print(f"   Kielce switching station:")
    print(f"   - Utilized capacity: {kielce_util:.2f} TB/s")
    print(f"   - Total capacity: {kielce_capacity} TB/s")
    print(f"   - Utilization rate: {kielce_percentage:.1f}%")
    
    # Show utilization for all hubs for completeness
    for h in HUBS:
        if h != "Kielce":
            hub_util = sum(city_flows.get(CITIES[i][0], {}).get(h, 0.0) for i in R)
            hub_capacity = HUB_CAP[h]
            hub_percentage = (hub_util / hub_capacity) * 100
            print(f"   {h} switching station:")
            print(f"   - Utilized capacity: {hub_util:.2f} TB/s")
            print(f"   - Total capacity: {hub_capacity} TB/s")
            print(f"   - Utilization rate: {hub_percentage:.1f}%")

    # Add sensitivity analysis section
    print(f"\n4. SENSITIVITY ANALYSIS")
    print(f"\n   Konin Demand Sensitivity:")
    analyze_konin_sensitivity(konin_info)
    
    print(f"\n   pi Parameter Sensitivity:")
    analyze_pi_interval(pi_rhs_ranges)
 
