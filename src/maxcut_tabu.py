import random
import time
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError:
    nx = None

# Graph representation
Edge = Tuple[int, int, int]      # (u, v, weight)
Adj = Dict[int, List[Tuple[int, int]]]  # adjacency list

# Random graph generation
def generate_random_graph(n: int, p: float, w_max: int, seed: int = 0):
    rng = random.Random(seed)
    edges: List[Edge] = []
    adj: Adj = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w = rng.randint(1, w_max)
                edges.append((i, j, w))
                adj[i].append((j, w))
                adj[j].append((i, w))
    return edges, adj

# Encoding and fitness
def random_solution(n: int, rng: random.Random):
    return [rng.randint(0, 1) for _ in range(n)]


def cut_value(edges: List[Edge], x: List[int]) -> int:
    total = 0
    for u, v, w in edges:
        if x[u] != x[v]:
            total += w
    return total


def delta_flip(adj: Adj, x: List[int], k: int) -> int:
    """Efficient delta computation for flipping vertex k: O(deg(k))."""
    old = x[k]
    new = 1 - old
    delta = 0
    for j, w in adj[k]:
        before = 1 if old != x[j] else 0
        after = 1 if new != x[j] else 0
        delta += w * (after - before)
    return delta

# Visualisation helpers
def plot_curves(curves: Dict[str, List[int]], title: str, filename: str, hc_limit: int = 200):
    plt.figure()
    for name, vals in curves.items():
        if "HillClimbing" in name:
            plt.plot(vals[:hc_limit], label=f"{name} (first {hc_limit} iters)")
        else:
            plt.plot(vals, label=name)

    plt.xlabel("Iteration")
    plt.ylabel("Cut value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def draw_cut_graph(edges: List[Edge], n: int, x: List[int], title: str, filename: str):
    if nx is None:
        print("[WARN] networkx not installed, skipping graph drawing:", filename)
        return

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)

    node_colors = ["tab:blue" if x[i] == 0 else "tab:red" for i in range(n)]
    cut_edges = [(u, v) for (u, v, w) in edges if x[u] != x[v]]
    same_edges = [(u, v) for (u, v, w) in edges if x[u] == x[v]]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, linewidths=0.5, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, edgelist=same_edges, width=1.0, alpha=0.25)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=2.5, alpha=0.9)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def draw_cut_graph_with_frequency(edges: List[Edge], n: int, x: List[int], flip_count: List[int],
                                  title: str, filename: str):
    """
    Advanced visual: node color still shows partition, but node size encodes flip frequency.
    """
    if nx is None:
        print("[WARN] networkx not installed, skipping graph drawing:", filename)
        return

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)

    node_colors = ["tab:blue" if x[i] == 0 else "tab:red" for i in range(n)]
    max_f = max(flip_count) if flip_count else 1
    # size in [300, 900]
    node_sizes = [300 + int(600 * (flip_count[i] / max_f)) for i in range(n)]

    cut_edges = [(u, v) for (u, v, w) in edges if x[u] != x[v]]
    same_edges = [(u, v) for (u, v, w) in edges if x[u] == x[v]]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           linewidths=0.5, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, edgelist=same_edges, width=1.0, alpha=0.20)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=2.8, alpha=0.90)

    plt.title(title + " (node size = flip frequency)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_flip_frequency(flip_count: List[int], title: str, filename: str):
    plt.figure()
    plt.bar(list(range(len(flip_count))), flip_count)
    plt.xlabel("Vertex")
    plt.ylabel("Flip count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_tenure_sensitivity(means: List[float], stds: List[float], labels: List[str], title: str, filename: str):
    plt.figure()
    x = list(range(len(labels)))
    plt.errorbar(x, means, yerr=stds, fmt='o-', capsize=4)
    plt.xticks(x, labels, rotation=15, ha='right')
    plt.xlabel("Tabu tenure scheme")
    plt.ylabel("Cut value (mean ± std)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_optimization_tradeoff(quality_means: Dict[str, float],
                               time_means: Dict[str, float],
                               title: str,
                               filename: str):
    """
    Visualize the optimization variant: quality vs time (lower time is better).
    """
    plt.figure()
    for name in quality_means:
        plt.scatter(time_means[name], quality_means[name])
        plt.text(time_means[name], quality_means[name], f" {name}")
    plt.xlabel("Mean runtime per run (s)")
    plt.ylabel("Mean best cut value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

# Hill Climbing (baseline) + history
@dataclass
class HCParams:
    max_iters: int = 20000
    seed: int = 0


def hill_climbing(edges, adj, n, params: HCParams, record_history: bool = False):
    rng = random.Random(params.seed)
    x = random_solution(n, rng)
    fx = cut_value(edges, x)

    history = [fx] if record_history else None

    for _ in range(params.max_iters):
        best_k = None
        best_delta = 0

        for k in range(n):
            d = delta_flip(adj, x, k)
            if d > best_delta:
                best_delta = d
                best_k = k

        if best_k is None:
            break

        x[best_k] ^= 1
        fx += best_delta

        if record_history:
            history.append(fx)

    if record_history:
        return x, fx, history
    return x, fx

# Tabu Search + history
@dataclass
class TabuParams:
    max_iters: int = 30000
    seed: int = 0
    tenure_min: int = 7
    tenure_rand: int = 7
    no_improve_limit: int = 8000


def tabu_search(edges, adj, n, params: TabuParams, record_history: bool = False, record_flips: bool = False):
    """
    Baseline Tabu: evaluates all n moves per iteration.
    """
    rng = random.Random(params.seed)
    x = random_solution(n, rng)
    fx = cut_value(edges, x)

    best_x = x[:]
    best_fx = fx

    tabu_until = [0] * n
    no_improve = 0

    cur_hist = [fx] if record_history else None
    best_hist = [best_fx] if record_history else None

    flip_count = [0] * n if record_flips else None

    for it in range(1, params.max_iters + 1):
        best_k = None
        best_delta = -10**18

        for k in range(n):
            d = delta_flip(adj, x, k)
            new_fx = fx + d

            tabu = it < tabu_until[k]
            aspiration = new_fx > best_fx
            if tabu and not aspiration:
                continue

            if d > best_delta:
                best_delta = d
                best_k = k

        if best_k is None:
            break

        x[best_k] ^= 1
        fx += best_delta

        if record_flips:
            flip_count[best_k] += 1

        tenure = params.tenure_min + rng.randint(0, params.tenure_rand)
        tabu_until[best_k] = it + tenure

        if fx > best_fx:
            best_fx = fx
            best_x = x[:]
            no_improve = 0
        else:
            no_improve += 1

        if record_history:
            cur_hist.append(fx)
            best_hist.append(best_fx)

        if no_improve >= params.no_improve_limit:
            break

    if record_history and record_flips:
        return best_x, best_fx, cur_hist, best_hist, flip_count
    if record_history:
        return best_x, best_fx, cur_hist, best_hist
    if record_flips:
        return best_x, best_fx, flip_count
    return best_x, best_fx

# OPTIMIZATION VARIANT: Candidate-list Tabu Search
@dataclass
class CandTabuParams(TabuParams):
    k_candidates: int = 20  # evaluate only top-K improving moves (plus tabu/aspiration rules)


def tabu_search_candidate_list(edges, adj, n, params: CandTabuParams):
    """
    Optimized Tabu: Candidate list.
    At each iteration:
      1) compute delta for all vertices (still O(n*deg) worst-case),
      2) select top-K vertices by delta (largest first),
      3) choose best admissible move among those K (tabu + aspiration).
    This reduces the expensive admissibility/selection loop and is a standard Tabu optimization.
    """
    rng = random.Random(params.seed)
    x = random_solution(n, rng)
    fx = cut_value(edges, x)

    best_x = x[:]
    best_fx = fx

    tabu_until = [0] * n
    no_improve = 0

    for it in range(1, params.max_iters + 1):
        deltas = [(delta_flip(adj, x, k), k) for k in range(n)]
        deltas.sort(reverse=True, key=lambda t: t[0])
        candidates = deltas[: max(1, min(params.k_candidates, n))]

        best_k = None
        best_delta = -10**18

        for d, k in candidates:
            new_fx = fx + d
            tabu = it < tabu_until[k]
            aspiration = new_fx > best_fx
            if tabu and not aspiration:
                continue
            if d > best_delta:
                best_delta = d
                best_k = k

        # fallback: if all candidates forbidden, try full scan
        if best_k is None:
            for k in range(n):
                d = delta_flip(adj, x, k)
                new_fx = fx + d
                tabu = it < tabu_until[k]
                aspiration = new_fx > best_fx
                if tabu and not aspiration:
                    continue
                if d > best_delta:
                    best_delta = d
                    best_k = k

        if best_k is None:
            break

        x[best_k] ^= 1
        fx += best_delta

        tenure = params.tenure_min + rng.randint(0, params.tenure_rand)
        tabu_until[best_k] = it + tenure

        if fx > best_fx:
            best_fx = fx
            best_x = x[:]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= params.no_improve_limit:
            break

    return best_x, best_fx


# Tabu Search with move recording (for simulation)
def tabu_search_with_moves(edges, adj, n, params: TabuParams, max_frames: int = 400):
    rng = random.Random(params.seed)
    x = random_solution(n, rng)
    fx = cut_value(edges, x)

    best_x = x[:]
    best_fx = fx

    tabu_until = [0] * n
    no_improve = 0

    moves = []
    cur_hist = [fx]
    best_hist = [best_fx]

    record_every = max(1, params.max_iters // max_frames)

    for it in range(1, params.max_iters + 1):
        best_k = None
        best_delta = -10**18

        for k in range(n):
            d = delta_flip(adj, x, k)
            new_fx = fx + d

            tabu = it < tabu_until[k]
            aspiration = new_fx > best_fx
            if tabu and not aspiration:
                continue

            if d > best_delta:
                best_delta = d
                best_k = k

        if best_k is None:
            break

        x[best_k] ^= 1
        fx += best_delta

        tenure = params.tenure_min + rng.randint(0, params.tenure_rand)
        tabu_until[best_k] = it + tenure

        if fx > best_fx:
            best_fx = fx
            best_x = x[:]
            no_improve = 0
        else:
            no_improve += 1

        cur_hist.append(fx)
        best_hist.append(best_fx)

        if it % record_every == 0:
            moves.append((it, best_k, fx, best_fx))

        if no_improve >= params.no_improve_limit:
            break

    return best_x, best_fx, moves, cur_hist, best_hist


def animate_tabu_simulation(edges: List[Edge], n: int, x0: List[int], moves,
                            filename_gif: Optional[str] = "tabu_sim.gif"):
    """
    Graph-only animation.
    """
    if nx is None:
        print("[WARN] networkx not installed, cannot animate graph.")
        return

    from matplotlib import animation

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)
    x = x0[:]

    def compute_edge_lists():
        cut_edges = [(u, v) for (u, v, w) in edges if x[u] != x[v]]
        same_edges = [(u, v) for (u, v, w) in edges if x[u] == x[v]]
        return same_edges, cut_edges

    fig = plt.figure(figsize=(9, 7))
    ax = plt.gca()
    ax.axis("off")

    node_colors = ["tab:blue" if x[i] == 0 else "tab:red" for i in range(n)]
    same_edges, cut_edges = compute_edge_lists()

    node_artist = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400,
                                         linewidths=0.5, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    same_artist = nx.draw_networkx_edges(G, pos, edgelist=same_edges, width=1.0, alpha=0.20, ax=ax)
    cut_artist = nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=2.8, alpha=0.90, ax=ax)
    title_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def update(frame_idx):
        nonlocal same_artist, cut_artist
        it, flipped_k, cur_fx, best_fx = moves[frame_idx]
        x[flipped_k] ^= 1

        node_artist.set_color(["tab:blue" if x[i] == 0 else "tab:red" for i in range(n)])

        same_edges, cut_edges = compute_edge_lists()
        if same_artist is not None:
            same_artist.remove()
        if cut_artist is not None:
            cut_artist.remove()
        same_artist = nx.draw_networkx_edges(G, pos, edgelist=same_edges, width=1.0, alpha=0.20, ax=ax)
        cut_artist = nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=2.8, alpha=0.90, ax=ax)

        title_text.set_text(f"Tabu Simulation | it={it} | flip={flipped_k} | cur={cur_fx} | best={best_fx}")
        return node_artist, same_artist, cut_artist, title_text

    anim = animation.FuncAnimation(fig, update, frames=len(moves), interval=120, blit=False, repeat=False)

    if filename_gif is not None:
        try:
            anim.save(filename_gif, writer="pillow", dpi=140)
            print("Saved simulation GIF:", filename_gif)
        except Exception as e:
            print("[WARN] Could not save GIF. Install pillow: pip install pillow")
            print("Error:", e)

    plt.show()


def animate_tabu_simulation_synced(edges: List[Edge], n: int, x0: List[int], moves,
                                   cur_hist: List[int], best_hist: List[int],
                                   filename_gif: Optional[str] = "tabu_sim_synced.gif"):
    """
    Advanced animation: graph + curve (synced).
    """
    if nx is None:
        print("[WARN] networkx not installed, cannot animate graph.")
        return

    from matplotlib import animation

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)
    x = x0[:]

    def compute_edge_lists():
        cut_edges = [(u, v) for (u, v, w) in edges if x[u] != x[v]]
        same_edges = [(u, v) for (u, v, w) in edges if x[u] == x[v]]
        return same_edges, cut_edges

    fig, (axG, axC) = plt.subplots(1, 2, figsize=(13, 6))
    axG.axis("off")

    # Initial graph
    node_colors = ["tab:blue" if x[i] == 0 else "tab:red" for i in range(n)]
    same_edges, cut_edges = compute_edge_lists()
    node_artist = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=360,
                                         linewidths=0.5, edgecolors="black", ax=axG)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=axG)
    same_artist = nx.draw_networkx_edges(G, pos, edgelist=same_edges, width=1.0, alpha=0.20, ax=axG)
    cut_artist = nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=2.5, alpha=0.90, ax=axG)
    title_text = axG.text(0.02, 0.98, "", transform=axG.transAxes, va="top")

    # Curve axes
    axC.set_xlabel("Iteration (sampled)")
    axC.set_ylabel("Cut value")
    axC.set_title("Tabu Search (current vs best)")
    line_cur, = axC.plot([], [], label="current")
    line_best, = axC.plot([], [], label="best-so-far")
    axC.legend()
    axC.set_xlim(0, max(1, len(moves)))
    ymin = min(min(cur_hist), min(best_hist))
    ymax = max(max(cur_hist), max(best_hist))
    pad = max(5, int(0.05 * (ymax - ymin + 1)))
    axC.set_ylim(ymin - pad, ymax + pad)

    def update(frame_idx):
        nonlocal same_artist, cut_artist
        it, flipped_k, cur_fx, best_fx = moves[frame_idx]
        x[flipped_k] ^= 1

        node_artist.set_color(["tab:blue" if x[i] == 0 else "tab:red" for i in range(n)])

        same_edges, cut_edges = compute_edge_lists()
        if same_artist is not None:
            same_artist.remove()
        if cut_artist is not None:
            cut_artist.remove()
        same_artist = nx.draw_networkx_edges(G, pos, edgelist=same_edges, width=1.0, alpha=0.20, ax=axG)
        cut_artist = nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=2.5, alpha=0.90, ax=axG)

        title_text.set_text(f"Synced Tabu | it={it} | flip={flipped_k} | cur={cur_fx} | best={best_fx}")

        # update curve (sampled points)
        xs = list(range(frame_idx + 1))
        cur_y = [moves[i][2] for i in xs]
        best_y = [moves[i][3] for i in xs]
        line_cur.set_data(xs, cur_y)
        line_best.set_data(xs, best_y)
        return node_artist, same_artist, cut_artist, title_text, line_cur, line_best

    anim = animation.FuncAnimation(fig, update, frames=len(moves), interval=120, blit=False, repeat=False)

    if filename_gif is not None:
        try:
            anim.save(filename_gif, writer="pillow", dpi=140)
            print("Saved synced simulation GIF:", filename_gif)
        except Exception as e:
            print("[WARN] Could not save synced GIF. Install pillow: pip install pillow")
            print("Error:", e)

    plt.show()

# Experiment runner (CSV output) + Tenure sensitivity figures
def run_experiments_and_tenure_plots():
    sizes = [50, 80, 120]
    p = 0.2
    w_max = 10

    seeds = [0, 1, 2, 3, 4]
    tenure_settings = [
        (5, 5),
        (10, 10),
        (20, 10),
    ]

    print("n,p,|E|,tenure_min,tenure_rand,random_best200_mean,hc_mean,tabu_mean,tabu_best")
    for n in sizes:
        edges, adj = generate_random_graph(n, p, w_max, seed=42)

        random_scores = []
        hc_scores = []

        for s in seeds:
            rng = random.Random(s)
            rb = max(cut_value(edges, random_solution(n, rng)) for _ in range(200))
            random_scores.append(rb)

            _, hc_fx = hill_climbing(edges, adj, n, HCParams(max_iters=20000, seed=s))
            hc_scores.append(hc_fx)

        random_mean = sum(random_scores) / len(random_scores)
        hc_mean = sum(hc_scores) / len(hc_scores)

        # tenure sensitivity collection
        labels = []
        means = []
        stds = []

        for tmin, trand in tenure_settings:
            tabu_scores = []
            tabu_best = -1

            for s in seeds:
                _, tabu_fx = tabu_search(
                    edges, adj, n,
                    TabuParams(max_iters=30000, seed=s, tenure_min=tmin, tenure_rand=trand, no_improve_limit=8000),
                    record_history=False,
                    record_flips=False
                )
                tabu_scores.append(tabu_fx)
                tabu_best = max(tabu_best, tabu_fx)

            tabu_mean = sum(tabu_scores) / len(tabu_scores)

            print(f"{n},{p},{len(edges)},{tmin},{trand},{random_mean:.1f},{hc_mean:.1f},{tabu_mean:.1f},{tabu_best}")

            labels.append(f"{tmin}+U(0,{trand})")
            means.append(statistics.mean(tabu_scores))
            stds.append(statistics.pstdev(tabu_scores) if len(tabu_scores) > 1 else 0.0)

        # plot tenure sensitivity for this n
        out = f"tenure_sensitivity_n{n}.png"
        plot_tenure_sensitivity(
            means, stds, labels,
            title=f"Tenure sensitivity (n={n}, p={p})",
            filename=out
        )
        print("Saved:", out)

# Visual demo: plots + graph cut images + flip frequency
def run_visual_demo():
    n = 40
    p = 0.2
    w_max = 10

    edges, adj = generate_random_graph(n, p, w_max, seed=42)

    hc_x, hc_fx, hc_hist = hill_climbing(
        edges, adj, n,
        HCParams(max_iters=5000, seed=0),
        record_history=True
    )

    tabu_x, tabu_fx, tabu_cur_hist, tabu_best_hist, flip_count = tabu_search(
        edges, adj, n,
        TabuParams(max_iters=8000, seed=0, tenure_min=10, tenure_rand=10, no_improve_limit=3000),
        record_history=True,
        record_flips=True
    )

    plot_curves(
        {
            "HillClimbing (current)": hc_hist,
            "Tabu (current)": tabu_cur_hist,
            "Tabu (best-so-far)": tabu_best_hist,
        },
        title=f"Max-Cut Search Dynamics (n={n}, p={p})",
        filename="curve_maxcut.png"
    )

    draw_cut_graph(edges, n, hc_x, title=f"Hill Climbing Cut (value={hc_fx})", filename="cut_hc.png")
    draw_cut_graph(edges, n, tabu_x, title=f"Tabu Search Cut (value={tabu_fx})", filename="cut_tabu.png")

    plot_flip_frequency(flip_count, title=f"Flip frequency (Tabu, n={n})", filename="flip_frequency.png")
    draw_cut_graph_with_frequency(
        edges, n, tabu_x, flip_count,
        title=f"Tabu Search Cut (value={tabu_fx})",
        filename="cut_tabu_with_frequency.png"
    )

    print("Saved visuals:")
    print("  curve_maxcut.png")
    print("  cut_hc.png")
    print("  cut_tabu.png")
    print("Saved:")
    print("  flip_frequency.png")
    print("  cut_tabu_with_frequency.png")

# Simulation demo: animated tabu search (GIF + synced GIF)
def run_simulation_demo():
    n = 35
    p = 0.2
    w_max = 10

    edges, adj = generate_random_graph(n, p, w_max, seed=42)

    rng = random.Random(0)
    x0 = random_solution(n, rng)

    params = TabuParams(max_iters=12000, seed=0, tenure_min=10, tenure_rand=10, no_improve_limit=4000)
    best_x, best_fx, moves, cur_hist, best_hist = tabu_search_with_moves(edges, adj, n, params, max_frames=250)

    plot_curves(
        {"Tabu (current)": cur_hist, "Tabu (best-so-far)": best_hist},
        title=f"Tabu Search Run (n={n}, p={p})",
        filename="curve_tabu_single_run.png"
    )

    animate_tabu_simulation(edges, n, x0, moves, filename_gif="tabu_sim.gif")
    animate_tabu_simulation_synced(edges, n, x0, moves, cur_hist, best_hist, filename_gif="tabu_sim_synced.gif")

    print("Best cut found (Tabu):", best_fx)
    print("Saved (if pillow installed): tabu_sim.gif, tabu_sim_synced.gif")

#Candidate-list vs Baseline
def run_optimization_comparison():
    n = 120
    p = 0.2
    w_max = 10
    edges, adj = generate_random_graph(n, p, w_max, seed=42)

    seeds = [0, 1, 2, 3, 4]
    base_params = TabuParams(max_iters=20000, tenure_min=10, tenure_rand=10, no_improve_limit=6000)
    cand_params = CandTabuParams(max_iters=20000, tenure_min=10, tenure_rand=10, no_improve_limit=6000, k_candidates=20)

    results_quality: Dict[str, List[float]] = {"Tabu (full scan)": [], "Tabu (candidate K=20)": []}
    results_time: Dict[str, List[float]] = {"Tabu (full scan)": [], "Tabu (candidate K=20)": []}

    for s in seeds:
        # baseline
        bp = TabuParams(**{**base_params.__dict__, "seed": s})
        t0 = time.perf_counter()
        _, fx_base = tabu_search(edges, adj, n, bp)
        t1 = time.perf_counter()
        results_quality["Tabu (full scan)"].append(fx_base)
        results_time["Tabu (full scan)"].append(t1 - t0)

        # candidate-list
        cp = CandTabuParams(**{**cand_params.__dict__, "seed": s})
        t0 = time.perf_counter()
        _, fx_cand = tabu_search_candidate_list(edges, adj, n, cp)
        t1 = time.perf_counter()
        results_quality["Tabu (candidate K=20)"].append(fx_cand)
        results_time["Tabu (candidate K=20)"].append(t1 - t0)

    def mean_std(xs: List[float]) -> Tuple[float, float]:
        return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)

    print("\n=== Optimization experiment (n=120, p=0.2) ===")
    for name in results_quality:
        q_mean, q_std = mean_std(results_quality[name])
        t_mean, t_std = mean_std(results_time[name])
        print(f"{name}: cut = {q_mean:.2f} ± {q_std:.2f} | time = {t_mean:.4f}s ± {t_std:.4f}s")

    quality_means = {name: statistics.mean(results_quality[name]) for name in results_quality}
    time_means = {name: statistics.mean(results_time[name]) for name in results_time}

    plot_optimization_tradeoff(
        quality_means=quality_means,
        time_means=time_means,
        title="Optimization: Candidate-list Tabu (quality vs runtime)",
        filename="optimization_tradeoff.png"
    )
    print("Saved: optimization_tradeoff.png")

# Main
if __name__ == "__main__":
    #Core experiments + tenure sensitivity (optimization via parameter tuning)
    run_experiments_and_tenure_plots()

    #Visual comparisons + memory interpretation (flip frequency)
    run_visual_demo()

    #Simulation
    run_simulation_demo()

    #optimization variant + experimental evaluation (candidate-list Tabu)
    run_optimization_comparison()
