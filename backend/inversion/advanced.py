"""Advanced Causal Analysis — Pure math extensions for true_inversion.

All functions take a CausalDAG (already inverted) and return analysis.
Zero LLM calls. Millisecond execution.

Includes:
  1. do-calculus (intervention vs observation)
  2. Mutual Information (info content per node)
  3. Critical Path (most probable causal chain)
  4. Markov Blanket (minimal explanatory set)
  5. Robustness (how many edges must flip to change prediction)
  6. Entropy Rate (entropy change per hour)
  7. Counterfactual (node removal experiments)
  8. Divergence (forward vs inverted KL-divergence)
"""

import math
import copy
from typing import Optional
import networkx as nx

from .true_inversion import (
    CausalDAG, CausalNode, CausalEdge,
    compute_marginals, bayesian_invert,
    compute_forward_entropy, compute_inverted_entropy,
)


# ══════════════════════════════════════════════════════════════
# 1. DO-CALCULUS — Intervention vs Observation
# ══════════════════════════════════════════════════════════════

def do_intervention(dag: CausalDAG, node_id: str, forced_value: float) -> dict:
    """Pearl's do-calculus: P(Y | do(X=x)) ≠ P(Y | X=x).

    Observation: P(outcome | node=high) — maybe node is high BECAUSE of outcome
    Intervention: P(outcome | do(node=high)) — we FORCE node to be high

    Implementation: cut all incoming edges to node, set its marginal = forced_value,
    then recompute everything downstream. This removes confounding.

    Returns: {outcome_id: new_probability} for all outcomes.
    """
    # Deep copy DAG to not mutate original
    dag_copy = _copy_dag(dag)

    # Cut incoming edges (do-operator removes causes)
    parents = dag_copy.get_parents(node_id)
    for pid in parents:
        key = (pid, node_id)
        if key in dag_copy.edges:
            del dag_copy.edges[key]
            dag_copy.graph.remove_edge(pid, node_id)

    # Force the node's probability
    dag_copy.nodes[node_id].prior_probability = forced_value
    dag_copy.nodes[node_id].marginal_probability = forced_value

    # Recompute downstream
    compute_marginals(dag_copy)

    # Collect outcome changes
    results = {}
    for lid in dag_copy.get_leaves():
        orig = dag.nodes[lid].marginal_probability
        new = dag_copy.nodes[lid].marginal_probability
        results[lid] = {
            "outcome": dag.nodes[lid].content,
            "original_prob": round(orig, 4),
            "do_prob": round(new, 4),
            "causal_effect": round(new - orig, 4),
            "is_causal": abs(new - orig) > 0.01,
        }

    return {
        "intervention": f"do({node_id}={forced_value})",
        "node_content": dag.nodes[node_id].content,
        "outcome_effects": results,
    }


def do_analysis(dag: CausalDAG, max_nodes: int = 20) -> list[dict]:
    """Run do-intervention on non-trivial nodes.
    
    max_nodes: limit for performance on large graphs. Picks highest-marginal nodes.
    """
    results = []
    candidates = [(nid, node) for nid, node in dag.nodes.items() 
                  if node.node_type not in ("seed", "outcome")]
    # Sort by marginal probability — most impactful nodes first
    candidates.sort(key=lambda x: x[1].marginal_probability, reverse=True)
    candidates = candidates[:max_nodes]

    for nid, node in candidates:

        high = do_intervention(dag, nid, 0.9)
        low = do_intervention(dag, nid, 0.1)

        # Causal power = max effect across outcomes
        max_effect = 0
        for lid in dag.get_leaves():
            h = high["outcome_effects"].get(lid, {}).get("do_prob", 0)
            l = low["outcome_effects"].get(lid, {}).get("do_prob", 0)
            max_effect = max(max_effect, abs(h - l))

        results.append({
            "node_id": nid,
            "content": node.content,
            "causal_power": round(max_effect, 4),
            "do_high": {k: v["do_prob"] for k, v in high["outcome_effects"].items()},
            "do_low": {k: v["do_prob"] for k, v in low["outcome_effects"].items()},
        })

    results.sort(key=lambda x: x["causal_power"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════
# 2. MUTUAL INFORMATION
# ══════════════════════════════════════════════════════════════

def mutual_information(dag: CausalDAG, node_id: str, outcome_id: str) -> float:
    """I(Node; Outcome) — how much knowing this node reduces uncertainty about outcome.

    MI = H(Outcome) - H(Outcome | Node)
    High MI = this node is very informative about the outcome.
    """
    # H(Outcome) — unconditional entropy of outcome
    p_out = dag.nodes[outcome_id].marginal_probability
    h_out = _binary_entropy(p_out)

    # H(Outcome | Node=high) via do-calculus
    p_node = dag.nodes[node_id].marginal_probability
    high_result = do_intervention(dag, node_id, 0.9)
    low_result = do_intervention(dag, node_id, 0.1)

    p_out_high = high_result["outcome_effects"].get(outcome_id, {}).get("do_prob", p_out)
    p_out_low = low_result["outcome_effects"].get(outcome_id, {}).get("do_prob", p_out)

    h_out_high = _binary_entropy(p_out_high)
    h_out_low = _binary_entropy(p_out_low)

    # Conditional entropy: H(Out|Node) = P(Node_high) × H(Out|high) + P(Node_low) × H(Out|low)
    h_conditional = p_node * h_out_high + (1 - p_node) * h_out_low

    return max(0.0, round(h_out - h_conditional, 4))


def information_ranking(dag: CausalDAG, max_nodes: int = 15) -> list[dict]:
    """Rank nodes by mutual information with outcomes. Limited for performance."""
    results = []
    leaves = dag.get_leaves()

    candidates = [(nid, node) for nid, node in dag.nodes.items()
                  if node.node_type not in ("seed", "outcome")]
    candidates.sort(key=lambda x: x[1].marginal_probability, reverse=True)
    candidates = candidates[:max_nodes]

    for nid, node in candidates:

        mi_scores = {}
        total_mi = 0
        for lid in leaves:
            mi = mutual_information(dag, nid, lid)
            mi_scores[lid] = mi
            total_mi += mi

        results.append({
            "node_id": nid,
            "content": node.content,
            "total_mi": round(total_mi, 4),
            "per_outcome": mi_scores,
        })

    results.sort(key=lambda x: x["total_mi"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════
# 3. CRITICAL PATH
# ══════════════════════════════════════════════════════════════

def critical_path(dag: CausalDAG, outcome_id: str) -> dict:
    """Find the most probable causal chain from seed to outcome.

    Uses negative log probability as weight → shortest path = most probable chain.
    """
    roots = dag.get_roots()
    if not roots:
        return {"found": False}

    # Build weight graph: -log(p) so shortest path = highest probability
    weight_graph = nx.DiGraph()
    for (src, tgt), edge in dag.edges.items():
        p = edge.probability * dag.nodes[src].marginal_probability
        if p > 0:
            weight_graph.add_edge(src, tgt, weight=-math.log(max(p, 1e-10)))

    best_path = None
    best_prob = 0

    for root in roots:
        try:
            path = nx.shortest_path(weight_graph, root, outcome_id, weight="weight")
            # Calculate path probability
            prob = 1.0
            for i in range(len(path) - 1):
                edge = dag.edges.get((path[i], path[i + 1]))
                if edge:
                    prob *= edge.probability
            if prob > best_prob:
                best_prob = prob
                best_path = path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if not best_path:
        return {"found": False}

    chain = []
    for i, nid in enumerate(best_path):
        node = dag.nodes[nid]
        edge_prob = None
        if i < len(best_path) - 1:
            e = dag.edges.get((nid, best_path[i + 1]))
            edge_prob = e.probability if e else None

        chain.append({
            "node_id": nid,
            "content": node.content,
            "time_hours": node.time_offset_hours,
            "edge_prob_to_next": edge_prob,
        })

    return {
        "found": True,
        "path": chain,
        "total_probability": round(best_prob, 4),
        "length": len(best_path),
        "outcome": dag.nodes[outcome_id].content,
    }


# ══════════════════════════════════════════════════════════════
# 4. MARKOV BLANKET
# ══════════════════════════════════════════════════════════════

def markov_blanket(dag: CausalDAG, outcome_id: str) -> dict:
    """Find the Markov Blanket of an outcome node.

    MB(X) = parents(X) ∪ children(X) ∪ parents_of_children(X)

    Knowing the Markov Blanket makes the outcome conditionally independent
    of ALL other nodes. This is the MINIMAL set you need to monitor.
    """
    parents = set(dag.get_parents(outcome_id))
    children = set(dag.get_children(outcome_id))
    parents_of_children = set()
    for cid in children:
        parents_of_children.update(dag.get_parents(cid))
    parents_of_children.discard(outcome_id)

    blanket = parents | children | parents_of_children

    return {
        "outcome": dag.nodes[outcome_id].content,
        "blanket_size": len(blanket),
        "total_nodes": len(dag.nodes),
        "compression": round(1 - len(blanket) / max(1, len(dag.nodes) - 1), 2),
        "nodes": [
            {
                "node_id": nid,
                "content": dag.nodes[nid].content,
                "role": "parent" if nid in parents else "child" if nid in children else "co-parent",
                "necessity": round(dag.nodes[nid].inverted_probability, 4),
            }
            for nid in blanket
        ],
        "interpretation": f"Monitor only these {len(blanket)} nodes (out of {len(dag.nodes)-1}). "
                         f"Everything else is irrelevant given this set.",
    }


# ══════════════════════════════════════════════════════════════
# 5. ROBUSTNESS
# ══════════════════════════════════════════════════════════════

def robustness_score(dag: CausalDAG) -> dict:
    """How many edges must change for the top prediction to flip?

    Progressively weaken the strongest edges and see when ranking changes.
    Higher robustness = more confident prediction.
    """
    # Get current outcome ranking
    compute_marginals(dag)
    outcomes = [(lid, dag.nodes[lid].marginal_probability) for lid in dag.get_leaves()]
    outcomes.sort(key=lambda x: x[1], reverse=True)

    if len(outcomes) < 2:
        return {"robustness": 1.0, "edges_to_flip": "N/A"}

    top_id = outcomes[0][0]
    runner_id = outcomes[1][0]

    # Try weakening edges one by one
    dag_copy = _copy_dag(dag)
    edges_by_importance = sorted(
        dag_copy.edges.items(),
        key=lambda x: x[1].probability,
        reverse=True,
    )

    flips_needed = 0
    for (src, tgt), edge in edges_by_importance:
        edge.probability *= 0.3  # Weaken by 70%
        compute_marginals(dag_copy)
        flips_needed += 1

        new_outcomes = [(lid, dag_copy.nodes[lid].marginal_probability) for lid in dag_copy.get_leaves()]
        new_outcomes.sort(key=lambda x: x[1], reverse=True)

        if new_outcomes[0][0] != top_id:
            break

    return {
        "top_prediction": dag.nodes[top_id].content,
        "runner_up": dag.nodes[runner_id].content,
        "edges_to_flip": flips_needed,
        "total_edges": len(dag.edges),
        "robustness": round(flips_needed / max(1, len(dag.edges)), 4),
        "interpretation": f"Need to break {flips_needed}/{len(dag.edges)} edges to change the top prediction.",
    }


# ══════════════════════════════════════════════════════════════
# 6. ENTROPY RATE
# ══════════════════════════════════════════════════════════════

def entropy_rate(dag: CausalDAG) -> list[dict]:
    """Entropy change per hour — where does uncertainty grow fastest?

    Nodes on the time axis are compared: ΔH/Δt.
    High rate = uncertainty explodes here (prediction degrades fast).
    Low rate = stable, predictable zone.
    """
    timed_nodes = [(nid, n) for nid, n in dag.nodes.items() if n.time_offset_hours > 0]
    timed_nodes.sort(key=lambda x: x[1].time_offset_hours)

    rates = []
    for i in range(1, len(timed_nodes)):
        nid, node = timed_nodes[i]
        prev_nid, prev_node = timed_nodes[i - 1]

        dt = node.time_offset_hours - prev_node.time_offset_hours
        if dt <= 0:
            continue

        dh_fwd = node.forward_entropy - prev_node.forward_entropy
        dh_inv = node.inverted_entropy - prev_node.inverted_entropy

        rates.append({
            "from_node": prev_nid,
            "to_node": nid,
            "from_content": prev_node.content[:30],
            "to_content": node.content[:30],
            "hours": round(dt, 1),
            "fwd_entropy_rate": round(dh_fwd / dt, 4) if dt > 0 else 0,
            "inv_entropy_rate": round(dh_inv / dt, 4) if dt > 0 else 0,
            "net_rate": round((dh_fwd - dh_inv) / dt, 4) if dt > 0 else 0,
        })

    return rates


# ══════════════════════════════════════════════════════════════
# 7. COUNTERFACTUAL — Node removal experiments
# ══════════════════════════════════════════════════════════════

def counterfactual_removal(dag: CausalDAG, node_id: str) -> dict:
    """'What if this node didn't exist?' — Remove node and see what changes.

    Different from do-calculus: do() sets a value, counterfactual REMOVES entirely.
    Shows how structurally important a node is to the causal network.
    """
    if node_id not in dag.nodes:
        return {"error": "node not found"}

    # Original outcomes
    compute_marginals(dag)
    original = {lid: dag.nodes[lid].marginal_probability for lid in dag.get_leaves()}

    # Remove node from copy
    dag_copy = _copy_dag(dag)
    # Remove all edges involving this node
    edges_to_remove = [(s, t) for (s, t) in list(dag_copy.edges.keys()) if s == node_id or t == node_id]
    for key in edges_to_remove:
        del dag_copy.edges[key]
        if dag_copy.graph.has_edge(*key):
            dag_copy.graph.remove_edge(*key)
    if node_id in dag_copy.nodes:
        del dag_copy.nodes[node_id]
        if dag_copy.graph.has_node(node_id):
            dag_copy.graph.remove_node(node_id)

    # Recompute
    try:
        compute_marginals(dag_copy)
    except Exception:
        return {"error": "graph disconnected after removal"}

    # Compare
    effects = {}
    for lid in dag.get_leaves():
        if lid in dag_copy.nodes:
            new_prob = dag_copy.nodes[lid].marginal_probability
        else:
            new_prob = 0.0
        orig_prob = original.get(lid, 0)
        effects[lid] = {
            "outcome": dag.nodes[lid].content,
            "original": round(orig_prob, 4),
            "without_node": round(new_prob, 4),
            "impact": round(orig_prob - new_prob, 4),
        }

    total_impact = sum(abs(v["impact"]) for v in effects.values())

    return {
        "removed_node": node_id,
        "content": dag.nodes[node_id].content,
        "total_impact": round(total_impact, 4),
        "outcome_effects": effects,
        "is_structurally_critical": total_impact > 0.05,
    }


def full_counterfactual_analysis(dag: CausalDAG) -> list[dict]:
    """Run counterfactual removal on every non-seed, non-outcome node."""
    results = []
    for nid, node in dag.nodes.items():
        if node.node_type in ("seed", "outcome"):
            continue
        cf = counterfactual_removal(dag, nid)
        if "error" not in cf:
            results.append(cf)
    results.sort(key=lambda x: x["total_impact"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════
# 8. KL DIVERGENCE — Forward vs Inverted distribution distance
# ══════════════════════════════════════════════════════════════

def kl_divergence(dag: CausalDAG) -> dict:
    """KL(Forward || Inverted) — how different are the two probability views?

    Uses proper binary KL divergence with numerical stability.
    Only computes on non-trivial nodes (skip seed/outcomes).
    """
    kl_sum = 0.0
    n = 0

    for nid, node in dag.nodes.items():
        if node.node_type in ("seed", "outcome"):
            continue

        p = max(1e-8, min(1 - 1e-8, node.marginal_probability))
        q = max(1e-8, min(1 - 1e-8, node.inverted_probability))

        # Binary KL: p*log(p/q) + (1-p)*log((1-p)/(1-q))
        kl_node = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        kl_sum += max(0.0, kl_node)  # KL is always non-negative
        n += 1

    avg_kl = kl_sum / max(1, n)

    return {
        "kl_divergence": round(avg_kl, 4),
        "nodes_compared": n,
        "interpretation": (
            "LOW — forward and inverted views largely agree. Results are reliable."
            if avg_kl < 0.15 else
            "MODERATE — some disagreement between forward prediction and backward inference."
            if avg_kl < 0.5 else
            "HIGH — significant disagreement. The forward model's assumptions may not hold under inversion."
        ),
    }


# ══════════════════════════════════════════════════════════════
# COMBINED: run_advanced_analysis
# ══════════════════════════════════════════════════════════════

def run_advanced_analysis(dag: CausalDAG) -> dict:
    """Run all advanced analyses. Add to run_full_inversion output."""

    # Ensure base inversion is done
    compute_marginals(dag)
    bayesian_invert(dag)
    compute_forward_entropy(dag)
    compute_inverted_entropy(dag)

    return {
        "do_calculus": do_analysis(dag),
        "information_ranking": information_ranking(dag),
        "critical_paths": {
            lid: critical_path(dag, lid) for lid in dag.get_leaves()
        },
        "markov_blankets": {
            lid: markov_blanket(dag, lid) for lid in dag.get_leaves()
        },
        "robustness": robustness_score(dag),
        "entropy_rate": entropy_rate(dag),
        "counterfactuals": full_counterfactual_analysis(dag),
        "kl_divergence": kl_divergence(dag),
    }


# ══════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════

def _binary_entropy(p: float) -> float:
    """H(p) = -p log p - (1-p) log(1-p)"""
    p = max(1e-10, min(1 - 1e-10, p))
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _copy_dag(dag: CausalDAG) -> CausalDAG:
    """Deep copy a CausalDAG without using copy.deepcopy (too slow for large graphs)."""
    new = CausalDAG()
    for nid, node in dag.nodes.items():
        new.add_node(CausalNode(
            id=node.id, content=node.content, node_type=node.node_type,
            prior_probability=node.prior_probability,
            time_offset_hours=node.time_offset_hours,
        ))
    for (src, tgt), edge in dag.edges.items():
        new.add_edge(src, tgt, edge.probability, edge.delay_hours, edge.gate)
    new.correlations = dict(dag.correlations)
    compute_marginals(new)
    bayesian_invert(new)
    return new
