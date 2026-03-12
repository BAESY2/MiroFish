"""Test TRUE Inversion — Pure math, zero LLM calls.

This demonstrates the actual inversion engine.
No API key needed. Runs in milliseconds.

Usage:
    python test_true_inversion.py
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inversion.true_inversion import (
    CausalDAG, CausalNode, CausalEdge,
    run_full_inversion, build_dag_from_mirofish,
)


def test_manual_dag():
    """Build a causal DAG manually and invert it.

    Scenario: Trump 60% tariff on China

    DAG:
      SEED(tariff_announced)
        → V1(market_reaction)      P=0.95
        │  → N1a(panic_sell)        P=0.6
        │  → N1b(buy_the_dip)      P=0.3
        → V2(china_response)        P=0.90
        │  → N2a(full_retaliation)  P=0.5
        │  → N2b(negotiate)         P=0.4
        │  → N2c(ignore)            P=0.1
        → V3(supply_chain)          P=0.85
           → N3a(reshoring)         P=0.3
           → N3b(redirect_vietnam)  P=0.5

    Outcomes (leaf nodes connected from nerves):
      O1(trade_war)     ← N2a, N1a        P=0.35
      O2(deal_made)     ← N2b, N1b        P=0.40
      O3(decoupling)    ← N2c, N3a, N3b   P=0.15
      O4(market_crash)  ← N1a, N2a        P=0.10
    """
    dag = CausalDAG()

    # Seed
    dag.add_node(CausalNode(id="seed", content="Trump announces 60% tariffs on all Chinese imports", node_type="seed", prior_probability=1.0))

    # Spine V1: Market Reaction
    dag.add_node(CausalNode(id="v1", content="Global markets react within hours — volatility spikes, safe havens rally", node_type="event", prior_probability=0.95))
    dag.add_edge("seed", "v1", 0.95)

    # Spine V2: China Response
    dag.add_node(CausalNode(id="v2", content="Chinese government formulates official response within 48-72 hours", node_type="event", prior_probability=0.90))
    dag.add_edge("seed", "v2", 0.90)

    # Spine V3: Supply Chain
    dag.add_node(CausalNode(id="v3", content="Multinational corporations begin supply chain contingency planning", node_type="event", prior_probability=0.85))
    dag.add_edge("v1", "v3", 0.85)

    # Nerve branches — V1
    dag.add_node(CausalNode(id="n1a", content="Panic selling — S&P drops 3-5% in single session, VIX above 30", node_type="event", prior_probability=0.60))
    dag.add_edge("v1", "n1a", 0.60)

    dag.add_node(CausalNode(id="n1b", content="Institutional investors buy the dip, seeing tariffs as negotiation tactic", node_type="event", prior_probability=0.30))
    dag.add_edge("v1", "n1b", 0.30)

    # Nerve branches — V2
    dag.add_node(CausalNode(id="n2a", content="China retaliates with matching tariffs on US agriculture + tech", node_type="event", prior_probability=0.50))
    dag.add_edge("v2", "n2a", 0.50)

    dag.add_node(CausalNode(id="n2b", content="China signals willingness to negotiate, proposes bilateral summit", node_type="event", prior_probability=0.40))
    dag.add_edge("v2", "n2b", 0.40)

    dag.add_node(CausalNode(id="n2c", content="China largely ignores tariffs, focuses on domestic consumption pivot", node_type="event", prior_probability=0.10))
    dag.add_edge("v2", "n2c", 0.10)

    # Nerve branches — V3
    dag.add_node(CausalNode(id="n3a", content="Major firms announce reshoring plans to US/Mexico", node_type="event", prior_probability=0.30))
    dag.add_edge("v3", "n3a", 0.30)

    dag.add_node(CausalNode(id="n3b", content="Supply chains redirect through Vietnam/India/Indonesia", node_type="event", prior_probability=0.50))
    dag.add_edge("v3", "n3b", 0.50)

    # Outcomes
    dag.add_node(CausalNode(id="o1", content="OUTCOME: Full trade war — tit-for-tat escalation lasting 6+ months", node_type="outcome", prior_probability=0.35))
    dag.add_edge("n2a", "o1", 0.70)
    dag.add_edge("n1a", "o1", 0.40)

    dag.add_node(CausalNode(id="o2", content="OUTCOME: Negotiated deal — tariffs reduced to 25-35% with exemptions", node_type="outcome", prior_probability=0.40))
    dag.add_edge("n2b", "o2", 0.75)
    dag.add_edge("n1b", "o2", 0.50)

    dag.add_node(CausalNode(id="o3", content="OUTCOME: Accelerated US-China economic decoupling", node_type="outcome", prior_probability=0.15))
    dag.add_edge("n2c", "o3", 0.60)
    dag.add_edge("n3a", "o3", 0.40)
    dag.add_edge("n3b", "o3", 0.30)

    dag.add_node(CausalNode(id="o4", content="OUTCOME: Market crash >10% forces policy reversal", node_type="outcome", prior_probability=0.10))
    dag.add_edge("n1a", "o4", 0.30)
    dag.add_edge("n2a", "o4", 0.20)

    return dag


def print_results(result: dict, dag: CausalDAG):
    """Print inversion results in a readable format."""

    print("=" * 70)
    print("  TRUE INVERSION — PURE MATH, ZERO LLM CALLS")
    print("  Bayesian DAG Reversal + Shannon Entropy")
    print("=" * 70)

    # Entropy stats
    stats = result["entropy"]
    print(f"\n  ENTROPY OVERVIEW")
    print(f"  Forward avg:  {stats['avg_fwd']:.4f} (uncertainty ↑ with depth)")
    print(f"  Inverted avg: {stats['avg_inv']:.4f} (certainty ↑ toward outcome)")
    g = result.get("graph", {})
    print(f"  Nodes: {g.get('nodes', len(dag.nodes))}  Edges: {g.get('edges', len(dag.edges))}")
    print(f"  LLM calls in inversion: {result.get('llm_calls', 0)}")

    # Node-by-node
    print(f"\n{'=' * 70}")
    print(f"  NODE-BY-NODE: FORWARD vs INVERTED")
    print(f"{'=' * 70}")
    print(f"  {'ID':<6} {'Content':<45} {'FwdH':>6} {'InvH':>6} {'InvP':>6} {'Grad':>6}")
    print(f"  {'─'*6} {'─'*45} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

    for node_id in dag.topo():
        n = dag.nodes[node_id]
        marker = ""
        if n.node_type == "outcome":
            marker = " ◄"
        print(f"  {node_id:<6} {n.content[:45]:<45} {n.forward_entropy:>6.3f} {n.inverted_entropy:>6.3f} {n.inverted_probability:>6.3f} {n.entropy_gradient:>+6.3f}{marker}")

    # Turnstile
    ts = result["turnstile"]
    print(f"\n{'=' * 70}")
    print(f"  ⚡ TURNSTILE (Point of No Return)")
    print(f"{'=' * 70}")
    if ts.get("found"):
        print(f"  Node: {ts['node_id']}")
        print(f"  Event: {ts['content']}")
        print(f"  Entropy gradient: {ts.get('gradient', 0):.4f} (closest to zero)")
        print(f"  Forward entropy:  {ts.get('fwd_entropy', 0):.4f}")
        print(f"  Inverted entropy: {ts.get('inv_entropy', 0):.4f}")
        print(f"  Necessity score:  {ts.get('necessity', 0):.4f}")
        print(f"  (Turnstile: point of no return)")
    else:
        print("  No turnstile found.")

    # Pincer scores (top 10)
    print(f"\n{'=' * 70}")
    print(f"  TEMPORAL PINCER (Forward × Inverted)")
    print(f"{'=' * 70}")
    print(f"  {'Node':<6} {'Content':<40} {'Fwd':>6} {'Inv':>6} {'Pincer':>7}")
    print(f"  {'─'*6} {'─'*40} {'─'*6} {'─'*6} {'─'*7}")
    pincer_list = result.get("pincer", [])[:10]
    for ps in pincer_list:
        fwd = ps.get("fwd", 0)
        inv = ps.get("inv", 0)
        pinc = ps.get("pincer", 0)
        print(f"  {ps.get('node_id',''):<6} {(ps.get('content','')[:40]):<40} {fwd:>6.3f} {inv:>6.3f} {pinc:>7.4f}")

    # Necessity analysis per outcome
    print(f"\n{'=' * 70}")
    print(f"  NECESSITY ANALYSIS — 'What HAD to be true?'")
    print(f"{'=' * 70}")
    for outcome_id, data in result.get("necessities", {}).items():
        print(f"\n  {data.get('outcome', outcome_id)[:60]}")
        for cond in data.get("conditions", []):
            crit = " ★ CRITICAL" if cond.get("critical") else ""
            bar_len = int(cond.get("necessity", 0) * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            print(f"    [{bar}] {cond.get('necessity', 0):.3f}  {cond.get('content', '')[:50]}{crit}")

    # Edge inversion detail
    print(f"\n{'=' * 70}")
    print(f"  EDGE INVERSION (Bayes' Theorem)")
    print(f"{'=' * 70}")
    print(f"  {'Edge':<15} {'P(B|A)':>8} {'P(A|B)':>8} {'Δ':>8}")
    print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*8}")
    for (src, tgt), edge in dag.edges.items():
        delta = edge.inverted_probability - edge.probability
        print(f"  {src+'→'+tgt:<15} {edge.probability:>8.3f} {edge.inverted_probability:>8.3f} {delta:>+8.3f}")


def main():
    # Build DAG
    dag = test_manual_dag()

    # Run TRUE inversion (pure math, instant)
    import time
    t0 = time.time()
    result = run_full_inversion(dag)
    elapsed = (time.time() - t0) * 1000  # milliseconds

    # Print
    print_results(result, dag)

    print(f"\n  ⏱  Inversion completed in {elapsed:.1f}ms")
    print(f"  📊 Method: {result['method']}")
    print(f"  LLM calls: {result.get('llm_calls', 0)}")

    # Save
    # Remove non-serializable objects for JSON
    output = {k: v for k, v in result.items()}
    with open("true_inversion_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  [SAVED] true_inversion_result.json")


if __name__ == "__main__":
    main()
