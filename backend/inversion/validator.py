"""Validator — LLM verification layer for inversion engine.

The math is exact. But math on bad inputs = precise garbage.
This module validates inputs AND interprets outputs.

THREE ROLES:

1. INPUT VALIDATOR (pre-inversion)
   - Are edge probabilities reasonable?
   - Does the causal chain make logical sense?
   - Are there missing obvious causes?
   
2. RESULT VALIDATOR (post-inversion)
   - Does the turnstile make intuitive sense?
   - Are necessity scores logically coherent?
   - Flag any mathematically correct but semantically absurd results

3. INTERPRETER (output)
   - Translate math into actionable natural language
   - Generate the "so what?" summary
   - Create the README-worthy one-liner

These are the ONLY LLM calls in the entire inversion pipeline.
The inversion itself remains pure math.
"""

import json


VALIDATOR_SYSTEM = """You are a causal reasoning validator.
Your job is to check if probabilistic causal models make sense.
Be specific. If something is wrong, say exactly what and why.
If everything checks out, say so briefly."""


async def validate_dag_inputs(llm_call, dag_description: str) -> dict:
    """Pre-inversion: check if the DAG structure and probabilities are reasonable.
    
    Args:
        llm_call: async (prompt, system) -> str
        dag_description: formatted string of nodes and edges with probabilities
    """
    prompt = f"""Validate this causal graph for logical consistency.

CAUSAL GRAPH:
{dag_description}

Check for:
1. PROBABILITY SANITY: Are the P(B|A) values reasonable? (e.g., P(market crash | tariff) = 0.99 would be too high)
2. MISSING LINKS: Are there obvious causal connections that are missing?
3. DIRECTION ERRORS: Any edges that should point the other way?
4. INDEPENDENCE VIOLATIONS: Are nodes marked independent that shouldn't be?

Return JSON:
{{"valid": true/false,
  "issues": [{{"type": "probability|missing_link|direction|independence", "description": "...", "severity": "low|medium|high", "fix": "suggested fix"}}],
  "suggested_edges": [{{"from": "node_id", "to": "node_id", "reason": "...", "suggested_prob": 0.0-1.0}}],
  "probability_adjustments": [{{"edge": "A→B", "current": 0.0, "suggested": 0.0, "reason": "..."}}]
}}"""

    text = await llm_call(prompt, VALIDATOR_SYSTEM)
    return _parse(text)


async def validate_results(llm_call, inversion_summary: str) -> dict:
    """Post-inversion: check if results make intuitive sense.
    
    Catches: mathematically correct but semantically absurd results.
    """
    prompt = f"""Validate these inversion analysis results for logical coherence.

RESULTS:
{inversion_summary}

Check for:
1. TURNSTILE SENSE: Does the identified "point of no return" make logical sense for this scenario?
2. NECESSITY COHERENCE: Do the "critical preconditions" logically follow? Any obviously wrong ones?
3. SENSITIVITY LOGIC: Do the most sensitive edges make intuitive sense?
4. MISSING INSIGHTS: Is there an obvious insight the math missed?

Return JSON:
{{"coherent": true/false,
  "flags": [{{"item": "turnstile|necessity|sensitivity|missing", "issue": "...", "severity": "low|medium|high"}}],
  "missing_insights": ["insight the math couldn't capture"],
  "confidence_in_results": 0.0-1.0
}}"""

    text = await llm_call(prompt, VALIDATOR_SYSTEM)
    return _parse(text)


async def interpret_results(llm_call, seed_text: str, full_result: dict) -> dict:
    """Translate math results into actionable natural language.
    
    This is what goes in the README / report.
    """
    # Build concise summary of key findings
    ts = full_result.get("turnstile", {})
    nec = full_result.get("necessities", {})
    sens = full_result.get("sensitivity", {})
    mc = full_result.get("monte_carlo", {})
    adv = full_result.get("advanced", {})

    summary_parts = [f"Scenario: {seed_text[:150]}"]

    if ts.get("found"):
        summary_parts.append(f"Turnstile: '{ts['content']}' at {ts.get('time_hours',0)}h (gradient={ts.get('gradient',0)})")

    for lid, nd in nec.items():
        critical = [c for c in nd.get("conditions", []) if c.get("critical") and not c.get("trivial")]
        if critical:
            summary_parts.append(f"For '{nd['outcome']}' (prob={nd['prob']}): critical conditions = {', '.join(c['content'][:30] for c in critical[:3])}")

    if adv:
        rob = adv.get("robustness", {})
        if rob:
            summary_parts.append(f"Robustness: {rob.get('edges_to_flip','?')}/{rob.get('total_edges','?')} edges to flip")
        mb = adv.get("markov_blankets", {})
        for lid, m in mb.items():
            summary_parts.append(f"Monitor for '{m.get('outcome','')[:20]}': only {m.get('blanket_size',0)} nodes needed ({m.get('compression',0):.0%} compression)")

    summary = "\n".join(summary_parts)

    prompt = f"""Based on this mathematical analysis, write a clear, actionable interpretation.

ANALYSIS:
{summary}

Write THREE things:
1. HEADLINE: One sentence that captures the key finding (for README)
2. ACTION ITEMS: 3-5 specific things someone should watch/do based on this analysis
3. CONFIDENCE ASSESSMENT: How much should someone trust these numbers and why

Return JSON:
{{"headline": "one powerful sentence",
  "action_items": ["specific action 1", "specific action 2", ...],
  "confidence_assessment": "honest assessment of reliability",
  "key_uncertainty": "the single biggest unknown",
  "time_to_decision": "how long before this becomes clear"
}}"""

    text = await llm_call(prompt, VALIDATOR_SYSTEM)
    return _parse(text)


def format_dag_for_validation(dag) -> str:
    """Format DAG into human-readable string for LLM validation."""
    lines = ["NODES:"]
    for nid, n in dag.nodes.items():
        lines.append(f"  {nid}: '{n.content}' (type={n.node_type}, prior={n.prior_probability}, t={n.time_offset_hours}h)")
    
    lines.append("\nEDGES:")
    for (src, tgt), e in dag.edges.items():
        lines.append(f"  {src} → {tgt}: P={e.probability} (delay={e.delay_hours}h, gate={e.gate})")
    
    if dag.correlations:
        lines.append("\nCORRELATIONS:")
        seen = set()
        for (e1, e2), corr in dag.correlations.items():
            key = (min(e1, e2), max(e1, e2))
            if key not in seen:
                seen.add(key)
                lines.append(f"  {e1} ~ {e2}: {corr}")
    
    return "\n".join(lines)


def format_results_for_validation(result: dict, advanced: dict = None) -> str:
    """Format inversion results for LLM validation."""
    lines = []
    
    ts = result.get("turnstile", {})
    if ts.get("found"):
        lines.append(f"TURNSTILE: {ts['content']} at {ts.get('time_hours',0)}h (gradient={ts.get('gradient',0):.4f})")
    
    for lid, nd in result.get("necessities", {}).items():
        lines.append(f"\nOUTCOME '{nd['outcome']}' (prob={nd['prob']}):")
        for c in nd.get("conditions", [])[:5]:
            triv = " [trivial]" if c.get("trivial") else ""
            crit = " ★" if c.get("critical") else ""
            lines.append(f"  necessity={c['necessity']:.3f} '{c['content'][:40]}'{crit}{triv}")
    
    if advanced:
        rob = advanced.get("robustness", {})
        if rob:
            lines.append(f"\nROBUSTNESS: {rob.get('edges_to_flip','?')}/{rob.get('total_edges','?')} edges to flip")
        
        for d in advanced.get("do_calculus", [])[:3]:
            lines.append(f"CAUSAL POWER: {d['node_id']} '{d['content'][:30]}' → power={d['causal_power']:.4f}")
    
    return "\n".join(lines)


def _parse(text: str) -> dict:
    """Parse JSON from LLM response."""
    import re
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for p in [r'\{.*\}', r'\[.*\]']:
            m = re.search(p, text, re.DOTALL)
            if m:
                try: return json.loads(m.group())
                except: continue
        return {"parse_error": text[:200]}
