"""Agent Verification Loop — Adversarial debate on inversion results.

FLOW:
  Math inversion (46ms, 0 LLM)
    → Agent verification (N turns, LLM calls)
      → Surviving findings fed back into DAG
        → Re-run math inversion with refined probabilities
          → Repeat until convergence

AGENTS:
  PROSECUTOR  — Tries to disprove every finding. "This can't be right because..."
  DEFENDER    — Supports findings with evidence. "This holds because..."
  DEVIL       — Finds edge cases and black swans. "But what if..."
  JUDGE       — Evaluates arguments, adjusts confidence scores.

Each necessity/turnstile/sensitivity finding goes through adversarial debate.
Only findings that survive get confidence boost.
Killed findings get probability adjusted in the DAG → re-inversion.

This is NOT asking LLM "is this true?"
This is N agents ARGUING about it for hundreds of turns until consensus.
"""

import asyncio
import json
import random
from typing import Callable, Optional
from dataclasses import dataclass


PROSECUTOR_SYSTEM = """You are a ruthless prosecutor. Your ONLY job is to DISPROVE the claim.
Find logical flaws, missing evidence, historical counterexamples.
If you can't disprove it, say so honestly — but TRY HARD.
Be specific. Name exact counterexamples or logical gaps."""

DEFENDER_SYSTEM = """You are a passionate defender. Your job is to SUPPORT the claim.
Find supporting evidence, historical precedents, logical arguments.
Address the prosecutor's attacks directly. Don't be vague — be specific."""

DEVIL_SYSTEM = """You are the devil's advocate. Find EDGE CASES the others missed.
What unlikely scenario breaks this? What assumption is everyone taking for granted?
You're not for or against — you're finding what everyone else overlooked."""

JUDGE_SYSTEM = """You are an impartial judge evaluating a debate about a causal claim.
Based on the arguments presented, score the claim's survival probability.
Be precise. Consider the strength of attacks AND defenses.
Output ONLY a JSON verdict."""


@dataclass
class DebateRound:
    round_num: int
    claim: str
    prosecutor_arg: str
    defender_arg: str
    devil_arg: str
    judge_score: float  # 0.0 = disproven, 1.0 = fully supported


@dataclass
class VerificationResult:
    claim: str
    claim_type: str  # necessity | turnstile | sensitivity | counterfactual
    node_id: str
    initial_score: float
    final_score: float
    survived: bool
    rounds: list[DebateRound]
    adjustment: float  # How much to adjust DAG probability


async def debate_claim(
    llm_call: Callable,
    claim: str,
    context: str,
    num_rounds: int = 3,
) -> list[DebateRound]:
    """Run adversarial debate on a single claim for N rounds.

    Each round: Prosecutor attacks → Defender responds → Devil adds edge case → Judge scores.
    Later rounds reference earlier arguments (memory).
    """
    rounds = []
    history = ""

    for r in range(num_rounds):
        round_context = f"""CLAIM: {claim}
SCENARIO CONTEXT: {context}
{"PREVIOUS DEBATE:" + chr(10) + history if history else "This is the first round."}
ROUND {r+1}/{num_rounds}"""

        # Prosecutor attacks
        prosecutor_prompt = f"""{round_context}

ATTACK this claim. Find the strongest argument against it.
{"Build on previous attacks — find NEW angles." if r > 0 else "Start with the most obvious weakness."}
Respond in 2-3 sentences. Be specific."""

        prosecutor_arg = await llm_call(prosecutor_prompt, PROSECUTOR_SYSTEM)

        # Defender responds
        defender_prompt = f"""{round_context}

PROSECUTOR'S ATTACK: {prosecutor_arg}

DEFEND the claim against this specific attack.
{"Address the NEW attack, don't repeat old defenses." if r > 0 else ""}
Respond in 2-3 sentences. Cite specific evidence or logic."""

        defender_arg = await llm_call(defender_prompt, DEFENDER_SYSTEM)

        # Devil's advocate
        devil_prompt = f"""{round_context}

PROSECUTOR: {prosecutor_arg}
DEFENDER: {defender_arg}

Find an EDGE CASE or ASSUMPTION both sides missed.
What breaks this claim in a way neither discussed?
Respond in 1-2 sentences."""

        devil_arg = await llm_call(devil_prompt, DEVIL_SYSTEM)

        # Judge scores
        judge_prompt = f"""{round_context}

PROSECUTOR: {prosecutor_arg}
DEFENDER: {defender_arg}
DEVIL'S ADVOCATE: {devil_arg}

Score this claim's survival probability based on the debate so far.

Return JSON only:
{{"score": 0.0-1.0, "reasoning": "one sentence", "strongest_attack": "...", "strongest_defense": "..."}}"""

        judge_text = await llm_call(judge_prompt, JUDGE_SYSTEM)
        try:
            judge_data = json.loads(judge_text.replace('```json','').replace('```','').strip())
        except:
            import re
            m = re.search(r'\{.*\}', judge_text, re.DOTALL)
            judge_data = json.loads(m.group()) if m else {"score": 0.5}

        score = judge_data.get("score", 0.5)

        debate_round = DebateRound(
            round_num=r + 1,
            claim=claim,
            prosecutor_arg=prosecutor_arg.strip(),
            defender_arg=defender_arg.strip(),
            devil_arg=devil_arg.strip(),
            judge_score=score,
        )
        rounds.append(debate_round)

        # Build history for next round
        history += f"\nRound {r+1}: P:{prosecutor_arg[:100]} | D:{defender_arg[:100]} | Score:{score:.2f}"

        # Early termination: if score drops below 0.15 or above 0.9, consensus reached
        if score < 0.15 or score > 0.9:
            break

    return rounds


async def verify_finding(
    llm_call: Callable,
    claim: str,
    claim_type: str,
    node_id: str,
    initial_score: float,
    context: str,
    num_rounds: int = 3,
) -> VerificationResult:
    """Verify a single finding through adversarial debate."""

    rounds = await debate_claim(llm_call, claim, context, num_rounds)

    # Final score = average of all judge scores (weighted toward later rounds)
    if rounds:
        weights = [1.0 + i * 0.5 for i in range(len(rounds))]  # Later rounds weigh more
        total_w = sum(weights)
        final_score = sum(r.judge_score * w for r, w in zip(rounds, weights)) / total_w
    else:
        final_score = initial_score

    survived = final_score > 0.35
    adjustment = final_score - initial_score  # How much to adjust

    return VerificationResult(
        claim=claim,
        claim_type=claim_type,
        node_id=node_id,
        initial_score=initial_score,
        final_score=round(final_score, 4),
        survived=survived,
        rounds=rounds,
        adjustment=round(adjustment, 4),
    )


async def verify_inversion_results(
    llm_call: Callable,
    seed_text: str,
    inversion_result: dict,
    advanced_result: dict = None,
    rounds_per_claim: int = 3,
    max_claims: int = 10,
) -> dict:
    """Verify all key findings from inversion through adversarial debate.

    Extracts claims from:
      - Necessity analysis (critical conditions)
      - Turnstile
      - Sensitivity (most sensitive edges)
      - Counterfactuals (if advanced results provided)

    Returns verification results + DAG adjustment recommendations.
    """

    context = f"Scenario: {seed_text[:200]}"
    claims_to_verify = []

    # Extract claims from necessity
    for lid, nd in inversion_result.get("necessities", {}).items():
        for cond in nd.get("conditions", []):
            if cond.get("critical") and not cond.get("trivial"):
                claims_to_verify.append({
                    "claim": f"'{cond['content']}' is a CRITICAL precondition for '{nd['outcome']}'."
                             f" Necessity score: {cond['necessity']}",
                    "type": "necessity",
                    "node_id": cond["node_id"],
                    "initial_score": cond["necessity"],
                })

    # Turnstile claim
    ts = inversion_result.get("turnstile", {})
    if ts.get("found"):
        claims_to_verify.append({
            "claim": f"The point of no return is '{ts['content']}' at {ts.get('time_hours',0)} hours. "
                     f"Once this happens, the outcome becomes inevitable.",
            "type": "turnstile",
            "node_id": ts["node_id"],
            "initial_score": ts.get("necessity", 0.5),
        })

    # Sensitivity claims
    for lid, sens_list in inversion_result.get("sensitivity", {}).items():
        if sens_list:
            top = sens_list[0]
            claims_to_verify.append({
                "claim": f"The edge '{top['edge']}' is the most sensitive factor for "
                         f"the outcome. Changing it by 10% shifts probability by {abs(top.get('up_effect',0)):.4f}.",
                "type": "sensitivity",
                "node_id": top["edge"].split("→")[0],
                "initial_score": top["sensitivity"],
            })

    # Counterfactual claims (from advanced)
    if advanced_result:
        for cf in advanced_result.get("counterfactuals", [])[:3]:
            if cf.get("is_structurally_critical"):
                claims_to_verify.append({
                    "claim": f"Removing '{cf['content']}' would change outcomes by {cf['total_impact']:.4f}. "
                             f"It's structurally critical to the causal network.",
                    "type": "counterfactual",
                    "node_id": cf["removed_node"],
                    "initial_score": cf["total_impact"],
                })

    # Limit claims
    claims_to_verify = claims_to_verify[:max_claims]

    # Run verification in parallel
    async def _verify(c):
        return await verify_finding(
            llm_call, c["claim"], c["type"], c["node_id"],
            c["initial_score"], context, rounds_per_claim
        )

    results = await asyncio.gather(*[_verify(c) for c in claims_to_verify])

    # Compile
    verified = [r for r in results if r.survived]
    killed = [r for r in results if not r.survived]

    # DAG adjustments: for killed findings, reduce the node's probability
    adjustments = []
    for r in results:
        if r.adjustment != 0:
            adjustments.append({
                "node_id": r.node_id,
                "claim_type": r.claim_type,
                "adjustment": r.adjustment,
                "new_score": r.final_score,
                "survived": r.survived,
            })

    total_rounds = sum(len(r.rounds) for r in results)
    total_llm_calls = total_rounds * 4  # prosecutor + defender + devil + judge

    return {
        "total_claims": len(results),
        "survived": len(verified),
        "killed": len(killed),
        "survival_rate": round(len(verified) / max(1, len(results)), 4),
        "total_debate_rounds": total_rounds,
        "total_llm_calls": total_llm_calls,
        "adjustments": adjustments,
        "verified_findings": [
            {"claim": r.claim, "type": r.claim_type, "node_id": r.node_id,
             "initial": r.initial_score, "final": r.final_score}
            for r in verified
        ],
        "killed_findings": [
            {"claim": r.claim, "type": r.claim_type, "node_id": r.node_id,
             "initial": r.initial_score, "final": r.final_score,
             "killing_argument": r.rounds[-1].prosecutor_arg if r.rounds else ""}
            for r in killed
        ],
    }


def apply_adjustments_to_dag(dag, adjustments: list[dict]) -> int:
    """Apply verification adjustments back to the DAG.

    Killed findings → reduce node's prior probability.
    Survived findings → boost node's prior probability.

    Returns number of adjustments applied.
    """
    applied = 0
    for adj in adjustments:
        nid = adj["node_id"]
        if nid in dag.nodes:
            node = dag.nodes[nid]
            if adj["survived"]:
                # Boost: survived debate → increase confidence
                node.prior_probability = min(0.99, node.prior_probability * 1.1)
            else:
                # Kill: failed debate → decrease confidence
                node.prior_probability = max(0.01, node.prior_probability * 0.6)
            applied += 1
    return applied


async def full_verification_loop(
    llm_call: Callable,
    dag,
    seed_text: str,
    inversion_fn,
    advanced_fn=None,
    max_iterations: int = 3,
    rounds_per_claim: int = 3,
    convergence_threshold: float = 0.02,
) -> dict:
    """Full loop: Invert → Verify → Adjust → Re-invert → until convergence.

    Args:
        dag: CausalDAG
        inversion_fn: function(dag) -> inversion_result
        advanced_fn: optional function(dag) -> advanced_result
        max_iterations: max feedback loops
        convergence_threshold: stop if adjustments are smaller than this

    Returns:
        Final inversion result + all verification history
    """
    history = []

    for iteration in range(max_iterations):
        # Step 1: Run math inversion
        inv_result = inversion_fn(dag)
        adv_result = advanced_fn(dag) if advanced_fn else None

        # Step 2: Agent verification
        verification = await verify_inversion_results(
            llm_call, seed_text, inv_result, adv_result,
            rounds_per_claim=rounds_per_claim,
        )

        # Step 3: Apply adjustments
        adjustments = verification.get("adjustments", [])
        n_applied = apply_adjustments_to_dag(dag, adjustments)

        # Track
        max_adj = max((abs(a["adjustment"]) for a in adjustments), default=0)
        history.append({
            "iteration": iteration + 1,
            "claims_verified": verification["total_claims"],
            "survived": verification["survived"],
            "killed": verification["killed"],
            "adjustments_applied": n_applied,
            "max_adjustment": round(max_adj, 4),
            "llm_calls": verification["total_llm_calls"],
        })

        # Step 4: Check convergence
        if max_adj < convergence_threshold:
            break

    # Final inversion with adjusted DAG
    final_inv = inversion_fn(dag)
    final_adv = advanced_fn(dag) if advanced_fn else None

    return {
        "final_inversion": final_inv,
        "final_advanced": final_adv,
        "iterations": len(history),
        "converged": len(history) < max_iterations or history[-1]["max_adjustment"] < convergence_threshold,
        "history": history,
        "total_llm_calls": sum(h["llm_calls"] for h in history),
        "total_debate_rounds": sum(h["claims_verified"] * rounds_per_claim for h in history),
    }
