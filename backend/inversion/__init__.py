"""ChaosEngine Temporal Inversion — Drop into MiroFish fork.

Quick:
    from inversion import CausalDAG, CausalNode, invert, run_advanced_analysis
    dag = build_dag_from_mirofish(seed, spine, nerves, outcomes)
    result = invert(dag)           # 46ms, 0 LLM
    advanced = run_advanced_analysis(dag)  # 33ms, 0 LLM

Full loop (with agent verification):
    from inversion import full_verification_loop, invert, run_advanced_analysis
    result = await full_verification_loop(llm_call, dag, seed, invert, run_advanced_analysis)
"""

from .true_inversion import (
    CausalDAG, CausalNode, CausalEdge,
    run_full_inversion, build_dag_from_mirofish, invert,
)
from .advanced import run_advanced_analysis
from .verify_loop import full_verification_loop, verify_inversion_results, apply_adjustments_to_dag
from .validator import validate_dag_inputs, validate_results, interpret_results
