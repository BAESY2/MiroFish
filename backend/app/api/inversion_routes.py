from flask import Blueprint, request, jsonify
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from inversion import CausalDAG, CausalNode, invert, run_advanced_analysis

inversion_bp = Blueprint('inversion', __name__)

@inversion_bp.route('/inversion', methods=['POST'])
def inversion_endpoint():
    data = request.json or {}

    # Build DAG from request
    dag = CausalDAG()
    for node in data.get('nodes', []):
        dag.add_node(CausalNode(
            id=node['id'], content=node.get('content', ''),
            node_type=node.get('type', 'event'),
            prior_probability=node.get('probability', 0.5),
            time_offset_hours=node.get('time_hours', 0),
        ))
    for edge in data.get('edges', []):
        dag.add_edge(edge['from'], edge['to'],
                     edge.get('probability', 0.5),
                     edge.get('delay_hours', 0),
                     edge.get('gate', 'or'))

    # Run inversion
    result = invert(dag)
    advanced = run_advanced_analysis(dag)

    return jsonify({
        'inversion': result,
        'advanced': {
            'do_calculus': advanced.get('do_calculus', [])[:5],
            'critical_paths': advanced.get('critical_paths', {}),
            'markov_blankets': advanced.get('markov_blankets', {}),
            'robustness': advanced.get('robustness', {}),
            'kl_divergence': advanced.get('kl_divergence', {}),
        }
    })
