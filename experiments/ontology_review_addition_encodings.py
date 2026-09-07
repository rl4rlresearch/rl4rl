"""Source-bound semantic descriptors for directly inspected input encoders.

Token-ID permutations are separate from changing the represented units, radix,
column order or a constant start token into a data-conditioned query. Reference
matches retain every executable AST node and numeric constant. Only docstrings
and source locations are omitted; candidate Python is never executed.
"""
import ast
import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path


def encoding_key(method_ast):
    if not isinstance(method_ast, ast.FunctionDef) or method_ast.name != 'encode_inputs':
        return None
    node = copy.deepcopy(method_ast)
    if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
        node.body = node.body[1:]
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def references():
    path = Path(__file__).with_name('ontology_addition_encoding_references.json')
    rows = json.loads(path.read_text(encoding='utf-8'))['references']
    entries = {}
    for row in rows:
        key = encoding_key(ast.parse(row['source']).body[0])
        if key != row['ast_sha256']:
            raise ValueError('Encoding reference AST binding changed')
        value = row['descriptor']
        if key in entries:
            fields = set(value) - {'evidence'}
            if any(entries[key].get(k) != value.get(k) for k in fields):
                raise ValueError('Conflicting direct descriptions of an encoding')
        else:
            entries[key] = value
    return entries


def encoding_descriptor(method_ast):
    """Return a complete encoder descriptor only for a source-audited body."""
    value = references().get(encoding_key(method_ast))
    return copy.deepcopy(value) if value is not None else None
