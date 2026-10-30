"""Read-only repository audit; optionally write a JSON report under project tmp/."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RETIRED_SUPPORT_PATHS = (
    'scripts/aggregate_force_support.py',
    'scripts/audit_edited_support.py',
    'packages/somaforge_core/somaforge_core/support_aggregation.py',
    'packages/somaforge_core/tests/test_support_aggregation.py',
)
RETIRED_ENTRYPOINT_PATHS = (
    'scripts/train_climb00_direct_baseline.py',
    'packages/motion_edit/motion_edit/workbench/stable_transition_rebuild.py',
    'packages/motion_edit/motion_edit/generation/newton_collision.py',
    'packages/motion_edit/motion_edit/generation/newton_collision_filter.py',
)
CANONICAL_ENVIRONMENT_PATHS = (
    'scripts/source_somaforge.sh',
    'scripts/setup_isaaclab3_newton.sh',
    'scripts/extract_all_rollout_ref_contact_force_demos.py',
    'packages/motion_edit/motion_edit/physics_retarget/newton_runner.py',
)


def generated_test_paths(name: str, tree: ast.AST) -> set[int]:
    """Recognize paths owned by fresh test directories, including fixture returns.

    Only a UUID-named directory created without exist_ok is considered fresh.
    A test filename alone never exempts reads of pre-existing temporary inputs.
    """
    path = Path(name)
    if 'tests' not in path.parts or not path.name.startswith('test_'):
        return set()
    generated = set()
    fixtures = {}

    class Paths(ast.NodeVisitor):
        def __init__(self, bindings=None):
            self.bindings = dict(bindings or {})
            self.pending = set()
            self.returns = []

        def value(self, node):
            if isinstance(node, ast.Name):
                return self.bindings.get(node.id, False)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if not Path(node.value).is_absolute() and '..' not in Path(node.value).parts:
                    return frozenset([node.value])
            if isinstance(node, (ast.Tuple, ast.List)):
                return tuple(self.value(item) for item in node.elts)
            if isinstance(node, ast.JoinedStr):
                choices = {''}
                for part in node.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        values = frozenset([part.value])
                    elif isinstance(part, ast.FormattedValue) and part.format_spec is None and part.conversion == -1:
                        values = self.value(part.value)
                    else:
                        return False
                    if not isinstance(values, frozenset):
                        return False
                    choices = {prefix + suffix for prefix in choices for suffix in values}
                    if len(choices) > 64:
                        return False
                if all(not Path(value).is_absolute() and '..' not in Path(value).parts for value in choices):
                    return frozenset(choices)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if self.value(node.left) is True and isinstance(self.value(node.right), frozenset):
                    return True
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'Path' and len(node.args) == 1:
                    return self.value(node.args[0])
                if isinstance(node.func, ast.Attribute) and node.func.attr in ('resolve', 'absolute', 'expanduser', 'with_suffix'):
                    return self.value(node.func.value) is True
            return False

        def visit(self, node):
            if self.value(node) is True:
                generated.add(id(node))
            return super().visit(node)

        def bind(self, target, value):
            if isinstance(target, ast.Name):
                self.bindings[target.id] = value
                self.pending.discard(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                values = value if isinstance(value, tuple) and len(value) == len(target.elts) else (False,) * len(target.elts)
                for item, state in zip(target.elts, values):
                    self.bind(item, state)

        def assignment(self, targets, value):
            self.visit(value)
            for target in targets:
                self.bind(target, self.value(value))
                if (isinstance(target, ast.Name) and isinstance(value, ast.BinOp)
                        and isinstance(value.op, ast.Div)
                        and isinstance(value.right, ast.Attribute) and value.right.attr == 'hex'
                        and isinstance(value.right.value, ast.Call)
                        and ast.unparse(value.right.value.func) == 'uuid.uuid4'):
                    self.pending.add(target.id)

        def visit_Assign(self, node):
            self.assignment(node.targets, node.value)

        def visit_AnnAssign(self, node):
            if node.value is not None:
                self.assignment([node.target], node.value)

        def visit_Expr(self, node):
            self.generic_visit(node)
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == 'mkdir' and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in self.pending and not call.args
                    and all(k.arg in ('parents', 'mode') or
                            (k.arg == 'exist_ok' and isinstance(k.value, ast.Constant) and k.value.value is False)
                            for k in call.keywords)):
                self.bindings[call.func.value.id] = True
                self.pending.discard(call.func.value.id)

        def visit_branches(self, node):
            # Creation in one branch cannot prove ownership after that branch.
            for _, field in ast.iter_fields(node):
                child = Paths(self.bindings)
                child.pending = self.pending.copy()
                for item in field if isinstance(field, list) else [field]:
                    if isinstance(item, ast.AST):
                        child.visit(item)
            for item in ast.walk(node):
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                    self.bind(item, False)

        visit_If = visit_branches
        visit_While = visit_branches
        visit_Try = visit_branches
        visit_TryStar = visit_branches
        visit_Match = visit_branches

        def visit_For(self, node):
            self.visit(node.iter)
            child = Paths(self.bindings)
            child.pending = self.pending.copy()
            values = self.value(node.iter)
            child.bind(node.target, frozenset().union(*values) if isinstance(values, tuple) and values
                       and all(isinstance(value, frozenset) for value in values) else False)
            for statement in node.body:
                child.visit(statement)
            other = Paths(self.bindings)
            for statement in node.orelse:
                other.visit(statement)
            for item in ast.walk(node):
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                    self.bind(item, False)

        visit_AsyncFor = visit_For

        def visit_FunctionDef(self, node):
            child = Paths(self.bindings)
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            for argument in arguments:
                child.bindings[argument.arg] = fixtures.get(argument.arg, False)
            for statement in node.body:
                child.visit(statement)
            decorators = [d.func if isinstance(d, ast.Call) else d for d in node.decorator_list]
            if any(ast.unparse(d) == 'pytest.fixture' for d in decorators) and child.returns:
                if all(value == child.returns[0] for value in child.returns):
                    fixtures[node.name] = child.returns[0]

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Return(self, node):
            if node.value is not None:
                self.visit(node.value)
                self.returns.append(self.value(node.value))

    # Fixtures may be declared after their consumers. Revisit with known returns.
    Paths().visit(tree)
    generated.clear()
    Paths().visit(tree)
    return generated


def temporary_input_errors(name: str, tree: ast.AST) -> list[str]:
    """Reject temporary code imports and file reads; diagnostic writes are allowed."""
    errors = []
    generated = generated_test_paths(name, tree)
    def unowned_nodes(node):
        if id(node) in generated:
            return
        yield node
        for child in ast.iter_child_nodes(node):
            yield from unowned_nodes(child)
    def temporary_literal(node):
        return (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and (node.value == 'tmp' or node.value.startswith('tmp/') or '/holosoma/tmp/' in node.value))
    def unproven_generated_path(node, *, literal_escape=False):
        for item in unowned_nodes(node):
            if isinstance(item, ast.BinOp) and isinstance(item.op, ast.Div) and id(item.left) in generated:
                if not literal_escape or any(
                    isinstance(part, ast.Constant) and isinstance(part.value, str)
                    and (Path(part.value).is_absolute() or '..' in Path(part.value).parts)
                    for part in ast.walk(item.right)
                ):
                    return True
        return False
    def temporary_path(node):
        return (any(temporary_literal(n) for n in unowned_nodes(node))
                or unproven_generated_path(node, literal_escape=True))
    def call_name(node):
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute): return call_name(node.value) + '.' + node.attr
        return ''
    temporary_names = set()
    for _ in range(4):
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)): continue
            value = node.value
            if value is None: continue
            if temporary_path(value) or any(isinstance(n, ast.Name) and n.id in temporary_names for n in unowned_nodes(value)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                temporary_names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
    def temporary_input(node):
        return id(node) not in generated and (
            temporary_path(node) or unproven_generated_path(node)
            or any(isinstance(n, ast.Name) and n.id in temporary_names for n in unowned_nodes(node))
        )
    def temporary_code_path(node):
        return any(temporary_literal(n) or (isinstance(n, ast.Name) and n.id in temporary_names)
                   for n in ast.walk(node))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None and temporary_input(node.value):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            if any(n.lower().endswith(('manifest', 'corpus', 'checkpoint', 'cache')) for n in names):
                errors.append(f'{name}:{node.lineno}: persistent input/cache assigned a temporary path')
        if not isinstance(node, ast.Call): continue
        function = call_name(node.func)
        if function in ('sys.path.insert', 'sys.path.append') and temporary_code_path(node):
            errors.append(f'{name}:{node.lineno}: temporary directory on import path')
        if function.endswith('spec_from_file_location') and temporary_code_path(node):
            errors.append(f'{name}:{node.lineno}: import of temporary source file')
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('read_text', 'read_bytes'):
            if temporary_input(node.func.value):
                errors.append(f'{name}:{node.lineno}: temporary file used as required input')
        if function in ('torch.load', 'np.load', 'numpy.load') and node.args and temporary_input(node.args[0]):
            errors.append(f'{name}:{node.lineno}: temporary data/model used as required input')
    return errors


def configuration_temporary_inputs(document):
    """Inspect inputs while retaining explicitly non-operational release notes."""
    release = isinstance(document, dict) and document.get('schema') == 'somaforge_data_release_catalog_v1'
    def inputs(value, key='', trail=()):
        if (release and len(trail) == 3 and trail[0] == 'components'
                and trail[2] == 'provenance_notes'):
            return
        if isinstance(value, dict):
            for name, item in value.items():
                yield from inputs(item, name, (*trail, name))
        elif isinstance(value, list):
            for item in value:
                yield from inputs(item, key, trail)
        elif isinstance(value, str) and value.startswith('tmp/') and key not in ('output', 'log'):
            yield value
    return sorted(set(inputs(document)))


def audit(root: Path = ROOT) -> dict:
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).splitlines()

    tracked = git('ls-files')
    paths = sorted(set(tracked + git('ls-files', '--others', '--exclude-standard')))
    errors, warnings = [], []
    for name in RETIRED_SUPPORT_PATHS:
        if (root / name).exists():
            errors.append(f'{name}: retired support judgment/aggregation path; use native support evidence')
    for name in RETIRED_ENTRYPOINT_PATHS:
        if (root / name).exists():
            errors.append(f'{name}: retired entrypoint; use the owning package or historical records')
    for name in CANONICAL_ENVIRONMENT_PATHS:
        path = root / name
        if path.is_file() and 'packages/climb00_pipeline' in path.read_text():
            errors.append(f'{name}: default environment must not add the historical compatibility package')
    binaries, temporary_references = [], []
    python_count = 0
    for name in paths:
        path = root / name
        if not path.is_file():
            continue  # staged/working-tree deletions are reported by git status
        if path.suffix in {'.npz', '.pt', '.pth', '.ckpt', '.pyc'}:
            binaries.append(dict(path=name, bytes=path.stat().st_size))
        if path.suffix == '.py' and name.startswith(('packages/', 'scripts/', 'src/', 'tests/')):
            python_count += 1
            try:
                tree = ast.parse(path.read_text(), filename=name)
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f'{name}: {exc}')
                continue
            errors.extend(temporary_input_errors(name, tree))
            for node in ast.walk(tree):
                modules = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                           [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                if any(m == 'somaforge_core.support_aggregation' for m in modules):
                    errors.append(f'{name}:{node.lineno}: retired support aggregation import')
                retired_newton_import = any(m in ('motion_edit.generation.newton_collision',
                                                  'motion_edit.generation.newton_collision_filter') for m in modules)
                if isinstance(node, ast.ImportFrom) and node.level and name.startswith('packages/motion_edit/motion_edit/generation/'):
                    retired_newton_import |= node.module in ('newton_collision', 'newton_collision_filter')
                if retired_newton_import:
                    errors.append(f'{name}:{node.lineno}: shared Newton geometry belongs to somaforge_core')
                if name.startswith('packages/contact_solver/contact_solver/'):
                    if any(m == 'motion_edit' or m.startswith('motion_edit.') for m in modules):
                        errors.append(f'{name}:{node.lineno}: solver must not import the editing application')
                if name.startswith(('packages/motion_edit/', 'packages/contact_solver/')):
                    if any(m == 'viser' or m.startswith('viser.') or m == 'viser_utils' for m in modules):
                        errors.append(f'{name}:{node.lineno}: retired Viser UI; use motion-edit contact-editor')
            # Canonical implementations must not depend on their historical aliases.
            if name.startswith(('packages/generator/generator/', 'packages/contact_solver/contact_solver/',
                                'packages/somaforge_core/somaforge_core/')):
                for node in ast.walk(tree):
                    modules = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                               [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                    if any(m == 'climb00_pipeline' or m.startswith('climb00_pipeline.') for m in modules):
                        errors.append(f'{name}:{node.lineno}: production import through historical alias')
        if path.suffix == '.json' and name.startswith('configs/'):
            try:
                document = json.loads(path.read_text())
            except (ValueError, UnicodeError) as exc:
                errors.append(f'{name}: {exc}')
                continue
            refs = configuration_temporary_inputs(document)
            for p in refs:
                temporary_references.append(dict(config=name, path=p, exists=(root/p).exists()))
                errors.append(f'{name}: temporary path used as configuration input: {p}')
        if path.suffix in {'.yaml', '.yml'} and name.startswith('.github/workflows/'):
            for script in re.findall(r'\bpython(?:3)?\s+(scripts/[\w./-]+\.py)\b', path.read_text()):
                if not (root / script).is_file():
                    errors.append(f'{name}: workflow calls missing script {script}')
    for path in root.iterdir():
        if path.is_symlink() and not path.exists():
            errors.append(f'Broken root symlink: {path.name}')
    if binaries:
        warnings.append('Tracked data/examples require dependency review; never blanket-delete them.')
    status = git('status', '--porcelain', '--untracked-files=all')
    return dict(schema='somaforge_repository_audit_v1', root=str(root),
                tracked_files=len(tracked), python_files_parsed=python_count,
                changes=dict(Counter(line[:2] for line in status)),
                tracked_data=binaries, temporary_config_references=temporary_references,
                errors=errors, warnings=warnings,
                scope='Static source/config/import audit; not physics, training or data acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to((ROOT/'tmp').resolve()):
            raise ValueError('Diagnostic reports must be stored under project tmp/')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return bool(report['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
