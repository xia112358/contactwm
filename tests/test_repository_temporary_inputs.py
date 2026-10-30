"""Repository boundaries distinguish temporary outputs from required inputs."""
import ast
import pytest
from scripts.check_repository import configuration_temporary_inputs, temporary_input_errors


def test_release_notes_retain_unavailable_historical_paths_without_using_them_as_inputs():
    catalog = dict(schema='somaforge_data_release_catalog_v1', components={
        'demo': dict(index={'path': 'runtime/current/archive/index.json'},
                     provenance_notes=[{'historical_or_unavailable_embedded_references': ['tmp/old/source.npz']}])})
    assert not configuration_temporary_inputs(catalog)
    catalog['components']['demo']['index']['path'] = 'tmp/required/index.json'
    assert configuration_temporary_inputs(catalog) == ['tmp/required/index.json']


@pytest.mark.parametrize('document', [
    {'provenance_notes': ['tmp/required/source.json']},
    {'schema': 'another_schema', 'components': {'demo': {'provenance_notes': ['tmp/required/source.json']}}},
    {'schema': 'somaforge_data_release_catalog_v1', 'provenance_notes': ['tmp/required/source.json']},
])
def test_only_release_component_notes_are_nonoperational_metadata(document):
    assert configuration_temporary_inputs(document) == ['tmp/required/source.json']


@pytest.mark.parametrize('source', [
    "sys.path.insert(0, ROOT / 'tmp')",
    "spec_from_file_location('helper', ROOT / 'tmp' / 'helper.py')",
    "data = torch.load('tmp/weights.pt')",
    "source = Path('tmp/source.json'); data = source.read_text()",
    "MANIFEST = 'tmp/dataset.json'",
    "cache = ROOT / 'tmp' / 'compiled'",
])
def test_required_temporary_inputs_are_rejected(source):
    assert temporary_input_errors('fixture.py', ast.parse(source))


@pytest.mark.parametrize('source', [
    "output = ROOT / 'tmp' / 'diagnostic.json'; output.write_text(report)",
    "cache = ROOT / 'runtime/current/cache' / 'compiled'",
    "source = Path('runtime/current/motions/source.json'); data = source.read_text()",
])
def test_temporary_outputs_and_formal_inputs_are_allowed(source):
    assert not temporary_input_errors('fixture.py', ast.parse(source))


@pytest.mark.parametrize('read', [
    "manifest = root / 'manifest.json'; manifest.read_text()",
    "checkpoint = root / 'checkpoint.pt'; torch.load(checkpoint)",
    "np.load(root / 'data.npz')",
    "(root / 'data.bin').read_bytes()",
])
def test_fresh_test_directory_allows_generated_data_round_trip(read):
    source = """
def test_round_trip():
    root = Path('tmp/round_trip') / uuid.uuid4().hex
    root.mkdir(parents=True)
""" + '\n'.join('    ' + line for line in read.splitlines())
    assert not temporary_input_errors('tests/test_round_trip.py', ast.parse(source))


def test_fresh_fixture_paths_propagate_to_consumers_and_nested_helpers():
    source = """
def test_round_trip(run_fixture):
    root, path = run_fixture
    manifest = path
    manifest.read_text()
    def helper():
        np.load(root / 'data.npz')

@pytest.fixture()
def run_fixture():
    root = Path('tmp/round_trip') / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    path = root / 'manifest.json'
    return root, path
"""
    assert not temporary_input_errors('tests/test_round_trip.py', ast.parse(source))


@pytest.mark.parametrize('setup', [
    "root = Path('tmp/old')",
    "root = Path('tmp/round_trip') / uuid.uuid4().hex",
    "root = Path('tmp/round_trip') / uuid.uuid4().hex; root.mkdir(exist_ok=True)",
    "root = Path('tmp/round_trip') / uuid.uuid4().hex; root.mkdir(**options)",
])
def test_test_filename_does_not_exempt_existing_or_unproven_data(setup):
    source = f'def test_read():\n    {setup}\n    np.load(root / "data.npz")'
    assert temporary_input_errors('tests/test_round_trip.py', ast.parse(source))


@pytest.mark.parametrize('read', [
    "np.load('tmp/old/data.npz')",
    "path = Path('tmp/old/data.json'); path.read_text()",
    "np.load(root / '..' / 'old.npz')",
    "np.load(root / '/tmp/old.npz')",
    "sys.path.insert(0, root / 'tmp')",
    "sys.path.insert(0, root / 'helpers')",
    "spec_from_file_location('helper', root / 'tmp/helper.py')",
    "spec_from_file_location('helper', root / 'helper.py')",
])
def test_fresh_test_directory_does_not_exempt_other_inputs_or_code(read):
    source = f"""
def test_read():
    root = Path('tmp/round_trip') / uuid.uuid4().hex
    root.mkdir(parents=True)
    {read}
"""
    assert temporary_input_errors('tests/test_round_trip.py', ast.parse(source))


def test_production_code_cannot_exempt_inputs_by_creating_a_uuid_directory():
    source = """
root = Path('tmp/round_trip') / uuid.uuid4().hex
root.mkdir(parents=True)
np.load(root / 'data.npz')
"""
    assert temporary_input_errors('scripts/collect.py', ast.parse(source))


def test_generated_path_bindings_do_not_leak_between_functions():
    source = """
def test_generated():
    root = Path('tmp/round_trip') / uuid.uuid4().hex
    root.mkdir(parents=True)
    np.load(root / 'data.npz')

def test_existing():
    root = Path('tmp/old')
    np.load(root / 'data.npz')
"""
    errors = temporary_input_errors('tests/test_round_trip.py', ast.parse(source))
    assert len(errors) == 1 and ':9:' in errors[0]


@pytest.mark.parametrize('body', [
    "if condition:\n        root.mkdir(parents=True)",
    "root.mkdir(parents=True)\n    if condition:\n        root = Path('tmp/old')",
])
def test_conditional_creation_or_rebinding_does_not_prove_a_fresh_path(body):
    source = f"""
def test_read():
    root = Path('tmp/round_trip') / uuid.uuid4().hex
    {body}
    np.load(root / 'data.npz')
"""
    assert temporary_input_errors('tests/test_round_trip.py', ast.parse(source))


@pytest.mark.parametrize('component,allowed', [("'a'", True), ("'../old'", False), ("'/tmp/old'", False)])
@pytest.mark.parametrize('path', ["root / name / 'data.npz'", "root / f'{name}/data.npz'"])
def test_loop_components_must_stay_within_the_fresh_directory(component, allowed, path):
    source = f"""
def test_read():
    root = Path('tmp/round_trip') / uuid.uuid4().hex
    root.mkdir(parents=True)
    for name in ({component},):
        np.load({path})
"""
    assert bool(temporary_input_errors('tests/test_round_trip.py', ast.parse(source))) is not allowed
