import pytest

from callscribe.synthetic import CallSpec, synth_call, write_dataset


@pytest.fixture(scope="session")
def call():
    return synth_call(CallSpec(n_speakers=3, n_turns=12, seed=11))


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    return write_dataset(out, n_calls=8, seed=4)
