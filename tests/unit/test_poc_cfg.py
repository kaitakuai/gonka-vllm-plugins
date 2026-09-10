"""poc_cfg reads the one PoC knob under additional_config["gonka_poc"] and falls back
to the same default on every tree; unknown knobs are a programming error."""
from types import SimpleNamespace

import pytest

from gonka_poc.mixed.policy import POC_CONFIG_DEFAULTS, poc_cfg


def _vc(section=None, extra=None):
    ac = dict(extra or {})
    if section is not None:
        ac["gonka_poc"] = section
    return SimpleNamespace(additional_config=ac)


def test_defaults_when_absent():
    for name, default in POC_CONFIG_DEFAULTS.items():
        assert poc_cfg(None, name) == default
        assert poc_cfg(SimpleNamespace(), name) == default
        assert poc_cfg(_vc(), name) == default
        assert poc_cfg(_vc(extra={"other": {"poc_vector_artifacts": True}}), name) == default


def test_only_vector_artifacts_is_a_knob():
    assert set(POC_CONFIG_DEFAULTS) == {"poc_vector_artifacts"}


def test_bool_coercion():
    assert poc_cfg(_vc({"poc_vector_artifacts": "1"}), "poc_vector_artifacts") is True
    assert poc_cfg(_vc({"poc_vector_artifacts": "0"}), "poc_vector_artifacts") is False
    assert poc_cfg(_vc({"poc_vector_artifacts": False}), "poc_vector_artifacts") is False
    assert poc_cfg(_vc({"poc_vector_artifacts": 1}), "poc_vector_artifacts") is True


def test_unknown_knob_is_an_error():
    with pytest.raises(KeyError):
        poc_cfg(_vc(), "poc_max_batch_size")
