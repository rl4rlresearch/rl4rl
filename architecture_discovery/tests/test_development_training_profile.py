from dataclasses import asdict

from common.training_config import (
    DEVELOPMENT_TRAIN_V1,
    FULL_COMPUTE_DEVELOPMENT_V1,
    FULL_TRAIN_V1,
    get_training_profile,
)


def test_development_training_profile_is_bounded_and_non_scientific():
    profile = get_training_profile("development_train_v1")
    assert profile is DEVELOPMENT_TRAIN_V1
    assert not profile.scientific
    assert profile.max_steps == 2_000
    assert profile.global_batch_size == 256
    assert profile.maximum_wall_seconds == 600
    assert profile.max_steps < get_training_profile("full_train_v1").max_steps


def test_full_compute_development_profile_matches_full_without_scientific_claim():
    profile = get_training_profile("full_compute_development_v1")
    assert profile is FULL_COMPUTE_DEVELOPMENT_V1
    assert not profile.scientific
    expected = asdict(FULL_TRAIN_V1)
    observed = asdict(profile)
    expected.update(name="full_compute_development_v1", scientific=False)
    assert observed == expected
