from supermimic_deploy.contracts import (
    ACTION_DIM,
    GOAL_DIM,
    OBS_DIM,
)


def test_contract_dimensions():
    assert OBS_DIM == 930
    assert GOAL_DIM == 680
    assert ACTION_DIM == 29
