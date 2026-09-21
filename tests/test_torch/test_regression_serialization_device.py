"""serialization device regressions migrated from test_issue_regressions (#160).

Original test function names are retained for issue/history traceability.
"""

from ._regression_support import (
    YatNMN,
    torch,
)


def test_yat_nmn_device_constructor_covers_all_owned_state_and_default():
    default_layer = YatNMN(4, 2, learnable_epsilon=True)
    assert default_layer.weight.device.type == "cpu"
    assert all(
        value.device.type == "cpu" for value in default_layer.state_dict().values()
    )

    meta_layer = YatNMN(
        4, 2, learnable_epsilon=True, device="meta", param_dtype=torch.float64
    )
    state = meta_layer.state_dict()
    assert set(state) == {"weight", "alpha", "bias", "epsilon_param"}
    assert all(value.device.type == "meta" for value in state.values())
    assert all(value.dtype == torch.float64 for value in state.values())
