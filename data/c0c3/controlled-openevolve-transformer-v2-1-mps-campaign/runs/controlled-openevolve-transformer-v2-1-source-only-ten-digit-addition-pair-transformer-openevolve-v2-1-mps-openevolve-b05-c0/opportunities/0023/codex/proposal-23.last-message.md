MECHANISM: Third quotient-aware key gauge localized to the second attention head

HYPOTHESIS: Removing the softmax-invisible coordinate from the second key row of the second head will reduce the model from 1631 to 1630 parameters while retaining at least 99% accuracy; unlike the failed two-row first-head configuration, this tests whether within-head gauge sensitivity is head-specific while preserving the successful cross-head gauges.

INTENDED_EDIT: Add the second key row of the second head to `GaugeFixedQKV.fixed_rows` and generalize virtual AdamW state allocation to the number of omitted coordinates.

EVIDENCE: Removing the first key row from each head achieved 99.93% at 1631 parameters, whereas removing the first two key rows from the first head achieved only 87.60%; placing the necessary third gauge in the second head is the closest untested titration.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """QKV projection with one softmax-invisible coordinate per head removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (d_model, d_model + second_offset)
=======
class GaugeFixedQKV(nn.Module):
    """QKV projection with three softmax-invisible coordinates removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (
            d_model,
            d_model + second_offset,
            d_model + second_offset + 1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.state = {
            parameter: {
                "step": 0,
                "exp_avg": torch.zeros(
                    parameter.numel() + 2,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + 2,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            }
            for parameter in self.parameters
        }
=======
        self.state = {
            parameter: {
                "step": 0,
                "exp_avg": torch.zeros(
                    parameter.numel() + len(fixed_rows),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + len(fixed_rows),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            }
            for parameter, _, _, fixed_rows in self.gauges
        }
>>>>>>> REPLACE