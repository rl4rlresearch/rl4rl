MECHANISM: Second orthogonal value/output basis gauge

HYPOTHESIS: A 1,376-parameter model will retain at least 99% accuracy because the qualified 1,377-parameter model reached 99.96%, and a second Givens rotation removes another exact value-basis redundancy while preserving its initialized function and full four-dimensional value stream.

INTENDED_EDIT: Extend the value-projection gauge to zero two coordinates of its first column, store one fewer scalar, and compensate both attention-output head blocks with the combined orthogonal rotation.

EVIDENCE: The current one-coordinate value/output basis gauge achieved 99.96% at 1,377 parameters, substantially outperforming prior 1,377-parameter reductions based on LayerNorm, attention-output, MLP, or query-key gauges; extending that successful exact symmetry by one coordinate is the most direct next test.

<<<<<<< SEARCH
class GaugeFixedValueProjection(nn.Module):
    """Value projection with one orthogonal basis gauge removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")
        self.in_features = in_features
        self.out_features = out_features
        self.first_column = nn.Parameter(
            torch.empty(out_features - 1)
        )
=======
class GaugeFixedValueProjection(nn.Module):
    """Value projection with two orthogonal basis gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if out_features < 3:
            raise ValueError("out_features must be at least three")
        self.in_features = in_features
        self.out_features = out_features
        self.first_column = nn.Parameter(
            torch.empty(out_features - 2)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pair = raw[:2, 0]
        radius = pair.square().sum().sqrt()
        rotation = torch.eye(
            self.out_features, device=raw.device, dtype=raw.dtype
        )
        if float(radius.item()) > 0.0:
            cosine = pair[0] / radius
            sine = pair[1] / radius
            rotation[0, 0] = cosine
            rotation[0, 1] = sine
            rotation[1, 0] = -sine
            rotation[1, 1] = cosine

        rotated = rotation @ raw
        self.first_column.copy_(
            torch.cat((rotated[:1, 0], rotated[2:, 0]))
        )
=======
        pair = raw[:2, 0]
        radius = pair.square().sum().sqrt()
        rotation = torch.eye(
            self.out_features, device=raw.device, dtype=raw.dtype
        )
        if float(radius.item()) > 0.0:
            cosine = pair[0] / radius
            sine = pair[1] / radius
            rotation[0, 0] = cosine
            rotation[0, 1] = sine
            rotation[1, 0] = -sine
            rotation[1, 1] = cosine

        triple_radius = (
            radius.square() + raw[2, 0].square()
        ).sqrt()
        if float(triple_radius.item()) > 0.0:
            second = torch.eye(
                self.out_features,
                device=raw.device,
                dtype=raw.dtype,
            )
            cosine = radius / triple_radius
            sine = raw[2, 0] / triple_radius
            second[0, 0] = cosine
            second[0, 2] = sine
            second[2, 0] = -sine
            second[2, 2] = cosine
            rotation = second @ rotation

        rotated = rotation @ raw
        self.first_column.copy_(
            torch.cat((rotated[:1, 0], rotated[3:, 0]))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_column = torch.cat(
            (
                self.first_column[:1],
                self.first_column.new_zeros(1),
                self.first_column[1:],
            )
        )
=======
        first_column = torch.cat(
            (
                self.first_column[:1],
                self.first_column.new_zeros(2),
                self.first_column[1:],
            )
        )
>>>>>>> REPLACE