MECHANISM: Third orthogonal value/output basis gauge

HYPOTHESIS: A 1,375-parameter model will retain at least 99% accuracy because the qualified 1,376-parameter model reached 99.88%, and a third Givens rotation removes one more exact value-basis redundancy while preserving the initialized function and full four-dimensional value stream.

INTENDED_EDIT: Extend the value-projection gauge to align its first input column with a single value coordinate, store one fewer scalar, and compensate both attention-output head blocks with the combined orthogonal rotation.

EVIDENCE: Successive value/output basis gauges achieved 99.96% at 1,377 parameters and 99.88% at 1,376 parameters, substantially outperforming other attempted reductions; extending the same successful exact symmetry is the most direct informative test.

<<<<<<< SEARCH
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
=======
class GaugeFixedValueProjection(nn.Module):
    """Value projection with three orthogonal basis gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if out_features < 4:
            raise ValueError("out_features must be at least four")
        self.in_features = in_features
        self.out_features = out_features
        self.first_column = nn.Parameter(
            torch.empty(out_features - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            second[2, 0] = -sine
            second[2, 2] = cosine
            rotation = second @ rotation

        rotated = rotation @ raw
        self.first_column.copy_(
            torch.cat((rotated[:1, 0], rotated[3:, 0]))
        )
=======
            second[2, 0] = -sine
            second[2, 2] = cosine
            rotation = second @ rotation

        fourth_radius = (
            triple_radius.square() + raw[3, 0].square()
        ).sqrt()
        if float(fourth_radius.item()) > 0.0:
            third = torch.eye(
                self.out_features,
                device=raw.device,
                dtype=raw.dtype,
            )
            cosine = triple_radius / fourth_radius
            sine = raw[3, 0] / fourth_radius
            third[0, 0] = cosine
            third[0, 3] = sine
            third[3, 0] = -sine
            third[3, 3] = cosine
            rotation = third @ rotation

        rotated = rotation @ raw
        self.first_column.copy_(
            torch.cat((rotated[:1, 0], rotated[4:, 0]))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_column = torch.cat(
            (
                self.first_column[:1],
                self.first_column.new_zeros(2),
                self.first_column[1:],
            )
        )
=======
        first_column = torch.cat(
            (
                self.first_column[:1],
                self.first_column.new_zeros(3),
                self.first_column[1:],
            )
        )
>>>>>>> REPLACE