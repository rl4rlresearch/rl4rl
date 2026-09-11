MECHANISM: Additional query-key rotation gauge fixing

HYPOTHESIS: Applying the qualified twelfth orthogonal query-key rotation will reduce the model from 1494 to 1493 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Fix the second attention head’s query-weight coordinate 2 at zero via a joint query-key rotation, removing one learned scalar.

EVIDENCE: Reference Design 3 achieved 0.9994 accuracy with 1493 parameters using this exact additional rotation.

<<<<<<< SEARCH
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and nine value-output gauges fixed."""
=======
class TwelveRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with twelve query-key and nine value-output gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 2:]
        )
=======
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 3:]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            (self.second_query + 1, 1),
            (0, 2),
        )
=======
            (self.second_query + 1, 1),
            (0, 2),
            (self.second_query, 2),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 2:]
            )
=======
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 3:]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_row = F.pad(self.head_two_weight, (2, 0))
=======
        head_two_row = F.pad(self.head_two_weight, (3, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationNineValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = TwelveRotationNineValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationNineValueGaugeFixedQKV):
=======
        elif isinstance(module, TwelveRotationNineValueGaugeFixedQKV):
>>>>>>> REPLACE