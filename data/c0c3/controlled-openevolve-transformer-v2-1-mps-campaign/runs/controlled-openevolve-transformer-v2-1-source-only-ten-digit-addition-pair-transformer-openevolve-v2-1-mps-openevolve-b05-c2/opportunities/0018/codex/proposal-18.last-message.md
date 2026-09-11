MECHANISM: Second compensated token/position shift-gauge anchor

HYPOTHESIS: Fixing a second mean-zero token/position shift coordinate will reduce the qualified 1579-parameter model to 1578 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove another coordinate from the final token embedding row and transfer its initialized two-coordinate contribution into every positional embedding, leaving the initialized input function and output probabilities unchanged.

EVIDENCE: The first compensated token/position gauge constraint achieved 99.97% accuracy at 1579 parameters, while removing all vocabulary-common modes failed; this motivates one incremental extension of the qualified gauge.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and token/position shift gauges fixed."""
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and two token/position shift gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_coords = gauged_weight[-1] @ basis
        shift = last_coords[-1]
        gauged_weight = gauged_weight - basis[:, -1] * shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-1])
=======
        last_coords = gauged_weight[-1] @ basis
        shift = last_coords[-2:]
        gauged_weight = gauged_weight - basis[:, -2:] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-2])
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords[-1]
        gauged_weight = gauged_weight - self.basis[:, -1] * shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-1])
=======
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords[-2:]
        gauged_weight = gauged_weight - self.basis[:, -2:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-2])
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_row = self.basis @ F.pad(self.last_weight, (0, 1))
=======
        last_row = self.basis @ F.pad(self.last_weight, (0, 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.pos_emb.weight[:, -1].add_(
                self.token_emb.initial_position_shift
            )
=======
            self.pos_emb.weight[:, -2:].add_(
                self.token_emb.initial_position_shift
            )
>>>>>>> REPLACE