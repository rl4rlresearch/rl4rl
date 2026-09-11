MECHANISM: Non-adjacent token/position shift-gauge anchoring

HYPOTHESIS: Anchoring the localized first mean-zero basis coordinate alongside the qualified final three coordinates will produce a 1576-parameter model with at least 99% accuracy, showing that the prior collapse was specific to the adjacent fourth coordinate rather than the gauge count.

INTENDED_EDIT: Replace the failed fourth-adjacent gauge with a non-adjacent four-coordinate gauge over coordinates 0, 4, 5, and 6, with exact positional compensation.

EVIDENCE: Three final-coordinate anchors achieved 99.8% at 1577 parameters, while adding adjacent coordinate 3 fell to 95.9%; testing coordinate 0 preserves the qualified anchors and isolates coordinate-selection effects.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and three token/position gauges fixed."""
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and four non-adjacent gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_coords = gauged_weight[-1] @ basis
        shift = last_coords[-3:]
        gauged_weight = gauged_weight - basis[:, -3:] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-3])
=======
        last_coords = gauged_weight[-1] @ basis
        shift = torch.cat((last_coords[:1], last_coords[-3:]))
        gauged_weight = gauged_weight - basis[:, [0, -3, -2, -1]] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[1:-3])
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords[-3:]
        gauged_weight = gauged_weight - self.basis[:, -3:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-3])
            self.initial_position_shift.copy_(shift)
=======
        last_coords = gauged_weight[-1] @ self.basis
        shift = torch.cat((last_coords[:1], last_coords[-3:]))
        gauged_weight = (
            gauged_weight - self.basis[:, [0, -3, -2, -1]] @ shift
        )
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[1:-3])
            self.initial_position_shift.copy_(shift)
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_row = self.basis @ F.pad(self.last_weight, (0, 3))
=======
        last_row = self.basis @ F.pad(self.last_weight, (1, 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            self.pos_emb.weight[:, -3:].add_(
                self.token_emb.initial_position_shift
            )
=======
        with torch.no_grad():
            self.pos_emb.weight[:, 0].add_(
                self.token_emb.initial_position_shift[0]
            )
            self.pos_emb.weight[:, -3:].add_(
                self.token_emb.initial_position_shift[1:]
            )
>>>>>>> REPLACE