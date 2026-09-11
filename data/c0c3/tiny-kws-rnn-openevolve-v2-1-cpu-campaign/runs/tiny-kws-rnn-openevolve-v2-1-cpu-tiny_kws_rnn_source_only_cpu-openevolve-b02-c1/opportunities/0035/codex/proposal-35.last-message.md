MECHANISM: Delta-only high-band coalescing

HYPOTHESIS: Compressing only the two highest-mel delta channels will retain at least 85% accuracy while reducing total inference MACs from 575,537,515 to approximately 570,708,640.

INTENDED_EDIT: Preserve all 20 normalized spectral bands, the 79-unit GRU, 25-frame schedule, and readout, but average the final two delta bands into one feature and reduce GRU input width from 40 to 39.

EVIDENCE: The current spectrum-plus-delta model achieves 85.77% at 25 steps. The prior 20-to-19 spectral merge failed, motivating the more conservative reduction of only redundant local-dynamics detail while retaining every absolute spectral band.

<<<<<<< SEARCH
        self.gru = nn.GRU(40, 79, num_layers=1, batch_first=True)
=======
        self.gru = nn.GRU(39, 79, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1).unsqueeze(1)
=======
        delta = (normalized - previous) * has_previous
        delta_features = torch.cat(
            (delta[:, :18], delta[:, 18:].mean(dim=1, keepdim=True)),
            dim=1,
        )
        features = torch.cat((normalized, delta_features), dim=1).unsqueeze(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        deltas = torch.cat((first_delta, remaining_deltas), dim=1)
        features = torch.cat((normalized, deltas), dim=2)
=======
        deltas = torch.cat((first_delta, remaining_deltas), dim=1)
        delta_features = torch.cat(
            (
                deltas[:, :, :18],
                deltas[:, :, 18:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
        features = torch.cat((normalized, delta_features), dim=2)
>>>>>>> REPLACE