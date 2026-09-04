MECHANISM: Progressive high-band delta coalescing

HYPOTHESIS: Pairwise averaging the six highest-mel delta channels will retain at least 85% validation accuracy while reducing GRU input width from 38 to 37 and total inference MACs by approximately 4.83 million.

INTENDED_EDIT: Preserve all 20 spectral bands, 25 recurrent steps, 79-unit GRU, and dual-timescale readout, while representing delta bands 14–19 as three adjacent pairwise averages.

EVIDENCE: Compressing delta bands 16–19 into two pairwise averages retained 86.38% accuracy and reduced MACs, while prior absolute-spectrum compression failed; this supports testing one further local reduction confined to adjacent high-band dynamics.

<<<<<<< SEARCH
        self.gru = nn.GRU(38, 79, num_layers=1, batch_first=True)
=======
        self.gru = nn.GRU(37, 79, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        delta_features = torch.cat(
            (
                delta[:, :16],
                delta[:, 16:18].mean(dim=1, keepdim=True),
                delta[:, 18:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
=======
        delta_features = torch.cat(
            (
                delta[:, :14],
                delta[:, 14:16].mean(dim=1, keepdim=True),
                delta[:, 16:18].mean(dim=1, keepdim=True),
                delta[:, 18:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        delta_features = torch.cat(
            (
                deltas[:, :, :16],
                deltas[:, :, 16:18].mean(dim=2, keepdim=True),
                deltas[:, :, 18:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
=======
        delta_features = torch.cat(
            (
                deltas[:, :, :14],
                deltas[:, :, 14:16].mean(dim=2, keepdim=True),
                deltas[:, :, 16:18].mean(dim=2, keepdim=True),
                deltas[:, :, 18:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
>>>>>>> REPLACE