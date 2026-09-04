MECHANISM: Progressive high-band delta coalescing

HYPOTHESIS: Pairwise averaging the ten highest-mel delta channels will retain at least 85% validation accuracy while reducing GRU input width from 36 to 35 and total inference MACs by exactly 4,828,875.

INTENDED_EDIT: Preserve all 20 spectral bands, 25 recurrent steps, the 79-unit GRU, and dual-timescale readout, while representing delta bands 10–19 as five adjacent pairwise averages.

EVIDENCE: Four successive high-band delta coalescing designs retained the accuracy target while reducing MACs; the current 36-input model achieves 85.64%, motivating one further adjacent-pair compression.

<<<<<<< SEARCH
        self.gru = nn.GRU(36, 79, num_layers=1, batch_first=True)
=======
        self.gru = nn.GRU(35, 79, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
                delta[:, :12],
                delta[:, 12:14].mean(dim=1, keepdim=True),
=======
                delta[:, :10],
                delta[:, 10:12].mean(dim=1, keepdim=True),
                delta[:, 12:14].mean(dim=1, keepdim=True),
>>>>>>> REPLACE

<<<<<<< SEARCH
                deltas[:, :, :12],
                deltas[:, :, 12:14].mean(dim=2, keepdim=True),
=======
                deltas[:, :, :10],
                deltas[:, :, 10:12].mean(dim=2, keepdim=True),
                deltas[:, :, 12:14].mean(dim=2, keepdim=True),
>>>>>>> REPLACE