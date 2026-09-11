MECHANISM: Selective upper-band delta coarsening

HYPOTHESIS: A 71-unit GRU using eight adjacent-band delta pairs plus one pooled upper-band delta will retain at least 85% validation accuracy while reducing total inference MACs from 474,319,405 to approximately 469,632,340.

INTENDED_EDIT: Preserve the qualified 71-unit recurrent state and 27-frame schedule, but reduce delta features from 10 to 9 by merging the two highest-frequency pooled deltas.

EVIDENCE: The 71-unit model with 10 pooled deltas achieved 85.52% accuracy at 474,319,405 MACs, while reducing recurrent width to 68 failed; this motivates preserving recurrent capacity and making a conservative reduction only in the spectrally redundant delta representation.

<<<<<<< SEARCH
        self.gru = nn.GRU(40, 71, num_layers=1, batch_first=True)
=======
        self.gru = nn.GRU(29, 71, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1)
=======
        delta = (normalized - previous) * has_previous
        paired_delta = 0.5 * (delta[:, 0::2] + delta[:, 1::2])
        pooled_delta = torch.cat(
            (
                paired_delta[:, :8],
                paired_delta[:, 8:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        features = torch.cat((normalized, pooled_delta), dim=1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = torch.cat((normalized, deltas), dim=2)
=======
        paired_deltas = 0.5 * (
            deltas[:, :, 0::2] + deltas[:, :, 1::2]
        )
        pooled_deltas = torch.cat(
            (
                paired_deltas[:, :, :8],
                paired_deltas[:, :, 8:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
        features = torch.cat((normalized, pooled_deltas), dim=2)
>>>>>>> REPLACE