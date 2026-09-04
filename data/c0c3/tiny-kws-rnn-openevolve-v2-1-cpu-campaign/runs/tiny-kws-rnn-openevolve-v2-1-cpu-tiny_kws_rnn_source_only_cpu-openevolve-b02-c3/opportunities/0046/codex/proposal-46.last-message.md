MECHANISM: Adjacent-width refinement with selective upper-band delta pooling

HYPOTHESIS: A 69-unit GRU preserving the qualified nine-delta representation and 27-frame schedule will achieve at least 85% validation accuracy while reducing total inference MACs from 458,396,750 to approximately 447,293,190.

INTENDED_EDIT: Reduce recurrent width from 71 to 69, resize the state and classifier, and compress the ten paired deltas into eight individual lower-band deltas plus one pooled upper-band delta.

EVIDENCE: The 70-unit nine-delta model qualified at 85.40% and 458,396,750 MACs, while a 69-unit full-delta model qualified at 85.77%; together they motivate testing the adjacent 69-unit boundary with the already-qualified nine-delta representation.

<<<<<<< SEARCH
        self.gru = nn.GRU(30, 71, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(142, 8)
=======
        self.gru = nn.GRU(29, 69, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(138, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 71, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 71, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 69, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled_delta = 0.5 * (delta[:, 0::2] + delta[:, 1::2])
        features = torch.cat((normalized, pooled_delta), dim=1)
=======
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
        pooled_deltas = 0.5 * (deltas[:, :, 0::2] + deltas[:, :, 1::2])
        features = torch.cat((normalized, pooled_deltas), dim=2)
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