MECHANISM: Adjacent-band delta pooling with preserved recurrent width

HYPOTHESIS: A 71-unit GRU receiving all 20 normalized bands plus 10 adjacent-band pooled deltas will retain at least 85% validation accuracy while reducing total inference MACs to approximately 474,319,405.

INTENDED_EDIT: Increase the recurrent state to the qualified 71-unit width, compress only the spectrally redundant delta features from 20 to 10 channels, and resize the classifier and state accordingly.

EVIDENCE: The full-delta 71-unit model achieved 86.13% accuracy, while shrinking the full-input recurrent state to 68 units failed at 84.17%; preserving recurrent capacity while structurally compressing locally correlated spectral motion tests a more favorable allocation of MACs.

<<<<<<< SEARCH
        self.gru = nn.GRU(40, 70, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(140, 8)
=======
        self.gru = nn.GRU(30, 71, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(142, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 70, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 70, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 71, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 71, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1)
=======
        delta = (normalized - previous) * has_previous
        pooled_delta = 0.5 * (delta[:, 0::2] + delta[:, 1::2])
        features = torch.cat((normalized, pooled_delta), dim=1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = torch.cat((normalized, deltas), dim=2)
        outputs, hidden = self.gru(
=======
        pooled_deltas = 0.5 * (deltas[:, :, 0::2] + deltas[:, :, 1::2])
        features = torch.cat((normalized, pooled_deltas), dim=2)
        outputs, hidden = self.gru(
>>>>>>> REPLACE