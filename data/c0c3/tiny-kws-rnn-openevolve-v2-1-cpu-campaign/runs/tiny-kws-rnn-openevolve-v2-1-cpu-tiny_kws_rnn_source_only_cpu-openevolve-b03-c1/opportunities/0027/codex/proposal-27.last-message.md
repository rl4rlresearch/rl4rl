MECHANISM: Temporal-max-assisted recurrent width reduction

HYPOTHESIS: A 76-unit GRU retaining the successful 27-step endpoint, mean, and temporal-maximum readout will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 8.8%.

INTENDED_EDIT: Reduce the GRU hidden width from 80 to 76 and resize all recurrent state summaries and the classifier input from 240 to 228 features.

EVIDENCE: The 80-unit 27-step model with temporal maximum achieved 87.61%, substantially above the threshold, while the 76-unit mean-only model narrowly missed at 84.79%; the complementary maximum readout provides a plausible margin for revisiting that width at materially lower cost.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
=======
        self.gru = nn.GRU(20, 76, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(228, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 80, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 76, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 76, device=device, dtype=dtype)
>>>>>>> REPLACE