MECHANISM: One-unit recurrent width boundary trim

HYPOTHESIS: A 98-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 461,518,200 to approximately 453,022,640.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 101 to 98 units while preserving the qualified batch-64 training procedure and 16-frame even-index schedule.

EVIDENCE: The adjacent 99-unit design achieved 85.521% accuracy at 461,518,200 MACs; 98 units is the next untested structural reduction, while the failed 15-frame design indicates temporal coverage should remain unchanged.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 101, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(101, 8)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 101, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 101, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
>>>>>>> REPLACE