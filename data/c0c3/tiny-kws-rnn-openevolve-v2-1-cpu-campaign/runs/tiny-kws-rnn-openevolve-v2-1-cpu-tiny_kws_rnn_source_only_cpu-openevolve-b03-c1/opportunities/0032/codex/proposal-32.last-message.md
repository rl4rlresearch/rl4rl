MECHANISM: Four-way block-diagonal recurrent ensemble

HYPOTHESIS: Four independent 29-unit GRU heads will maintain at least 85% validation accuracy while reducing total inference MACs by approximately 5.9% versus three 36-unit heads, because aggregate recurrent width increases from 108 to 116 while each hidden-to-hidden block becomes smaller.

INTENDED_EDIT: Replace the three 36-unit GRU heads with four 29-unit heads and resize the recurrent state summaries and endpoint/mean/maximum classifier input accordingly.

EVIDENCE: Moving from two 48-unit heads to three 36-unit heads retained 86.01% accuracy while reducing recurrent MACs by 7.4%; this continues the successful factorization pattern with greater aggregate representation and lower structural cost.

<<<<<<< SEARCH
    """Three parallel causal GRU heads with complementary learned dynamics."""
=======
    """Four parallel causal GRU heads with complementary learned dynamics."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 36, num_layers=1, batch_first=True),
                nn.GRU(20, 36, num_layers=1, batch_first=True),
                nn.GRU(20, 36, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(324, 8)
=======
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 29, num_layers=1, batch_first=True),
                nn.GRU(20, 29, num_layers=1, batch_first=True),
                nn.GRU(20, 29, num_layers=1, batch_first=True),
                nn.GRU(20, 29, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(348, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 3, 36, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 108, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 4, 29, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 116, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 116, device=device, dtype=dtype)
>>>>>>> REPLACE