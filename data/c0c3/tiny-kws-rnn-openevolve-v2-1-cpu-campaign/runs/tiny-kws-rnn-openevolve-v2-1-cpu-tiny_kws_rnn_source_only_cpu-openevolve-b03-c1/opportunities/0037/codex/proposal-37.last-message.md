MECHANISM: Asymmetric single-head width boundary reduction

HYPOTHESIS: One 33-unit GRU head plus two 32-unit heads will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.6% versus the verified 33/33/32 model.

INTENDED_EDIT: Reduce the second GRU head from 33 to 32 units and resize its hidden state, aggregate summaries, and classifier input from 98/294 to 97/291 features.

EVIDENCE: The verified 33/33/32 model achieved 86.13% accuracy, leaving 1.13 percentage points of margin; reducing one additional head by a single unit is the smallest structural change that continues the successful asymmetric width search.

<<<<<<< SEARCH
    """Three compact parallel causal GRU heads with asymmetric widths."""
=======
    """Three compact parallel causal GRU heads with asymmetric widths."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                nn.GRU(20, 33, num_layers=1, batch_first=True),
                nn.GRU(20, 33, num_layers=1, batch_first=True),
                nn.GRU(20, 32, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(294, 8)
=======
                nn.GRU(20, 33, num_layers=1, batch_first=True),
                nn.GRU(20, 32, num_layers=1, batch_first=True),
                nn.GRU(20, 32, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(291, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_0 = torch.zeros(batch_size, 33, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 33, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden_0 = torch.zeros(batch_size, 33, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 97, device=device, dtype=dtype)
>>>>>>> REPLACE