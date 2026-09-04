MECHANISM: Three-head recurrent width boundary reduction

HYPOTHESIS: Three independent 34-unit GRU heads will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.6% versus the verified three-by-35 model.

INTENDED_EDIT: Reduce every GRU head from 35 to 34 units and resize the hidden state, trajectory summaries, and classifier input from 105/315 to 102/306 features.

EVIDENCE: Three 35-unit heads achieved 86.26% accuracy at 383.29M total MACs, leaving 1.26 percentage points of margin; a one-unit-per-head reduction is the smallest structural change that tests the remaining capacity boundary.

<<<<<<< SEARCH
                nn.GRU(20, 35, num_layers=1, batch_first=True),
                nn.GRU(20, 35, num_layers=1, batch_first=True),
                nn.GRU(20, 35, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(315, 8)
=======
                nn.GRU(20, 34, num_layers=1, batch_first=True),
                nn.GRU(20, 34, num_layers=1, batch_first=True),
                nn.GRU(20, 34, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(306, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 3, 35, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 105, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 105, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 3, 34, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 102, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 102, device=device, dtype=dtype)
>>>>>>> REPLACE