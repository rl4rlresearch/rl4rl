MECHANISM: Three-head recurrent width boundary reduction

HYPOTHESIS: Three independent 33-unit GRU heads will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.7% versus the verified three-by-34 model.

INTENDED_EDIT: Reduce each GRU head from 34 to 33 units and resize the hidden state, trajectory summaries, and classifier input from 102/306 to 99/297 features.

EVIDENCE: Three-by-34 achieved 85.64% accuracy at 365.61M total MACs after three-by-35 achieved 86.26%; another one-unit-per-head reduction is the smallest symmetric structural change that directly tests the remaining capacity boundary.

<<<<<<< SEARCH
                nn.GRU(20, 34, num_layers=1, batch_first=True),
                nn.GRU(20, 34, num_layers=1, batch_first=True),
                nn.GRU(20, 34, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(306, 8)
=======
                nn.GRU(20, 33, num_layers=1, batch_first=True),
                nn.GRU(20, 33, num_layers=1, batch_first=True),
                nn.GRU(20, 33, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(297, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 3, 34, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 102, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 102, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 3, 33, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 99, device=device, dtype=dtype)
>>>>>>> REPLACE