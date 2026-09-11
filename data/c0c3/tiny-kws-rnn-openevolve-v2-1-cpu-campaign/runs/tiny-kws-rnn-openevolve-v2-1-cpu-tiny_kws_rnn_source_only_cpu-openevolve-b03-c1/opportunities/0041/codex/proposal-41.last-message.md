MECHANISM: Asymmetric single-head width reduction

HYPOTHESIS: A 32/31/31 GRU ensemble will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.7% versus the verified 32/32/31 model.

INTENDED_EDIT: Reduce the second GRU head from 32 to 31 units and resize its hidden state, aggregate summaries, and classifier input from 95/285 to 94/282 features.

EVIDENCE: The 32/32/31 model achieved 85.40% accuracy, and earlier one-unit asymmetric reductions from 33/33/33 through 32/32/31 consistently met the target; changing one head is the smallest structural continuation of that successful boundary search.

<<<<<<< SEARCH
                nn.GRU(20, 32, num_layers=1, batch_first=True),
                nn.GRU(20, 32, num_layers=1, batch_first=True),
                nn.GRU(20, 31, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(285, 8)
=======
                nn.GRU(20, 32, num_layers=1, batch_first=True),
                nn.GRU(20, 31, num_layers=1, batch_first=True),
                nn.GRU(20, 31, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(282, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_0 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 31, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 95, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 95, device=device, dtype=dtype)
=======
        hidden_0 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 31, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 31, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 94, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 94, device=device, dtype=dtype)
>>>>>>> REPLACE