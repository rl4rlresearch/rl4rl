MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 108-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 109-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 109 to 108 while preserving the verified schedule and training procedure.

EVIDENCE: The 109-unit model achieved 87.607% accuracy, and every adjacent batch-64 reduction from 117 through 109 units passed, making the next one-channel trim the most informative lower-cost boundary test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 109, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(109, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 109, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 109, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
>>>>>>> REPLACE