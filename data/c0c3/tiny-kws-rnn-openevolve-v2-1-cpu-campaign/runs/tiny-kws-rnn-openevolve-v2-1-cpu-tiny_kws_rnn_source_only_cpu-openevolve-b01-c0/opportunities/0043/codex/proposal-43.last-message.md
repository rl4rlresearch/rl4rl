MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 112-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 113-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 113 to 112 while preserving the successful 23-frame schedule and training procedure.

EVIDENCE: Every adjacent batch-64 width reduction from 117 through 113 units passed, and the current 113-unit model achieved 86.012% accuracy, leaving a 1.012-point margin for testing the next lower-cost width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 113, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(113, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 113, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 113, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
>>>>>>> REPLACE