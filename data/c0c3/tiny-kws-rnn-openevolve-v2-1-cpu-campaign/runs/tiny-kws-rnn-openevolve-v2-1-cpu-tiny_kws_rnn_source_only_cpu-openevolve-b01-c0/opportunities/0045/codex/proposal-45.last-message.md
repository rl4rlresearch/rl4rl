MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 110-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 111-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 111 to 110 while preserving the verified schedule and training procedure.

EVIDENCE: Every adjacent batch-64 width reduction from 117 through 111 units passed, and the 111-unit model achieved 86.748% accuracy, leaving a 1.748-point margin for testing the next lower-cost width.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(111, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(110, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
>>>>>>> REPLACE