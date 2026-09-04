MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 114-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 115-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 115 to 114 while preserving the successful schedule and training procedure.

EVIDENCE: The adjacent width reductions from 117 to 116 and from 116 to 115 both passed; the 115-unit model retained 86.380% accuracy, leaving a 1.38-point margin for another isolated width-boundary test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 115, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(115, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 115, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 115, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 114, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(114, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 114, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 114, device=device, dtype=dtype)
>>>>>>> REPLACE