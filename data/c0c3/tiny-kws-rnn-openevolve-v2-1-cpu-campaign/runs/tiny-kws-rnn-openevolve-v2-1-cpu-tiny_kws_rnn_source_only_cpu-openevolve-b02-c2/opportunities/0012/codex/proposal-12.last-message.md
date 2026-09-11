MECHANISM: Dual-view recurrent pooling with narrower state

HYPOTHESIS: A 96-unit GRU classifying concatenated mean and terminal outputs will retain at least 85% validation accuracy while reducing total dense MACs by about 3.7% versus the qualified 98-unit dual-view model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 98 to 96 while retaining all 32 frames and the established training procedure.

EVIDENCE: The 98-unit dual-view model achieved 87.12% accuracy versus 84.54% for 98-unit mean-only pooling, showing that the terminal representation recovered substantial accuracy and provides margin for the next two-unit width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE