MECHANISM: Fine-grained recurrent-width scaling with full temporal coverage

HYPOTHESIS: A 60-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 9% versus the verified 64-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 64 to 60 and resize the classifier and recurrent state tensors accordingly.

EVIDENCE: The 64-unit model achieved 86.63% accuracy, and every tested width reduction from 104 through 64 units remained above the target while full temporal coverage proved more reliable than frame subsampling.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 60, device=device, dtype=dtype)
>>>>>>> REPLACE