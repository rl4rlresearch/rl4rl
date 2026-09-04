MECHANISM: Fine-grained full-resolution GRU width reduction

HYPOTHESIS: A 98-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while reducing total inference MACs below the qualified 100-unit model’s 939,532,000 MACs.

INTENDED_EDIT: Reduce the GRU hidden width, classifier input width, and recurrent state tensors from 108 to 98 while preserving the full-frame schedule and training procedure.

EVIDENCE: The full-resolution 100-unit GRU qualified at 85.03% accuracy after successive 112→108→104→100 reductions, while temporal subsampling failed below 83%; a two-unit width step probes the apparent capacity boundary more precisely.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
>>>>>>> REPLACE