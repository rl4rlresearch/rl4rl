MECHANISM: One-channel recurrent-width boundary trim

HYPOTHESIS: A 127-unit GRU using the verified 28-frame schedule will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.45% versus the passing 128-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 127 while preserving the successful schedule and training procedure.

EVIDENCE: The 128-unit model passed at 85.276%, while 125 units narrowly missed at 84.908%; despite the noisy 126-unit result, 127 is the most conservative untested structural reduction and isolates the remaining width boundary.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 127, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(127, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
>>>>>>> REPLACE