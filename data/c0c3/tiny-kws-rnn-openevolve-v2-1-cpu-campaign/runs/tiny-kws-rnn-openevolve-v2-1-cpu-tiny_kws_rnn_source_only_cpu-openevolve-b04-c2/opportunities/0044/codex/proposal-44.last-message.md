MECHANISM: One-unit recurrent-width reduction on the residual-pooled frontier

HYPOTHESIS: An 84-unit GRU processing frames 4–23 with the qualified 25% final-state residual will retain at least 85% validation accuracy while reducing total inference MACs to approximately 427,738,080.

INTENDED_EDIT: Reduce the qualified 85-unit architecture to 84 hidden units, retain its 20-frame schedule and residual pooling, and use batch size 16.

EVIDENCE: The 85-unit residual-pooled model qualified at 85.40% and 436,986,700 MACs; reducing the qualified 86-unit model by one unit also preserved qualification, making the adjacent 84-unit width the cheapest unresolved frontier.

<<<<<<< SEARCH
BATCH_SIZE = 32
=======
BATCH_SIZE = 16
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(84, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))
=======
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        pooled = 0.75 * mean_output + 0.25 * hidden[:, 0, :]
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 13:
            return list(range(available_frames))
        return list(range(4, available_frames - 8))
>>>>>>> REPLACE