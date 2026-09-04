MECHANISM: Peak-augmented 76-unit full-resolution GRU

HYPOTHESIS: A 76-unit GRU using mean, maximum, and final recurrent features will retain at least 85% validation accuracy while reducing total dense inference MACs from 627,484,800 to approximately 572,325,600.

INTENDED_EDIT: Preserve all 32 causal frames, add the proven online maximum summary, and narrow the GRU from 96 to 76 hidden units.

EVIDENCE: The qualified 80-unit peak-augmented design achieved 86.63% accuracy at 627,484,800 MACs; its 1.63-point margin motivates the next controlled four-unit width reduction while preserving every temporal observation.

<<<<<<< SEARCH
        self.hidden_size = 96
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(2 * self.hidden_size, 8)
=======
        self.hidden_size = 76
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, 1, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, 1, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        maximum = torch.full(
            (batch_size, self.hidden_size),
            -1.0,
            device=device,
            dtype=dtype,
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        return self.classifier(features)
=======
    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)
>>>>>>> REPLACE