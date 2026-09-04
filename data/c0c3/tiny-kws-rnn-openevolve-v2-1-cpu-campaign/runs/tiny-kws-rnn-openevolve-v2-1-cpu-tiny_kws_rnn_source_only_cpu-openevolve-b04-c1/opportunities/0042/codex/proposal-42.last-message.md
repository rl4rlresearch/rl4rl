MECHANISM: Recurrent-to-readout MAC reallocation with temporal-pyramid pooling

HYPOTHESIS: A 69-unit paired-frame GRU with separate early/late means will reach at least 85% accuracy while reducing estimated total inference MACs from 246.11M to 240.85M.

INTENDED_EDIT: Reduce the paired GRU from 70 to 69 units and replace its global mean with separate means for the first six and remaining seven learned transitions, expanding the classifier from three to four pooled views.

EVIDENCE: The 69-unit model narrowly missed the threshold at 84.79%, only 0.21 points short, while the otherwise identical 70-unit model reached 85.77%; spending 552 additional classifier MACs per example on explicit temporal structure tests whether cheap readout capacity can recover that narrow loss.

<<<<<<< SEARCH
        self.gru = nn.GRU(40, 70, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(210, 8)
=======
        self.gru = nn.GRU(40, 69, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(276, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 70, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 70, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 70, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count, pending, phase
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 69, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden, summary, maximum, count, pending, phase = state
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
>>>>>>> REPLACE

<<<<<<< SEARCH
        if phase[0, 0].item() < 0.5:
            return (
                hidden,
                summary,
                maximum,
                count,
                frame,
                torch.ones_like(phase),
            )
=======
        if phase[0, 0].item() < 0.5:
            return (
                hidden,
                early_summary,
                late_summary,
                maximum,
                count,
                frame,
                torch.ones_like(phase),
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        output = output[:, 0, :]
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            summary + output,
            maximum,
            count + 1.0,
            torch.zeros_like(pending),
            torch.zeros_like(phase),
        )
=======
        output = output[:, 0, :]
        early_weight = (count < 6.0).to(dtype=output.dtype)
        late_weight = 1.0 - early_weight
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            early_summary + early_weight * output,
            late_summary + late_weight * output,
            maximum,
            count + 1.0,
            torch.zeros_like(pending),
            torch.zeros_like(phase),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden, summary, maximum, count, pending, phase = state
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
>>>>>>> REPLACE

<<<<<<< SEARCH
            sequence_maximum = outputs.amax(dim=1)
            maximum = torch.where(
                count > 0,
                torch.maximum(maximum, sequence_maximum),
                sequence_maximum,
            )
            summary = summary + outputs.sum(dim=1)
            count = count + paired.shape[1]
=======
            sequence_maximum = outputs.amax(dim=1)
            maximum = torch.where(
                count > 0,
                torch.maximum(maximum, sequence_maximum),
                sequence_maximum,
            )
            positions = count.unsqueeze(1) + torch.arange(
                paired.shape[1],
                device=outputs.device,
                dtype=outputs.dtype,
            ).view(1, -1, 1)
            early_weights = (positions < 6.0).to(dtype=outputs.dtype)
            late_weights = 1.0 - early_weights
            early_summary = early_summary + (
                outputs * early_weights
            ).sum(dim=1)
            late_summary = late_summary + (
                outputs * late_weights
            ).sum(dim=1)
            count = count + paired.shape[1]
>>>>>>> REPLACE

<<<<<<< SEARCH
        return hidden, summary, maximum, count, pending, phase
=======
        return (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count, pending, phase = state
        del pending, phase
        pooled = torch.cat(
            (summary / count.clamp_min(1.0), maximum, hidden[:, 0, :]),
            dim=1,
        )
        return self.classifier(pooled)
=======
    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        del pending, phase
        early_count = count.clamp(max=6.0).clamp_min(1.0)
        late_count = (count - 6.0).clamp_min(1.0)
        pooled = torch.cat(
            (
                early_summary / early_count,
                late_summary / late_count,
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
        return self.classifier(pooled)
>>>>>>> REPLACE