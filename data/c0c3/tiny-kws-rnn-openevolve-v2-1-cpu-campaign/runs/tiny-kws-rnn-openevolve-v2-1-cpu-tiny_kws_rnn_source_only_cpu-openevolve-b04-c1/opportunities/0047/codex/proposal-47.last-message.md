MECHANISM: Progressive temporal-pyramid readout for recurrent-width pruning

HYPOTHESIS: A 66-unit paired-frame GRU with the seven-transition late pool split into three- and four-transition summaries will retain at least 85% validation accuracy while reducing estimated total inference MACs from 229,777,840 to 224,627,040.

INTENDED_EDIT: Reduce GRU width from 67 to 66 and add a sixth pooled classifier view by splitting the late summary at transition nine; preserve all 26 frames, 13 learned transitions, max pooling, final-state pooling, and seven relative logits.

EVIDENCE: Reducing the successful 68-unit model to 67 units while adding one inexpensive temporal summary retained 85.52% accuracy and lowered MACs; applying the same recurrent-to-readout trade reallocates capacity toward temporal structure while pruning another hidden unit.

<<<<<<< SEARCH
        self.gru = nn.GRU(40, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(335, 7)
=======
        self.gru = nn.GRU(40, 66, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(396, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        middle_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )
=======
        hidden = torch.zeros(batch_size, 1, 66, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        middle_summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        tail_summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            tail_summary,
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
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        frame = self.input_norm(frame)
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            tail_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        frame = self.input_norm(frame)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if phase[0, 0].item() < 0.5:
            return (
                hidden,
                early_summary,
                middle_summary,
                late_summary,
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
                middle_summary,
                late_summary,
                tail_summary,
                maximum,
                count,
                frame,
                torch.ones_like(phase),
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        early_weight = (count < 3.0).to(dtype=output.dtype)
        middle_weight = (
            (count >= 3.0) & (count < 6.0)
        ).to(dtype=output.dtype)
        late_weight = (count >= 6.0).to(dtype=output.dtype)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            early_summary + early_weight * output,
            middle_summary + middle_weight * output,
            late_summary + late_weight * output,
            maximum,
            count + 1.0,
            torch.zeros_like(pending),
            torch.zeros_like(phase),
        )
=======
        early_weight = (count < 3.0).to(dtype=output.dtype)
        middle_weight = (
            (count >= 3.0) & (count < 6.0)
        ).to(dtype=output.dtype)
        late_weight = (
            (count >= 6.0) & (count < 9.0)
        ).to(dtype=output.dtype)
        tail_weight = (count >= 9.0).to(dtype=output.dtype)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            early_summary + early_weight * output,
            middle_summary + middle_weight * output,
            late_summary + late_weight * output,
            tail_summary + tail_weight * output,
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
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        if frames.shape[1] == 0:
            return state
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            tail_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        if frames.shape[1] == 0:
            return state
>>>>>>> REPLACE

<<<<<<< SEARCH
            early_weights = (positions < 3.0).to(dtype=outputs.dtype)
            middle_weights = (
                (positions >= 3.0) & (positions < 6.0)
            ).to(dtype=outputs.dtype)
            late_weights = (positions >= 6.0).to(dtype=outputs.dtype)
            early_summary = early_summary + (
                outputs * early_weights
            ).sum(dim=1)
            middle_summary = middle_summary + (
                outputs * middle_weights
            ).sum(dim=1)
            late_summary = late_summary + (
                outputs * late_weights
            ).sum(dim=1)
=======
            early_weights = (positions < 3.0).to(dtype=outputs.dtype)
            middle_weights = (
                (positions >= 3.0) & (positions < 6.0)
            ).to(dtype=outputs.dtype)
            late_weights = (
                (positions >= 6.0) & (positions < 9.0)
            ).to(dtype=outputs.dtype)
            tail_weights = (positions >= 9.0).to(dtype=outputs.dtype)
            early_summary = early_summary + (
                outputs * early_weights
            ).sum(dim=1)
            middle_summary = middle_summary + (
                outputs * middle_weights
            ).sum(dim=1)
            late_summary = late_summary + (
                outputs * late_weights
            ).sum(dim=1)
            tail_summary = tail_summary + (
                outputs * tail_weights
            ).sum(dim=1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )

    def classify(
=======
        return (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            tail_summary,
            maximum,
            count,
            pending,
            phase,
        )

    def classify(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        del pending, phase
        early_count = count.clamp(max=3.0).clamp_min(1.0)
        middle_count = (count - 3.0).clamp(max=3.0).clamp_min(1.0)
        late_count = (count - 6.0).clamp_min(1.0)
        pooled = torch.cat(
            (
                early_summary / early_count,
                middle_summary / middle_count,
                late_summary / late_count,
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
=======
    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            tail_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        del pending, phase
        early_count = count.clamp(max=3.0).clamp_min(1.0)
        middle_count = (count - 3.0).clamp(max=3.0).clamp_min(1.0)
        late_count = (count - 6.0).clamp(max=3.0).clamp_min(1.0)
        tail_count = (count - 9.0).clamp_min(1.0)
        pooled = torch.cat(
            (
                early_summary / early_count,
                middle_summary / middle_count,
                late_summary / late_count,
                tail_summary / tail_count,
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
>>>>>>> REPLACE