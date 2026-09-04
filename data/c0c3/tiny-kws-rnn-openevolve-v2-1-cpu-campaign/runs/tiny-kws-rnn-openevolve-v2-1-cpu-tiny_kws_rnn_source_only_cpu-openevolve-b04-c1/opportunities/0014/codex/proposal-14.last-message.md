MECHANISM: Salience-preserving max-augmented recurrent readout

HYPOTHESIS: An 84-unit GRU classified from mean, maximum, and final recurrent outputs will recover the 0.22-point accuracy shortfall of the prior 84-unit model and reach at least 85% while retaining substantially fewer MACs than the verified 88-unit model.

INTENDED_EDIT: Reduce the GRU width to 84 and add a running elementwise maximum of recurrent outputs to the state and classifier features, preserving all 32 causal steps and the existing training procedure.

EVIDENCE: The 84-unit mean-plus-final model narrowly missed at 84.79%, while the 88-unit version reached 85.77%; adding a parameter-light maximum readout exposes salient transient speech evidence that temporal averaging may suppress, while the 84-unit recurrent core preserves the prior model’s structural MAC reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
=======
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return hidden.transpose(0, 1), summary + output, maximum, count + 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        sequence_maximum = outputs.amax(dim=1)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, sequence_maximum),
            sequence_maximum,
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            maximum,
            count + frames.shape[1],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        pooled = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]),
            dim=1,
        )
        return self.classifier(pooled)
=======
    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        pooled = torch.cat(
            (summary / count.clamp_min(1.0), maximum, hidden[:, 0, :]),
            dim=1,
        )
        return self.classifier(pooled)
>>>>>>> REPLACE