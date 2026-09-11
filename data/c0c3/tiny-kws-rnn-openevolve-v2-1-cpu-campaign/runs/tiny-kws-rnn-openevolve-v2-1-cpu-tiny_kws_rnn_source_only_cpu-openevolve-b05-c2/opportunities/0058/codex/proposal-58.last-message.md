MECHANISM: Reference-class softmax parameterization

HYPOTHESIS: The qualified 63-unit, 21-step GRU will retain at least 85% validation accuracy when one redundant softmax logit is fixed at zero, reducing total inference MACs from 270,126,045 to approximately 269,920,665.

INTENDED_EDIT: Restore the qualified 63-unit recurrent path and replace the eight-output classifier with seven learned logits plus one fixed reference logit.

EVIDENCE: The 63-unit, 21-step model qualified at 85.03%; unlike the failed summary fusion, fixing one softmax logit removes only the common-offset redundancy and preserves the full space of eight-class probability distributions.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 63, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 63, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 63, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 63, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 63, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(
            torch.cat(
                (mean_output, final_output, maximum, minimum),
                dim=-1,
            )
        )
=======
        learned_logits = self.classifier(
            torch.cat(
                (mean_output, final_output, maximum, minimum),
                dim=-1,
            )
        )
        reference_logit = learned_logits.new_zeros(
            (learned_logits.shape[0], 1)
        )
        return torch.cat((learned_logits, reference_logit), dim=-1)
>>>>>>> REPLACE