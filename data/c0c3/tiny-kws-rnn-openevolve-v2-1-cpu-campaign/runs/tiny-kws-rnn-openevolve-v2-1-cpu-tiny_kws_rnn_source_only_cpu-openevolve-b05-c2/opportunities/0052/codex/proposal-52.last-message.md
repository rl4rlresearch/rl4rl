MECHANISM: Fixed mean-terminal summary fusion

HYPOTHESIS: The qualified 63-unit, 21-step GRU will retain at least 85% accuracy when its correlated mean and terminal summaries share classifier weights, reducing total inference MACs from 270,126,045 to approximately 269,715,285.

INTENDED_EDIT: Restore the qualified 21-frame, 63-unit design and replace the four-way readout with a three-way readout over mean-plus-final, maximum, and minimum summaries.

EVIDENCE: The 63-unit, 21-step model qualified at 85.03%, whereas 62 units and every tested 20-step schedule failed; preserving its recurrent path while structurally reducing only the redundant summary readout is the smallest orthogonal compute reduction.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 63, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(189, 8)
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
        return self.classifier(
            torch.cat(
                (mean_output + final_output, maximum, minimum),
                dim=-1,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(15 * available_frames // 16)
            schedule = [index for index in schedule if index not in skipped]
=======
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(3 * available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
>>>>>>> REPLACE