MECHANISM: LayerNorm offset absorption into GRU biases

HYPOTHESIS: Removing only LayerNorm’s 20 bias parameters from the qualified 110-unit, 22-step, 219-feature bias-free-readout model will retain at least 85% validation accuracy with unchanged MACs and recurrent steps, while reducing learned parameters from 45,352 to 45,332.

INTENDED_EDIT: Restore the qualified 219-feature bias-free classifier and disable only the input LayerNorm bias while retaining its learned scale.

EVIDENCE: The 219-feature bias-free design achieved 85.40% accuracy; fully non-affine LayerNorm missed by only 0.09 points after removing both scale and offset, motivating the narrower test of removing the offset, which can be represented by the following GRU biases.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(220, 8)
=======
        self.input_norm = nn.LayerNorm(20, bias=False)
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(219, 8, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))
=======
        return self.classifier(
            torch.cat((mean_output, hidden[:, 0, :-1]), dim=1)
        )
>>>>>>> REPLACE