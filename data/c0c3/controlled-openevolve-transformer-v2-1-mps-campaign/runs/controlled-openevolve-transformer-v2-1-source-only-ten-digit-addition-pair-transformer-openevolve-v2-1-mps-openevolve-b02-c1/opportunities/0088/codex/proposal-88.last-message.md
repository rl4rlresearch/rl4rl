MECHANISM: Disjoint nonterminal scalar-MLP input pairing

HYPOTHESIS: Sharing the third and fourth effective MLP input coefficients in addition to the successful fifth/sixth pair will reduce the model from 998 to 997 parameters while retaining at least 99% accuracy, because it preserves the first two score-basis coefficients and the reference-sensitive seventh coefficient independently.

INTENDED_EDIT: Store five `fc1` coefficients and reconstruct effective pairs at channels three/four and five/six, while preserving constructor and initialization RNG streams.

EVIDENCE: Fifth/sixth `fc1` sharing achieved 99.83%, whereas terminal sharing narrowly missed at 98.95% and interpolating the terminal coefficient collapsed to 26.89%; this motivates another disjoint nonterminal pair while leaving the terminal coefficient independent.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model - 2, d_ff, bias=False)
        self.fc1._removed_input_features = 1

        # Preserve the constructor RNG stream of the removed input weight.
        fc1_weight_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(d_ff).uniform_(
            -fc1_weight_bound, fc1_weight_bound
        )
=======
        self.fc1 = nn.Linear(d_model - 3, d_ff, bias=False)
        self.fc1._removed_input_features = 2

        # Preserve the constructor RNG stream of the removed input weights.
        fc1_weight_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(2 * d_ff).uniform_(
            -fc1_weight_bound, fc1_weight_bound
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        fc1_weight = torch.cat(
            [
                self.fc1.weight[:, :-1],
                self.fc1.weight[:, -2:-1],
                self.fc1.weight[:, -1:],
            ],
            dim=-1,
        )
=======
        fc1_weight = torch.cat(
            [
                self.fc1.weight[:, :2],
                self.fc1.weight[:, 2:3],
                self.fc1.weight[:, 2:3],
                self.fc1.weight[:, 3:4],
                self.fc1.weight[:, 3:4],
                self.fc1.weight[:, 4:],
            ],
            dim=-1,
        )
>>>>>>> REPLACE