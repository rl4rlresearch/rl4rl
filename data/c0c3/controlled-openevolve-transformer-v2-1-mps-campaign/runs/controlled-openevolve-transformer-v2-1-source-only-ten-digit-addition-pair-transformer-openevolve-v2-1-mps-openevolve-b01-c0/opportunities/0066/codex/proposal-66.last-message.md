MECHANISM: Final-column MLP output common-mode gauge removal

HYPOTHESIS: Centering the final `fc2` column will reduce the model to 1605 parameters while maintaining at least 99% accuracy, because it removes the same downstream-LayerNorm-invariant component already removed successfully from the first two columns while avoiding the sensitive third column.

INTENDED_EDIT: Store the middle `fc2` columns in full and represent the final column with seven learned coordinates whose eighth coordinate enforces a zero column sum.

EVIDENCE: Centering the first two `fc2` columns supports 99.89% accuracy at 1606 parameters, whereas centering the third fell to 13.83%; prior LayerNorm results show that changing which coordinate receives an otherwise equivalent reduction can restore accuracy, motivating a distant final-column test.

<<<<<<< SEARCH
        self.fc2_first_column = nn.Parameter(torch.empty(d_model - 1))
        self.fc2_second_column = nn.Parameter(torch.empty(d_model - 1))
        self.fc2_rest = nn.Parameter(torch.empty(d_model, d_ff - 2))
        self.fc2.weight = None
=======
        self.fc2_first_column = nn.Parameter(torch.empty(d_model - 1))
        self.fc2_second_column = nn.Parameter(torch.empty(d_model - 1))
        self.fc2_middle = nn.Parameter(torch.empty(d_model, d_ff - 3))
        self.fc2_last_column = nn.Parameter(torch.empty(d_model - 1))
        self.fc2.weight = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        second_final_coordinate = -self.fc2_second_column.sum().reshape(1)
        second_column = torch.cat((self.fc2_second_column, second_final_coordinate))
        weight = torch.cat(
            (first_column.unsqueeze(1), second_column.unsqueeze(1), self.fc2_rest),
            dim=1,
        )
=======
        second_final_coordinate = -self.fc2_second_column.sum().reshape(1)
        second_column = torch.cat((self.fc2_second_column, second_final_coordinate))
        last_final_coordinate = -self.fc2_last_column.sum().reshape(1)
        last_column = torch.cat((self.fc2_last_column, last_final_coordinate))
        weight = torch.cat(
            (
                first_column.unsqueeze(1),
                second_column.unsqueeze(1),
                self.fc2_middle,
                last_column.unsqueeze(1),
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            first_column = full_weight[:, 0] - full_weight[:, 0].mean()
            second_column = full_weight[:, 1] - full_weight[:, 1].mean()
            with torch.no_grad():
                module.fc2_first_column.copy_(first_column[:-1])
                module.fc2_second_column.copy_(second_column[:-1])
                module.fc2_rest.copy_(full_weight[:, 2:])
=======
            first_column = full_weight[:, 0] - full_weight[:, 0].mean()
            second_column = full_weight[:, 1] - full_weight[:, 1].mean()
            last_column = full_weight[:, -1] - full_weight[:, -1].mean()
            with torch.no_grad():
                module.fc2_first_column.copy_(first_column[:-1])
                module.fc2_second_column.copy_(second_column[:-1])
                module.fc2_middle.copy_(full_weight[:, 2:-1])
                module.fc2_last_column.copy_(last_column[:-1])
>>>>>>> REPLACE