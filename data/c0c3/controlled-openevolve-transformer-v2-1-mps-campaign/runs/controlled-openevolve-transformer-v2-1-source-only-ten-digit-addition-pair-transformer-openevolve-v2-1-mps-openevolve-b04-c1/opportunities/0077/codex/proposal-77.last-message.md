MECHANISM: Eighth-row QKV input-shift quotient

HYPOTHESIS: Quotienting the eighth `qkv` weight row will reduce the model from 968 to 967 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes its common-mode input weight unobservable and identical ambient AdamW treatment succeeded for the first seven rows.

INTENDED_EDIT: Add a seven-coordinate Householder parameterization for the eighth `qkv` row, preserve conceptual initialization and random-stream order, and optimize all eight quotiented rows in eight-dimensional ambient space.

EVIDENCE: The immediately preceding seventh-row quotient achieved 99.83% accuracy at 968 parameters; extending the same exact symmetry and optimizer treatment to the adjacent eighth row is the closest supported incremental reduction.

<<<<<<< SEARCH
        self.seventh_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.rest_weight = nn.Parameter(
            torch.empty(out_features - 7, in_features)
        )
=======
        self.seventh_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.eighth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.rest_weight = nn.Parameter(
            torch.empty(out_features - 8, in_features)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            transformed_seventh = self._householder(conceptual_weight[6])
            self.seventh_coordinates.copy_(transformed_seventh[1:])
            self.rest_weight.copy_(conceptual_weight[7:])
=======
            transformed_seventh = self._householder(conceptual_weight[6])
            self.seventh_coordinates.copy_(transformed_seventh[1:])
            transformed_eighth = self._householder(conceptual_weight[7])
            self.eighth_coordinates.copy_(transformed_eighth[1:])
            self.rest_weight.copy_(conceptual_weight[8:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        seventh = F.pad(self.seventh_coordinates, (1, 0))
        seventh = self._householder(seventh)
        weight = torch.cat(
            (
                first.unsqueeze(0),
                second.unsqueeze(0),
                third.unsqueeze(0),
                fourth.unsqueeze(0),
                fifth.unsqueeze(0),
                sixth.unsqueeze(0),
                seventh.unsqueeze(0),
                self.rest_weight,
=======
        seventh = F.pad(self.seventh_coordinates, (1, 0))
        seventh = self._householder(seventh)
        eighth = F.pad(self.eighth_coordinates, (1, 0))
        eighth = self._householder(eighth)
        weight = torch.cat(
            (
                first.unsqueeze(0),
                second.unsqueeze(0),
                third.unsqueeze(0),
                fourth.unsqueeze(0),
                fifth.unsqueeze(0),
                sixth.unsqueeze(0),
                seventh.unsqueeze(0),
                eighth.unsqueeze(0),
                self.rest_weight,
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.qkv.fifth_coordinates,
            block.attn.qkv.sixth_coordinates,
            block.attn.qkv.seventh_coordinates,
        )
=======
            block.attn.qkv.fifth_coordinates,
            block.attn.qkv.sixth_coordinates,
            block.attn.qkv.seventh_coordinates,
            block.attn.qkv.eighth_coordinates,
        )
>>>>>>> REPLACE