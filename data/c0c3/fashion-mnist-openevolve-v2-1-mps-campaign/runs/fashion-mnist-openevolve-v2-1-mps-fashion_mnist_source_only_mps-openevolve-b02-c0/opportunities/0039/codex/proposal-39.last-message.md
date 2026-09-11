MECHANISM: Final-anchored late weight averaging

HYPOTHESIS: Blending 75% of the successful final-5% parameter average with 25% of the final checkpoint will exceed 9,252 correct predictions, or tie while lowering cross-entropy, by retaining snapshot variance reduction while reducing mismatch with final BatchNorm statistics.

INTENDED_EDIT: Replace full averaged-parameter substitution with a 75/25 interpolation between the late average and final trained parameters.

EVIDENCE: Uniform final-5% averaging reached 9,252 correct, while shortening the window to 2.5% fell to 9,247 and averaging BatchNorm statistics fell to 9,248; a final-checkpoint anchor tests an intermediate correction without discarding useful earlier snapshots.

<<<<<<< SEARCH
            if is_final_step:
                for parameter, average in zip(
                    parameters, optimizer._late_averaged_parameters
                ):
                    parameter.copy_(average)
=======
            if is_final_step:
                for parameter, average in zip(
                    parameters, optimizer._late_averaged_parameters
                ):
                    parameter.lerp_(average, 0.75)
>>>>>>> REPLACE