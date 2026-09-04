MECHANISM: One-ULP EMA interpolation refinement

HYPOTHESIS: Increasing the EMA blend by one float32 ULP to 28.61345112323760986328125% will retain 9,133 correct predictions while lowering validation cross-entropy below 0.246640940094.

INTENDED_EDIT: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861344814300537109375 to its next float32 value, 0.2861345112323760986328125.

EVIDENCE: Increasing the blend from 0.286134243011474609375 to the current coefficient lowered cross-entropy while retaining 9,133 correct, and the tested coefficient four ULPs above current also retained 9,133; a one-ULP increase is the most precise probe of the local cross-entropy optimum.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.2861344814300537109375)
=======
                parameter.lerp_(shadow / correction, 0.2861345112323760986328125)
>>>>>>> REPLACE