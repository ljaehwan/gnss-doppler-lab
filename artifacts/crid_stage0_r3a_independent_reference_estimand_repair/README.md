# CRID Stage-0 R3a independent-reference estimand repair

Final verdict: `INDEPENDENT_REFERENCE_ESTIMAND_REPAIR_PASS`
Next state: `READY_TO_REPEAT_CRID_PHASE_A`

This versioned post-result method repair leaves R3 permanently at `INCONCLUSIVE_CONTROL_PROVENANCE`. It changes neither the frozen generator nor any R3 artifact/control. The legacy single-PRN diagnostic reproduced 171/180 PASS and 9 FAIL, all and only OAK PRN 21. The preregistered independent five-PRN joint complex-LS reference produced 180/180 PASS.

OAK PRN 21's legacy single authentic magnitude is 259.02062327131716, the independent joint magnitude is 228.27974173261686, and single-minus-joint is 1.097339430699107 dB. Maximum target delay error is 0.0 chip, maximum target power error is 0.0025041235959696807 dB, maximum non-target relative energy is 0.0034457958445727657, and maximum condition number is 1.0452200216306857.

All source/control and existing R3 artifact hashes were freshly verified. Attack bytes read: 0. CRID score, threshold/alarm evaluation, Phase A, C1/C2/C3 replay, control regeneration, and attack evaluation were not executed. A PASS authorizes only a future repeat of CRID Phase A.
