# Q-SET-GNSS Stage-0A R2a

Final verdict: `RECEIVER_REPAIR_FAILED_CLEAN_REGRESSION`.

The locked audit isolated the original three-PRN SS-1 support deficit to receiver acquisition integration: V3 (12 concurrent acquisition channels, 8 ms coherent integration, original PFA) yielded six stable PRNs and 132 consecutive M>=5 windows. The candidate was rejected because the unchanged R2 model and threshold produced clean FPR 0.3944 on C-1 and 0.0791 on C-3, both failing the frozen regression gate. This is receiver engineering evidence only. No SS-1 spoofing score, morphology, ROC/AUC, or detection claim was made, and no unopened attack raw was accessed.
