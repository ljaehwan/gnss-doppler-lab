# SCI ds4 baseline sweep

Normal-only robust-z baseline. Thresholds are calibrated on cleanStatic only; ds records are only evaluation.

## Best ds4 methods by q99 post detection
- full_no_raw_power/<bound method Series.aggregate of group                           full_no_raw_power
agg                                    score_mean
scenario                                      ds4
feature_count                                  57
rows                                          255
pfa10_q90_normal_flag_rate                    NaN
pfa5_q95_normal_flag_rate                     NaN
pfa1_q99_normal_flag_rate                     NaN
pfa0_5_q995_normal_flag_rate                  NaN
pfa0_1_q999_normal_flag_rate                  NaN
pre_windows_t_lt_90                         179.0
post_windows_t_ge_110                        36.0
auc_pre_vs_post_buffered                 0.646493
pre_score_median                         2.371151
post_score_median                        2.444258
pre_score_q95                            3.173812
post_score_q95                           2.530576
pfa10_q90_pre_fp_rate                    0.664804
pfa10_q90_post_det_rate                  0.805556
pfa10_q90_first_delay_s                      10.5
pfa5_q95_pre_fp_rate                     0.357542
pfa5_q95_post_det_rate                   0.722222
pfa5_q95_first_delay_s                       10.5
pfa1_q99_pre_fp_rate                     0.189944
pfa1_q99_post_det_rate                   0.333333
pfa1_q99_first_delay_s                       15.0
pfa0_5_q995_pre_fp_rate                  0.184358
pfa0_5_q995_post_det_rate                0.166667
pfa0_5_q995_first_delay_s                    17.0
pfa0_1_q999_pre_fp_rate                  0.117318
pfa0_1_q999_post_det_rate                     0.0
pfa0_1_q999_first_delay_s                     NaN
Name: 147, dtype: object>: q99 det=0.333, q99 FP=0.190, delay=15.0, AUC=0.646, q95 det=0.722
- tap_cv_only/<bound method Series.aggregate of group                           tap_cv_only
agg                              score_mean
scenario                                ds4
feature_count                             9
rows                                    255
pfa10_q90_normal_flag_rate              NaN
pfa5_q95_normal_flag_rate               NaN
pfa1_q99_normal_flag_rate               NaN
pfa0_5_q995_normal_flag_rate            NaN
pfa0_1_q999_normal_flag_rate            NaN
pre_windows_t_lt_90                   179.0
post_windows_t_ge_110                  36.0
auc_pre_vs_post_buffered           0.172564
pre_score_median                   1.143337
post_score_median                  0.902882
pre_score_q95                      1.625082
post_score_q95                     1.367691
pfa10_q90_pre_fp_rate              0.011173
pfa10_q90_post_det_rate            0.055556
pfa10_q90_first_delay_s                13.5
pfa5_q95_pre_fp_rate               0.011173
pfa5_q95_post_det_rate             0.027778
pfa5_q95_first_delay_s                 14.0
pfa1_q99_pre_fp_rate                    0.0
pfa1_q99_post_det_rate             0.027778
pfa1_q99_first_delay_s                 14.0
pfa0_5_q995_pre_fp_rate                 0.0
pfa0_5_q995_post_det_rate          0.027778
pfa0_5_q995_first_delay_s              14.0
pfa0_1_q999_pre_fp_rate                 0.0
pfa0_1_q999_post_det_rate          0.027778
pfa0_1_q999_first_delay_s              14.0
Name: 67, dtype: object>: q99 det=0.028, q99 FP=0.000, delay=14.0, AUC=0.173, q95 det=0.028
- tap_cv_only/<bound method Series.aggregate of group                           tap_cv_only
agg                              score_top5
scenario                                ds4
feature_count                             9
rows                                    255
pfa10_q90_normal_flag_rate              NaN
pfa5_q95_normal_flag_rate               NaN
pfa1_q99_normal_flag_rate               NaN
pfa0_5_q995_normal_flag_rate            NaN
pfa0_1_q999_normal_flag_rate            NaN
pre_windows_t_lt_90                   179.0
post_windows_t_ge_110                  36.0
auc_pre_vs_post_buffered           0.130199
pre_score_median                   1.966059
post_score_median                  1.364613
pre_score_q95                      2.931778
post_score_q95                     2.187599
pfa10_q90_pre_fp_rate              0.011173
pfa10_q90_post_det_rate            0.027778
pfa10_q90_first_delay_s                14.0
pfa5_q95_pre_fp_rate               0.005587
pfa5_q95_post_det_rate             0.027778
pfa5_q95_first_delay_s                 14.0
pfa1_q99_pre_fp_rate               0.005587
pfa1_q99_post_det_rate             0.027778
pfa1_q99_first_delay_s                 14.0
pfa0_5_q995_pre_fp_rate            0.005587
pfa0_5_q995_post_det_rate          0.027778
pfa0_5_q995_first_delay_s              14.0
pfa0_1_q999_pre_fp_rate                 0.0
pfa0_1_q999_post_det_rate          0.027778
pfa0_1_q999_first_delay_s              14.0
Name: 75, dtype: object>: q99 det=0.028, q99 FP=0.006, delay=14.0, AUC=0.130, q95 det=0.028
- tap_rel_prompt_only/<bound method Series.aggregate of group                           tap_rel_prompt_only
agg                                      score_mean
scenario                                        ds4
feature_count                                     9
rows                                            255
pfa10_q90_normal_flag_rate                      NaN
pfa5_q95_normal_flag_rate                       NaN
pfa1_q99_normal_flag_rate                       NaN
pfa0_5_q995_normal_flag_rate                    NaN
pfa0_1_q999_normal_flag_rate                    NaN
pre_windows_t_lt_90                           179.0
post_windows_t_ge_110                          36.0
auc_pre_vs_post_buffered                     0.9072
pre_score_median                           1.125149
post_score_median                          2.464234
pre_score_q95                              1.966129
post_score_q95                             2.672335
pfa10_q90_pre_fp_rate                      0.027933
pfa10_q90_post_det_rate                    0.277778
pfa10_q90_first_delay_s                        15.0
pfa5_q95_pre_fp_rate                       0.022346
pfa5_q95_post_det_rate                     0.055556
pfa5_q95_first_delay_s                         15.5
pfa1_q99_pre_fp_rate                            0.0
pfa1_q99_post_det_rate                          0.0
pfa1_q99_first_delay_s                          NaN
pfa0_5_q995_pre_fp_rate                         0.0
pfa0_5_q995_post_det_rate                       0.0
pfa0_5_q995_first_delay_s                       NaN
pfa0_1_q999_pre_fp_rate                         0.0
pfa0_1_q999_post_det_rate                       0.0
pfa0_1_q999_first_delay_s                       NaN
Name: 83, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.907, q95 det=0.056
- dmcpd_only/<bound method Series.aggregate of group                           dmcpd_only
agg                             score_mean
scenario                               ds4
feature_count                           25
rows                                   255
pfa10_q90_normal_flag_rate             NaN
pfa5_q95_normal_flag_rate              NaN
pfa1_q99_normal_flag_rate              NaN
pfa0_5_q995_normal_flag_rate           NaN
pfa0_1_q999_normal_flag_rate           NaN
pre_windows_t_lt_90                  179.0
post_windows_t_ge_110                 36.0
auc_pre_vs_post_buffered          0.887647
pre_score_median                  1.182254
post_score_median                 2.268861
pre_score_q95                     1.858608
post_score_q95                    2.434863
pfa10_q90_pre_fp_rate             0.011173
pfa10_q90_post_det_rate                0.0
pfa10_q90_first_delay_s                NaN
pfa5_q95_pre_fp_rate              0.005587
pfa5_q95_post_det_rate                 0.0
pfa5_q95_first_delay_s                 NaN
pfa1_q99_pre_fp_rate                   0.0
pfa1_q99_post_det_rate                 0.0
pfa1_q99_first_delay_s                 NaN
pfa0_5_q995_pre_fp_rate                0.0
pfa0_5_q995_post_det_rate              0.0
pfa0_5_q995_first_delay_s              NaN
pfa0_1_q999_pre_fp_rate                0.0
pfa0_1_q999_post_det_rate              0.0
pfa0_1_q999_first_delay_s              NaN
Name: 35, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.888, q95 det=0.000
- tap_rel_cv_only/<bound method Series.aggregate of group                           tap_rel_cv_only
agg                                  score_mean
scenario                                    ds4
feature_count                                27
rows                                        255
pfa10_q90_normal_flag_rate                  NaN
pfa5_q95_normal_flag_rate                   NaN
pfa1_q99_normal_flag_rate                   NaN
pfa0_5_q995_normal_flag_rate                NaN
pfa0_1_q999_normal_flag_rate                NaN
pre_windows_t_lt_90                       179.0
post_windows_t_ge_110                      36.0
auc_pre_vs_post_buffered               0.881285
pre_score_median                       1.163013
post_score_median                      1.923652
pre_score_q95                          1.708183
post_score_q95                         2.074542
pfa10_q90_pre_fp_rate                   0.01676
pfa10_q90_post_det_rate                     0.0
pfa10_q90_first_delay_s                     NaN
pfa5_q95_pre_fp_rate                   0.011173
pfa5_q95_post_det_rate                      0.0
pfa5_q95_first_delay_s                      NaN
pfa1_q99_pre_fp_rate                        0.0
pfa1_q99_post_det_rate                      0.0
pfa1_q99_first_delay_s                      NaN
pfa0_5_q995_pre_fp_rate                     0.0
pfa0_5_q995_post_det_rate                   0.0
pfa0_5_q995_first_delay_s                   NaN
pfa0_1_q999_pre_fp_rate                     0.0
pfa0_1_q999_post_det_rate                   0.0
pfa0_1_q999_first_delay_s                   NaN
Name: 51, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.881, q95 det=0.000
- morph_core_no_doppler_no_cn0_no_power/<bound method Series.aggregate of group                           morph_core_no_doppler_no_cn0_no_power
agg                                                        score_mean
scenario                                                          ds4
feature_count                                                      63
rows                                                              255
pfa10_q90_normal_flag_rate                                        NaN
pfa5_q95_normal_flag_rate                                         NaN
pfa1_q99_normal_flag_rate                                         NaN
pfa0_5_q995_normal_flag_rate                                      NaN
pfa0_1_q999_normal_flag_rate                                      NaN
pre_windows_t_lt_90                                             179.0
post_windows_t_ge_110                                            36.0
auc_pre_vs_post_buffered                                     0.876164
pre_score_median                                             1.179697
post_score_median                                            1.996186
pre_score_q95                                                 1.81235
post_score_q95                                               2.134233
pfa10_q90_pre_fp_rate                                        0.011173
pfa10_q90_post_det_rate                                           0.0
pfa10_q90_first_delay_s                                           NaN
pfa5_q95_pre_fp_rate                                         0.011173
pfa5_q95_post_det_rate                                            0.0
pfa5_q95_first_delay_s                                            NaN
pfa1_q99_pre_fp_rate                                              0.0
pfa1_q99_post_det_rate                                            0.0
pfa1_q99_first_delay_s                                            NaN
pfa0_5_q995_pre_fp_rate                                           0.0
pfa0_5_q995_post_det_rate                                         0.0
pfa0_5_q995_first_delay_s                                         NaN
pfa0_1_q999_pre_fp_rate                                           0.0
pfa0_1_q999_post_det_rate                                         0.0
pfa0_1_q999_first_delay_s                                         NaN
Name: 131, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.876, q95 det=0.000
- tap_rel_sum_only/<bound method Series.aggregate of group                           tap_rel_sum_only
agg                                   score_mean
scenario                                     ds4
feature_count                                  9
rows                                         255
pfa10_q90_normal_flag_rate                   NaN
pfa5_q95_normal_flag_rate                    NaN
pfa1_q99_normal_flag_rate                    NaN
pfa0_5_q995_normal_flag_rate                 NaN
pfa0_1_q999_normal_flag_rate                 NaN
pre_windows_t_lt_90                        179.0
post_windows_t_ge_110                       36.0
auc_pre_vs_post_buffered                 0.87306
pre_score_median                        1.022734
post_score_median                       1.782752
pre_score_q95                           1.497052
post_score_q95                          2.036235
pfa10_q90_pre_fp_rate                   0.011173
pfa10_q90_post_det_rate                      0.0
pfa10_q90_first_delay_s                      NaN
pfa5_q95_pre_fp_rate                         0.0
pfa5_q95_post_det_rate                       0.0
pfa5_q95_first_delay_s                       NaN
pfa1_q99_pre_fp_rate                         0.0
pfa1_q99_post_det_rate                       0.0
pfa1_q99_first_delay_s                       NaN
pfa0_5_q995_pre_fp_rate                      0.0
pfa0_5_q995_post_det_rate                    0.0
pfa0_5_q995_first_delay_s                    NaN
pfa0_1_q999_pre_fp_rate                      0.0
pfa0_1_q999_post_det_rate                    0.0
pfa0_1_q999_first_delay_s                    NaN
Name: 99, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.873, q95 det=0.000
- tap_rel_prompt_only/<bound method Series.aggregate of group                           tap_rel_prompt_only
agg                                      score_top5
scenario                                        ds4
feature_count                                     9
rows                                            255
pfa10_q90_normal_flag_rate                      NaN
pfa5_q95_normal_flag_rate                       NaN
pfa1_q99_normal_flag_rate                       NaN
pfa0_5_q995_normal_flag_rate                    NaN
pfa0_1_q999_normal_flag_rate                    NaN
pre_windows_t_lt_90                           179.0
post_windows_t_ge_110                          36.0
auc_pre_vs_post_buffered                    0.86406
pre_score_median                           1.568834
post_score_median                          3.481008
pre_score_q95                              3.152518
post_score_q95                              3.76493
pfa10_q90_pre_fp_rate                       0.01676
pfa10_q90_post_det_rate                         0.0
pfa10_q90_first_delay_s                         NaN
pfa5_q95_pre_fp_rate                       0.005587
pfa5_q95_post_det_rate                          0.0
pfa5_q95_first_delay_s                          NaN
pfa1_q99_pre_fp_rate                            0.0
pfa1_q99_post_det_rate                          0.0
pfa1_q99_first_delay_s                          NaN
pfa0_5_q995_pre_fp_rate                         0.0
pfa0_5_q995_post_det_rate                       0.0
pfa0_5_q995_first_delay_s                       NaN
pfa0_1_q999_pre_fp_rate                         0.0
pfa0_1_q999_post_det_rate                       0.0
pfa0_1_q999_first_delay_s                       NaN
Name: 91, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.864, q95 det=0.000
- dmcpd_only/<bound method Series.aggregate of group                           dmcpd_only
agg                             score_top5
scenario                               ds4
feature_count                           25
rows                                   255
pfa10_q90_normal_flag_rate             NaN
pfa5_q95_normal_flag_rate              NaN
pfa1_q99_normal_flag_rate              NaN
pfa0_5_q995_normal_flag_rate           NaN
pfa0_1_q999_normal_flag_rate           NaN
pre_windows_t_lt_90                  179.0
post_windows_t_ge_110                 36.0
auc_pre_vs_post_buffered          0.854749
pre_score_median                  1.762896
post_score_median                 3.122783
pre_score_q95                         3.37
post_score_q95                    3.343796
pfa10_q90_pre_fp_rate             0.011173
pfa10_q90_post_det_rate                0.0
pfa10_q90_first_delay_s                NaN
pfa5_q95_pre_fp_rate              0.011173
pfa5_q95_post_det_rate                 0.0
pfa5_q95_first_delay_s                 NaN
pfa1_q99_pre_fp_rate                   0.0
pfa1_q99_post_det_rate                 0.0
pfa1_q99_first_delay_s                 NaN
pfa0_5_q995_pre_fp_rate                0.0
pfa0_5_q995_post_det_rate              0.0
pfa0_5_q995_first_delay_s              NaN
pfa0_1_q999_pre_fp_rate                0.0
pfa0_1_q999_post_det_rate              0.0
pfa0_1_q999_first_delay_s              NaN
Name: 43, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.855, q95 det=0.000
- tap_rel_prompt_only/<bound method Series.aggregate of group                           tap_rel_prompt_only
agg                                      score_top3
scenario                                        ds4
feature_count                                     9
rows                                            255
pfa10_q90_normal_flag_rate                      NaN
pfa5_q95_normal_flag_rate                       NaN
pfa1_q99_normal_flag_rate                       NaN
pfa0_5_q995_normal_flag_rate                    NaN
pfa0_1_q999_normal_flag_rate                    NaN
pre_windows_t_lt_90                           179.0
post_windows_t_ge_110                          36.0
auc_pre_vs_post_buffered                   0.852266
pre_score_median                           1.766803
post_score_median                          4.091612
pre_score_q95                              4.522846
post_score_q95                             4.445735
pfa10_q90_pre_fp_rate                       0.01676
pfa10_q90_post_det_rate                         0.0
pfa10_q90_first_delay_s                         NaN
pfa5_q95_pre_fp_rate                       0.005587
pfa5_q95_post_det_rate                          0.0
pfa5_q95_first_delay_s                          NaN
pfa1_q99_pre_fp_rate                            0.0
pfa1_q99_post_det_rate                          0.0
pfa1_q99_first_delay_s                          NaN
pfa0_5_q995_pre_fp_rate                         0.0
pfa0_5_q995_post_det_rate                       0.0
pfa0_5_q995_first_delay_s                       NaN
pfa0_1_q999_pre_fp_rate                         0.0
pfa0_1_q999_post_det_rate                       0.0
pfa0_1_q999_first_delay_s                       NaN
Name: 87, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.852, q95 det=0.000
- tap_rel_cv_only/<bound method Series.aggregate of group                           tap_rel_cv_only
agg                                  score_top5
scenario                                    ds4
feature_count                                27
rows                                        255
pfa10_q90_normal_flag_rate                  NaN
pfa5_q95_normal_flag_rate                   NaN
pfa1_q99_normal_flag_rate                   NaN
pfa0_5_q995_normal_flag_rate                NaN
pfa0_1_q999_normal_flag_rate                NaN
pre_windows_t_lt_90                       179.0
post_windows_t_ge_110                      36.0
auc_pre_vs_post_buffered                 0.8518
pre_score_median                       1.687607
post_score_median                      2.545771
pre_score_q95                          2.985442
post_score_q95                         2.758357
pfa10_q90_pre_fp_rate                   0.01676
pfa10_q90_post_det_rate                     0.0
pfa10_q90_first_delay_s                     NaN
pfa5_q95_pre_fp_rate                   0.005587
pfa5_q95_post_det_rate                      0.0
pfa5_q95_first_delay_s                      NaN
pfa1_q99_pre_fp_rate                        0.0
pfa1_q99_post_det_rate                      0.0
pfa1_q99_first_delay_s                      NaN
pfa0_5_q995_pre_fp_rate                     0.0
pfa0_5_q995_post_det_rate                   0.0
pfa0_5_q995_first_delay_s                   NaN
pfa0_1_q999_pre_fp_rate                     0.0
pfa0_1_q999_post_det_rate                   0.0
pfa0_1_q999_first_delay_s                   NaN
Name: 59, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.852, q95 det=0.000
- morph_core_no_doppler_no_cn0_no_power/<bound method Series.aggregate of group                           morph_core_no_doppler_no_cn0_no_power
agg                                                        score_top5
scenario                                                          ds4
feature_count                                                      63
rows                                                              255
pfa10_q90_normal_flag_rate                                        NaN
pfa5_q95_normal_flag_rate                                         NaN
pfa1_q99_normal_flag_rate                                         NaN
pfa0_5_q995_normal_flag_rate                                      NaN
pfa0_1_q999_normal_flag_rate                                      NaN
pre_windows_t_lt_90                                             179.0
post_windows_t_ge_110                                            36.0
auc_pre_vs_post_buffered                                     0.850869
pre_score_median                                             1.762278
post_score_median                                            2.698968
pre_score_q95                                                3.260959
post_score_q95                                               2.892893
pfa10_q90_pre_fp_rate                                        0.011173
pfa10_q90_post_det_rate                                           0.0
pfa10_q90_first_delay_s                                           NaN
pfa5_q95_pre_fp_rate                                         0.011173
pfa5_q95_post_det_rate                                            0.0
pfa5_q95_first_delay_s                                            NaN
pfa1_q99_pre_fp_rate                                              0.0
pfa1_q99_post_det_rate                                            0.0
pfa1_q99_first_delay_s                                            NaN
pfa0_5_q995_pre_fp_rate                                           0.0
pfa0_5_q995_post_det_rate                                         0.0
pfa0_5_q995_first_delay_s                                         NaN
pfa0_1_q999_pre_fp_rate                                           0.0
pfa0_1_q999_post_det_rate                                         0.0
pfa0_1_q999_first_delay_s                                         NaN
Name: 139, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.851, q95 det=0.000
- tap_rel_prompt_only/<bound method Series.aggregate of group                           tap_rel_prompt_only
agg                                       score_max
scenario                                        ds4
feature_count                                     9
rows                                            255
pfa10_q90_normal_flag_rate                      NaN
pfa5_q95_normal_flag_rate                       NaN
pfa1_q99_normal_flag_rate                       NaN
pfa0_5_q995_normal_flag_rate                    NaN
pfa0_1_q999_normal_flag_rate                    NaN
pre_windows_t_lt_90                           179.0
post_windows_t_ge_110                          36.0
auc_pre_vs_post_buffered                   0.835351
pre_score_median                           2.071165
post_score_median                          4.479409
pre_score_q95                             10.726926
post_score_q95                             5.044721
pfa10_q90_pre_fp_rate                      0.011173
pfa10_q90_post_det_rate                         0.0
pfa10_q90_first_delay_s                         NaN
pfa5_q95_pre_fp_rate                       0.011173
pfa5_q95_post_det_rate                          0.0
pfa5_q95_first_delay_s                          NaN
pfa1_q99_pre_fp_rate                       0.011173
pfa1_q99_post_det_rate                          0.0
pfa1_q99_first_delay_s                          NaN
pfa0_5_q995_pre_fp_rate                    0.011173
pfa0_5_q995_post_det_rate                       0.0
pfa0_5_q995_first_delay_s                       NaN
pfa0_1_q999_pre_fp_rate                    0.011173
pfa0_1_q999_post_det_rate                       0.0
pfa0_1_q999_first_delay_s                       NaN
Name: 95, dtype: object>: q99 det=0.000, q99 FP=0.011, delay=nan, AUC=0.835, q95 det=0.000
- peak_shape_only/<bound method Series.aggregate of group                           peak_shape_only
agg                                  score_mean
scenario                                    ds4
feature_count                                 8
rows                                        255
pfa10_q90_normal_flag_rate                  NaN
pfa5_q95_normal_flag_rate                   NaN
pfa1_q99_normal_flag_rate                   NaN
pfa0_5_q995_normal_flag_rate                NaN
pfa0_1_q999_normal_flag_rate                NaN
pre_windows_t_lt_90                       179.0
post_windows_t_ge_110                      36.0
auc_pre_vs_post_buffered               0.833644
pre_score_median                       1.096945
post_score_median                      1.491209
pre_score_q95                          1.795319
post_score_q95                         1.605224
pfa10_q90_pre_fp_rate                   0.01676
pfa10_q90_post_det_rate                     0.0
pfa10_q90_first_delay_s                     NaN
pfa5_q95_pre_fp_rate                   0.011173
pfa5_q95_post_det_rate                      0.0
pfa5_q95_first_delay_s                      NaN
pfa1_q99_pre_fp_rate                        0.0
pfa1_q99_post_det_rate                      0.0
pfa1_q99_first_delay_s                      NaN
pfa0_5_q995_pre_fp_rate                     0.0
pfa0_5_q995_post_det_rate                   0.0
pfa0_5_q995_first_delay_s                   NaN
pfa0_1_q999_pre_fp_rate                     0.0
pfa0_1_q999_post_det_rate                   0.0
pfa0_1_q999_first_delay_s                   NaN
Name: 115, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.834, q95 det=0.000
- tap_rel_sum_only/<bound method Series.aggregate of group                           tap_rel_sum_only
agg                                   score_top5
scenario                                     ds4
feature_count                                  9
rows                                         255
pfa10_q90_normal_flag_rate                   NaN
pfa5_q95_normal_flag_rate                    NaN
pfa1_q99_normal_flag_rate                    NaN
pfa0_5_q995_normal_flag_rate                 NaN
pfa0_1_q999_normal_flag_rate                 NaN
pre_windows_t_lt_90                        179.0
post_windows_t_ge_110                       36.0
auc_pre_vs_post_buffered                0.831316
pre_score_median                        1.444367
post_score_median                       2.335503
pre_score_q95                           2.524465
post_score_q95                           2.75961
pfa10_q90_pre_fp_rate                    0.01676
pfa10_q90_post_det_rate                      0.0
pfa10_q90_first_delay_s                      NaN
pfa5_q95_pre_fp_rate                         0.0
pfa5_q95_post_det_rate                       0.0
pfa5_q95_first_delay_s                       NaN
pfa1_q99_pre_fp_rate                         0.0
pfa1_q99_post_det_rate                       0.0
pfa1_q99_first_delay_s                       NaN
pfa0_5_q995_pre_fp_rate                      0.0
pfa0_5_q995_post_det_rate                    0.0
pfa0_5_q995_first_delay_s                    NaN
pfa0_1_q999_pre_fp_rate                      0.0
pfa0_1_q999_post_det_rate                    0.0
pfa0_1_q999_first_delay_s                    NaN
Name: 107, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.831, q95 det=0.000
- dmcpd_only/<bound method Series.aggregate of group                           dmcpd_only
agg                             score_top3
scenario                               ds4
feature_count                           25
rows                                   255
pfa10_q90_normal_flag_rate             NaN
pfa5_q95_normal_flag_rate              NaN
pfa1_q99_normal_flag_rate              NaN
pfa0_5_q995_normal_flag_rate           NaN
pfa0_1_q999_normal_flag_rate           NaN
pre_windows_t_lt_90                  179.0
post_windows_t_ge_110                 36.0
auc_pre_vs_post_buffered          0.825885
pre_score_median                  2.197514
post_score_median                 3.601954
pre_score_q95                     4.773433
post_score_q95                    3.841616
pfa10_q90_pre_fp_rate              0.01676
pfa10_q90_post_det_rate                0.0
pfa10_q90_first_delay_s                NaN
pfa5_q95_pre_fp_rate              0.011173
pfa5_q95_post_det_rate                 0.0
pfa5_q95_first_delay_s                 NaN
pfa1_q99_pre_fp_rate              0.005587
pfa1_q99_post_det_rate                 0.0
pfa1_q99_first_delay_s                 NaN
pfa0_5_q995_pre_fp_rate                0.0
pfa0_5_q995_post_det_rate              0.0
pfa0_5_q995_first_delay_s              NaN
pfa0_1_q999_pre_fp_rate                0.0
pfa0_1_q999_post_det_rate              0.0
pfa0_1_q999_first_delay_s              NaN
Name: 39, dtype: object>: q99 det=0.000, q99 FP=0.006, delay=nan, AUC=0.826, q95 det=0.000
- tap_rel_sum_only/<bound method Series.aggregate of group                           tap_rel_sum_only
agg                                   score_top3
scenario                                     ds4
feature_count                                  9
rows                                         255
pfa10_q90_normal_flag_rate                   NaN
pfa5_q95_normal_flag_rate                    NaN
pfa1_q99_normal_flag_rate                    NaN
pfa0_5_q995_normal_flag_rate                 NaN
pfa0_1_q999_normal_flag_rate                 NaN
pre_windows_t_lt_90                        179.0
post_windows_t_ge_110                       36.0
auc_pre_vs_post_buffered                 0.80897
pre_score_median                         1.69121
post_score_median                         2.7545
pre_score_q95                           3.614113
post_score_q95                          3.131512
pfa10_q90_pre_fp_rate                    0.01676
pfa10_q90_post_det_rate                      0.0
pfa10_q90_first_delay_s                      NaN
pfa5_q95_pre_fp_rate                         0.0
pfa5_q95_post_det_rate                       0.0
pfa5_q95_first_delay_s                       NaN
pfa1_q99_pre_fp_rate                         0.0
pfa1_q99_post_det_rate                       0.0
pfa1_q99_first_delay_s                       NaN
pfa0_5_q995_pre_fp_rate                      0.0
pfa0_5_q995_post_det_rate                    0.0
pfa0_5_q995_first_delay_s                    NaN
pfa0_1_q999_pre_fp_rate                      0.0
pfa0_1_q999_post_det_rate                    0.0
pfa0_1_q999_first_delay_s                    NaN
Name: 103, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.809, q95 det=0.000
- tap_rel_cv_only/<bound method Series.aggregate of group                           tap_rel_cv_only
agg                                  score_top3
scenario                                    ds4
feature_count                                27
rows                                        255
pfa10_q90_normal_flag_rate                  NaN
pfa5_q95_normal_flag_rate                   NaN
pfa1_q99_normal_flag_rate                   NaN
pfa0_5_q995_normal_flag_rate                NaN
pfa0_1_q999_normal_flag_rate                NaN
pre_windows_t_lt_90                       179.0
post_windows_t_ge_110                      36.0
auc_pre_vs_post_buffered               0.808038
pre_score_median                        2.03018
post_score_median                      2.866626
pre_score_q95                          4.035647
post_score_q95                         3.121951
pfa10_q90_pre_fp_rate                   0.01676
pfa10_q90_post_det_rate                     0.0
pfa10_q90_first_delay_s                     NaN
pfa5_q95_pre_fp_rate                   0.005587
pfa5_q95_post_det_rate                      0.0
pfa5_q95_first_delay_s                      NaN
pfa1_q99_pre_fp_rate                        0.0
pfa1_q99_post_det_rate                      0.0
pfa1_q99_first_delay_s                      NaN
pfa0_5_q995_pre_fp_rate                     0.0
pfa0_5_q995_post_det_rate                   0.0
pfa0_5_q995_first_delay_s                   NaN
pfa0_1_q999_pre_fp_rate                     0.0
pfa0_1_q999_post_det_rate                   0.0
pfa0_1_q999_first_delay_s                   NaN
Name: 55, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.808, q95 det=0.000
- morph_core_no_doppler_no_cn0_no_power/<bound method Series.aggregate of group                           morph_core_no_doppler_no_cn0_no_power
agg                                                        score_top3
scenario                                                          ds4
feature_count                                                      63
rows                                                              255
pfa10_q90_normal_flag_rate                                        NaN
pfa5_q95_normal_flag_rate                                         NaN
pfa1_q99_normal_flag_rate                                         NaN
pfa0_5_q995_normal_flag_rate                                      NaN
pfa0_1_q999_normal_flag_rate                                      NaN
pre_windows_t_lt_90                                             179.0
post_windows_t_ge_110                                            36.0
auc_pre_vs_post_buffered                                     0.801831
pre_score_median                                             2.218259
post_score_median                                            3.067971
pre_score_q95                                                4.475792
post_score_q95                                               3.309677
pfa10_q90_pre_fp_rate                                        0.011173
pfa10_q90_post_det_rate                                           0.0
pfa10_q90_first_delay_s                                           NaN
pfa5_q95_pre_fp_rate                                         0.011173
pfa5_q95_post_det_rate                                            0.0
pfa5_q95_first_delay_s                                            NaN
pfa1_q99_pre_fp_rate                                              0.0
pfa1_q99_post_det_rate                                            0.0
pfa1_q99_first_delay_s                                            NaN
pfa0_5_q995_pre_fp_rate                                           0.0
pfa0_5_q995_post_det_rate                                         0.0
pfa0_5_q995_first_delay_s                                         NaN
pfa0_1_q999_pre_fp_rate                                           0.0
pfa0_1_q999_post_det_rate                                         0.0
pfa0_1_q999_first_delay_s                                         NaN
Name: 135, dtype: object>: q99 det=0.000, q99 FP=0.000, delay=nan, AUC=0.802, q95 det=0.000
- dmcpd_only/<bound method Series.aggregate of group                           dmcpd_only
agg                              score_max
scenario                               ds4
feature_count                           25
rows                                   255
pfa10_q90_normal_flag_rate             NaN
pfa5_q95_normal_flag_rate              NaN
pfa1_q99_normal_flag_rate              NaN
pfa0_5_q995_normal_flag_rate           NaN
pfa0_1_q999_normal_flag_rate           NaN
pre_windows_t_lt_90                  179.0
post_windows_t_ge_110                 36.0
auc_pre_vs_post_buffered           0.78352
pre_score_median                  2.574263
post_score_median                 4.025896
pre_score_q95                    10.638851
post_score_q95                    4.437074
pfa10_q90_pre_fp_rate              0.01676
pfa10_q90_post_det_rate                0.0
pfa10_q90_first_delay_s                NaN
pfa5_q95_pre_fp_rate               0.01676
pfa5_q95_post_det_rate                 0.0
pfa5_q95_first_delay_s                 NaN
pfa1_q99_pre_fp_rate              0.005587
pfa1_q99_post_det_rate                 0.0
pfa1_q99_first_delay_s                 NaN
pfa0_5_q995_pre_fp_rate           0.005587
pfa0_5_q995_post_det_rate              0.0
pfa0_5_q995_first_delay_s              NaN
pfa0_1_q999_pre_fp_rate           0.005587
pfa0_1_q999_post_det_rate              0.0
pfa0_1_q999_first_delay_s              NaN
Name: 47, dtype: object>: q99 det=0.000, q99 FP=0.006, delay=nan, AUC=0.784, q95 det=0.000
- morph_core_no_doppler_no_cn0_no_power/<bound method Series.aggregate of group                           morph_core_no_doppler_no_cn0_no_power
agg                                                         score_max
scenario                                                          ds4
feature_count                                                      63
rows                                                              255
pfa10_q90_normal_flag_rate                                        NaN
pfa5_q95_normal_flag_rate                                         NaN
pfa1_q99_normal_flag_rate                                         NaN
pfa0_5_q995_normal_flag_rate                                      NaN
pfa0_1_q999_normal_flag_rate                                      NaN
pre_windows_t_lt_90                                             179.0
post_windows_t_ge_110                                            36.0
auc_pre_vs_post_buffered                                     0.768001
pre_score_median                                             2.540082
post_score_median                                            3.420119
pre_score_q95                                                9.669478
post_score_q95                                               3.750909
pfa10_q90_pre_fp_rate                                        0.011173
pfa10_q90_post_det_rate                                           0.0
pfa10_q90_first_delay_s                                           NaN
pfa5_q95_pre_fp_rate                                         0.011173
pfa5_q95_post_det_rate                                            0.0
pfa5_q95_first_delay_s                                            NaN
pfa1_q99_pre_fp_rate                                         0.011173
pfa1_q99_post_det_rate                                            0.0
pfa1_q99_first_delay_s                                            NaN
pfa0_5_q995_pre_fp_rate                                      0.005587
pfa0_5_q995_post_det_rate                                         0.0
pfa0_5_q995_first_delay_s                                         NaN
pfa0_1_q999_pre_fp_rate                                      0.005587
pfa0_1_q999_post_det_rate                                         0.0
pfa0_1_q999_first_delay_s                                         NaN
Name: 143, dtype: object>: q99 det=0.000, q99 FP=0.011, delay=nan, AUC=0.768, q95 det=0.000
- tap_rel_cv_only/<bound method Series.aggregate of group                           tap_rel_cv_only
agg                                   score_max
scenario                                    ds4
feature_count                                27
rows                                        255
pfa10_q90_normal_flag_rate                  NaN
pfa5_q95_normal_flag_rate                   NaN
pfa1_q99_normal_flag_rate                   NaN
pfa0_5_q995_normal_flag_rate                NaN
pfa0_1_q999_normal_flag_rate                NaN
pre_windows_t_lt_90                       179.0
post_windows_t_ge_110                      36.0
auc_pre_vs_post_buffered               0.763035
pre_score_median                       2.357247
post_score_median                      3.202341
pre_score_q95                          8.588128
post_score_q95                         3.486287
pfa10_q90_pre_fp_rate                  0.011173
pfa10_q90_post_det_rate                     0.0
pfa10_q90_first_delay_s                     NaN
pfa5_q95_pre_fp_rate                   0.011173
pfa5_q95_post_det_rate                      0.0
pfa5_q95_first_delay_s                      NaN
pfa1_q99_pre_fp_rate                   0.011173
pfa1_q99_post_det_rate                      0.0
pfa1_q99_first_delay_s                      NaN
pfa0_5_q995_pre_fp_rate                0.005587
pfa0_5_q995_post_det_rate                   0.0
pfa0_5_q995_first_delay_s                   NaN
pfa0_1_q999_pre_fp_rate                     0.0
pfa0_1_q999_post_det_rate                   0.0
pfa0_1_q999_first_delay_s                   NaN
Name: 63, dtype: object>: q99 det=0.000, q99 FP=0.011, delay=nan, AUC=0.763, q95 det=0.000
- tap_rel_sum_only/<bound method Series.aggregate of group                           tap_rel_sum_only
agg                                    score_max
scenario                                     ds4
feature_count                                  9
rows                                         255
pfa10_q90_normal_flag_rate                   NaN
pfa5_q95_normal_flag_rate                    NaN
pfa1_q99_normal_flag_rate                    NaN
pfa0_5_q995_normal_flag_rate                 NaN
pfa0_1_q999_normal_flag_rate                 NaN
pre_windows_t_lt_90                        179.0
post_windows_t_ge_110                       36.0
auc_pre_vs_post_buffered                0.759932
pre_score_median                         2.02061
post_score_median                       3.211215
pre_score_q95                           8.073045
post_score_q95                          3.711821
pfa10_q90_pre_fp_rate                   0.011173
pfa10_q90_post_det_rate                      0.0
pfa10_q90_first_delay_s                      NaN
pfa5_q95_pre_fp_rate                    0.011173
pfa5_q95_post_det_rate                       0.0
pfa5_q95_first_delay_s                       NaN
pfa1_q99_pre_fp_rate                    0.005587
pfa1_q99_post_det_rate                       0.0
pfa1_q99_first_delay_s                       NaN
pfa0_5_q995_pre_fp_rate                      0.0
pfa0_5_q995_post_det_rate                    0.0
pfa0_5_q995_first_delay_s                    NaN
pfa0_1_q999_pre_fp_rate                      0.0
pfa0_1_q999_post_det_rate                    0.0
pfa0_1_q999_first_delay_s                    NaN
Name: 111, dtype: object>: q99 det=0.000, q99 FP=0.006, delay=nan, AUC=0.760, q95 det=0.000
- full_all_features/<bound method Series.aggregate of group                           full_all_features
agg                                    score_mean
scenario                                      ds4
feature_count                                  84
rows                                          255
pfa10_q90_normal_flag_rate                    NaN
pfa5_q95_normal_flag_rate                     NaN
pfa1_q99_normal_flag_rate                     NaN
pfa0_5_q995_normal_flag_rate                  NaN
pfa0_1_q999_normal_flag_rate                  NaN
pre_windows_t_lt_90                         179.0
post_windows_t_ge_110                        36.0
auc_pre_vs_post_buffered                 0.748293
pre_score_median                         2.085207
post_score_median                        2.304857
pre_score_q95                            2.883253
post_score_q95                           2.396968
pfa10_q90_pre_fp_rate                    0.145251
pfa10_q90_post_det_rate                  0.694444
pfa10_q90_first_delay_s                      14.5
pfa5_q95_pre_fp_rate                     0.111732
pfa5_q95_post_det_rate                   0.361111
pfa5_q95_first_delay_s                       15.0
pfa1_q99_pre_fp_rate                     0.094972
pfa1_q99_post_det_rate                        0.0
pfa1_q99_first_delay_s                        NaN
pfa0_5_q995_pre_fp_rate                  0.094972
pfa0_5_q995_post_det_rate                     0.0
pfa0_5_q995_first_delay_s                     NaN
pfa0_1_q999_pre_fp_rate                  0.094972
pfa0_1_q999_post_det_rate                     0.0
pfa0_1_q999_first_delay_s                     NaN
Name: 163, dtype: object>: q99 det=0.000, q99 FP=0.095, delay=nan, AUC=0.748, q95 det=0.361

## Top ds4 shifted features
- dmcpd_curvature_e1l1_mean: median |z| pre=0.668, post=3.097, delta=2.430, post_q95=8.271
- dmcpd_pair1_sum_to_prompt_mean: median |z| pre=0.668, post=3.097, delta=2.430, post_q95=8.271
- tap_E_rel_prompt_mean: median |z| pre=0.658, post=2.585, delta=1.927, post_q95=6.759
- dmcpd_second_side_to_prompt_mean: median |z| pre=0.669, post=2.557, delta=1.888, post_q95=6.539
- tap_L_rel_prompt_mean: median |z| pre=0.670, post=2.410, delta=1.740, post_q95=6.590
- dmcpd_width_variance_mean: median |z| pre=0.639, post=1.714, delta=1.075, post_q95=4.443
- dmcpd_pair2_sum_to_prompt_mean: median |z| pre=0.664, post=1.542, delta=0.878, post_q95=6.339
- dmcpd_max_side_to_prompt_mean: median |z| pre=0.637, post=1.513, delta=0.876, post_q95=3.900
- dmcpd_prompt_to_max_side_mean: median |z| pre=0.641, post=1.506, delta=0.865, post_q95=3.840
- peak_width_mean: median |z| pre=0.663, post=1.474, delta=0.811, post_q95=3.429
- tap_L_rel_sum_mean: median |z| pre=0.635, post=1.413, delta=0.778, post_q95=2.932
- tap_E4_rel_sum_mean: median |z| pre=0.662, post=1.424, delta=0.761, post_q95=3.863
- tap_E4_rel_prompt_mean: median |z| pre=0.667, post=1.421, delta=0.754, post_q95=4.433
- tap_E3_rel_prompt_mean: median |z| pre=0.664, post=1.368, delta=0.705, post_q95=5.316
- tap_E2_rel_prompt_mean: median |z| pre=0.656, post=1.351, delta=0.696, post_q95=5.707
- tap_E_rel_sum_mean: median |z| pre=0.636, post=1.322, delta=0.686, post_q95=3.042
- tap_L2_rel_prompt_mean: median |z| pre=0.654, post=1.337, delta=0.683, post_q95=4.502
- dmcpd_pair4_signed_asym_mean: median |z| pre=0.656, post=1.308, delta=0.652, post_q95=4.351
- dmcpd_pair3_sum_to_prompt_mean: median |z| pre=0.670, post=1.319, delta=0.650, post_q95=5.120
- tap_E3_rel_sum_mean: median |z| pre=0.643, post=1.285, delta=0.642, post_q95=4.334
- dmcpd_pair4_ratio_mean: median |z| pre=0.632, post=1.258, delta=0.626, post_q95=3.380
- tap_P_rel_sum_mean: median |z| pre=0.657, post=1.266, delta=0.608, post_q95=5.672
- dmcpd_prompt_dominance_mean: median |z| pre=0.657, post=1.266, delta=0.608, post_q95=5.672
- dmcpd_pair4_sum_to_prompt_mean: median |z| pre=0.673, post=1.256, delta=0.583, post_q95=3.821
- dmcpd_pair3_ratio_mean: median |z| pre=0.637, post=1.165, delta=0.529, post_q95=3.279
- peak_sharpness_mean: median |z| pre=0.687, post=1.204, delta=0.517, post_q95=5.560
- dmcpd_pair3_signed_asym_mean: median |z| pre=0.650, post=1.139, delta=0.489, post_q95=4.201
- dmcpd_centroid_shift_mean: median |z| pre=0.651, post=1.137, delta=0.485, post_q95=3.914
- tap_L3_rel_prompt_mean: median |z| pre=0.636, post=1.110, delta=0.474, post_q95=3.601
- tap_E2_rel_sum_mean: median |z| pre=0.647, post=1.078, delta=0.430, post_q95=3.507
