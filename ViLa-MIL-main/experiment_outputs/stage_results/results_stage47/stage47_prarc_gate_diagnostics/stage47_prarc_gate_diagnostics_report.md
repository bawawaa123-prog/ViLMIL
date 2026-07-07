# Step47 PRARC Gate Diagnostics Report

## Purpose
- Probe per-slide PRARC gates on the requested split using trained Step47 checkpoints.
- Check whether the gate has meaningful sample-adaptive spread instead of Step46 smoke-level near-constant behavior.

## Probe Status
- requested_variants: `prarc_v1_g05, prarc_v1_g08, prarc_v1_g10`
- split: `test`
- max_slides_per_fold: `0`
- probed_rows: `2904`

## Variant Diagnostics
- prarc_v1_g05: gate_std_mean=`0.002768640560822254` gate_range_mean=`0.013543564081192016` error_minus_correct=`-0.0026979808573009967` conflict_minus_nonconflict=`-0.008662194639222798` sample_adaptive_flag=`True`
- prarc_v1_g08: gate_std_mean=`0.0011475386210503785` gate_range_mean=`0.003925776481628418` error_minus_correct=`-0.0008785093956801271` conflict_minus_nonconflict=`-0.0024614717034565636` sample_adaptive_flag=`False`
- prarc_v1_g10: gate_std_mean=`5.236272175911539e-05` gate_range_mean=`0.00016201734542846679` error_minus_correct=`-2.5517751217174478e-05` conflict_minus_nonconflict=`-5.585680521580372e-05` sample_adaptive_flag=`False`
