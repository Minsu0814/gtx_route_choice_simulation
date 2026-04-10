# Raptor Calibration AutoResearch - Program

## Goal

Find Raptor GC parameters that maximize **composite match rate** between Raptor GC-optimal routes and SC (smart card) actual chosen routes.

## Background

- Raptor generates transit routes with generalized cost (GC)
- GC = board + transfer*n_transfers + wait_rel*wait + bus_rel*ivt_bus + train_rel*(ivt_train+ivt_gtx) + walk_rel\*walk
- SC data shows actual route choices by passengers
- Default params (all=1.0) give ~48% GC accuracy
- Grid search proxy found candidates up to ~63% (within existing choice set)
- Now we validate with **actual Raptor routing** on 2,000 OD sample

## Composite Score

```
score = 0.3*gc_accuracy + 0.3*category_match + 0.2*duration_180s + 0.2*transfer_match
```

- gc_accuracy: Raptor GC-optimal route == SC chosen route (exact match)
- category_match: same transport_category (bus_only, train_only, bus+train, etc.)
- duration_180s: |Raptor_duration - SC_duration| <= 180 seconds
- transfer_match: same number of transfers

## File Structure

```
program.md      <- This file (agent SOP, DO NOT modify)
prepare.py      <- Data loading + Raptor routing + evaluation (DO NOT modify)
calibrate.py    <- Parameter config (THE ONLY file agent modifies)
results.tsv     <- Experiment log (auto-recorded)
baseline.json   <- Current best performance (auto-updated)
```

## Rules

1. **Only modify calibrate.py.** Never touch prepare.py or program.md.
2. Each experiment must complete within **10 minutes (600 seconds)**.
3. Optimization target: **composite score** (higher is better).
4. Change EXPERIMENT_NAME before each experiment.
5. Results are auto-compared with baseline and auto-logged.

## Parameter Space

| Parameter          | Range    | MNL hint | Description                 |
| ------------------ | -------- | -------- | --------------------------- |
| transfer_cost_secs | 120~4800 | ~9387s   | Transfer penalty in seconds |
| walk_reluctance    | 1.0~12.0 | ~10.1    | Walk time multiplier        |
| bus_reluctance     | 0.3~2.0  | -        | Bus IVT multiplier          |
| train_reluctance   | 0.3~2.0  | -        | Train/GTX IVT multiplier    |
| wait_reluctance    | 1.0~3.0  | -        | Wait time multiplier        |

Grid search proxy results (top candidates):

- transfer=600, walk=10, bus=1.0, train=0.3 → proxy score 0.850
- transfer=1200, walk=10, bus=1.5, train=0.3 → proxy score 0.859
- transfer=600, walk=6, bus=1.0, train=0.5 → proxy score 0.850

## Experiment Loop

```
1. Read calibrate.py to see current params and results.tsv for history.
2. Choose next params to try:
   - Start from grid search proxy top candidates
   - Then explore nearby values (fine-tuning)
   - Try to understand which params matter most
3. Update EXPERIMENT_NAME and PARAMS in calibrate.py.
4. Run: python calibrate.py
5. Check results:
   - score > baseline → SUCCESS (auto-updated)
   - score <= baseline → FAIL (auto-logged, try different params)
6. Review results.tsv for patterns.
7. Go to step 2.
```

## What You Can Modify (in calibrate.py)

- EXPERIMENT_NAME: descriptive name for each run
- PARAMS dict: the 5 Raptor parameters
- N_SAMPLE: number of ODs to test (default 2000, max 5000)

## Constraints (DO NOT change)

- GTFS data is fixed (data/gtfs/a1/)
- SC training data is fixed
- Evaluation logic in prepare.py is fixed
- Raptor engine (dtumos_raptor) is fixed

## Start

```bash
# First run with proxy-optimal params
python calibrate.py

# Then iterate: modify calibrate.py -> run -> check results
```
