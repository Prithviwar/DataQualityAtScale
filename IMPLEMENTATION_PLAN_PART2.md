# Part 2: Verify, Analyze, and Document

Part 2 consumes the clean/corrupted Parquet, injection metadata, validation
results, and quarantine outputs produced by [Part 1](./IMPLEMENTATION_PLAN_PART1.md).

## Model-impact experiment

Implement `src/model_simulation.py` or an equivalent deterministic simulator.
Measure overall accuracy and accuracy for subgroups A and B in three states:

| State | Overall | A | B |
|---|---:|---:|---:|
| Clean baseline | ~88.4% | ~89.1% | ~87.6% |
| Corrupted, no checks | ~71.2% | ~76.4% | ~61.8% |
| Validated/quarantined | ~87.9% | ~88.5% | ~87.1% |

Treat these as targets, not values to fabricate. Report exact, approximate, or
controlled-simulation status. The validated state must explicitly say whether
it uses clean records, remediated records, or the original baseline after
rejecting a bad batch. Subgroup B must lose materially more accuracy than A.

Write:

- `output/accuracy_comparison.csv`
- `output/runtime_metrics.json`

Include absolute and relative changes, subgroup counts, and the policy used
for rejected records. The target differences are 17.2 points overall, 12.7
points for A, and 25.8 points for B.

## Runtime and scalability

Measure generation, injection, validation, model evaluation, and total runtime.
Record machine/container resources and, where available, driver/executor
memory. Use Spark aggregations, approximate statistics, partitioned Parquet,
and configurable shuffle partitions. Do not collect 100 million rows.

Report the measured validation overhead against the report target of about
4.2 minutes. Explain hardware/environment differences instead of fabricating
the target.

## Tests

Add focused tests under `tests/`:

- seeded reproducibility;
- null, scale-shift, and duplicate rates/counts;
- completeness, uniqueness, range/distribution, and clean-data checks;
- invalid-batch quarantine and non-zero failure behavior;
- accuracy-difference calculations.

Run a small integration workflow (for example 10,000 rows):

1. Generate clean data.
2. Inject all faults.
3. Validate with `collect_all`.
4. Quarantine the invalid batch.
5. Confirm inference is not run for that batch.
6. Produce metrics and report files.

Acceptance thresholds:

- each injected rate within ±0.1 percentage points of its target;
- clean data passes all blocking checks;
- corrupted data fails at least one blocking check;
- all three injected categories are detected in the demonstration;
- subgroup B degradation exceeds subgroup A degradation;
- failed data cannot reach inference.

## Analysis notebook

Create `results_analysis.ipynb` to load output files and:

1. show injected rates and PyDeequ outcomes;
2. compare clean, corrupted, and validated accuracy;
3. calculate accuracy changes and subgroup disparity;
4. compare `feature_2` distributions before/after corruption;
5. show validation overhead and fault detection;
6. export tables and figures to `data/reports/`.

Required visuals: grouped accuracy bars (overall/A/B), `feature_2`
distribution comparison, null/duplicate summary, runtime chart, and fault
detection matrix. Keep notebook outputs reproducible from saved files.

## README

Create `README.md` covering:

- motivation and architecture flow;
- prerequisites and local/Docker setup;
- development and 100-million-row commands;
- quality rules, reference-bound calibration, and quarantine behavior;
- output files and result interpretation;
- reproducibility, resource assumptions, known limitations, and deviations from
  target values.

Explain why overall accuracy alone is insufficient and why subgroup B needs
separate monitoring. Include a text architecture flow if no diagram is added.

## Completion checklist

- [ ] Clean and corrupted datasets are reproducible.
- [ ] All three fault vectors are configurable and measured.
- [ ] PyDeequ detects missing values, duplicate IDs, and scale corruption.
- [ ] Invalid data is quarantined before inference.
- [ ] Results and runtime metrics are machine-readable.
- [ ] Global and subgroup accuracy are reported.
- [ ] Notebook exports required charts/tables.
- [ ] Docker Compose runs the workflow.
- [ ] Unit and small integration tests pass.
- [ ] README documents commands, assumptions, thresholds, and measured results.
