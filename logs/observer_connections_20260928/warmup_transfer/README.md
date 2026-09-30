# Momentum warmup transfers to PD and replicates in SOAP-PD at the doubled horizon

**Completed replication update:** the second SOAP-PD seed now finishes at
3.964582 with warmup versus 3.988526 with constant beta .9, difference
**−.023944**. Both planned cohort pairs satisfy their original ≥.01 gain
criterion. The original SOAP-PD seed gained −.021193; PD gains −.026671
as detailed below. This is SOAP-PD across two seeds plus PD at one seed,
not both methods crossed with both seeds. See the
[replication review](SOAP_REPLICATION_REVIEW.md), `replication_result.json`
and `replication_extract.py`. Earlier PD-only records remain unchanged.

2026-09-28. This is an observer readout of newly completed **main-program**
training, not a new observer run. The matched PD pair satisfies its original
criterion of at least .01 NLL improvement over constant beta .9:

| Update | Warmup | Constant beta .9 | Difference |
|---:|---:|---:|---:|
| 50 | 5.344819 | 5.461515 | −.116697 |
| 100 | 4.494536 | 4.565437 | −.070901 |
| 150 | 4.176683 | 4.209169 | −.032485 |
| 184 | **4.017883** | **4.044554** | **−.026671** |

The pair shares initialization, data, source bytes, per-step body/auxiliary
LRs and token counts. Only the momentum ramp differs: .8→.9 over half the
token budget versus constant .9. Both use PD alpha .5, 16M batch, LR .028,
3.08B training tokens, seed 260925 and four L40S GPUs. All 25 source hashes
match between the two runs. The final shortened batch is retained.

The early lead narrows but remains after both runs use beta .9 from update
93 onward; common LR cooldown starts at update 167. This is a lasting
effect of different training histories, not evidence of a continuing
different-beta rate at the endpoint. Both runs clip nearly throughout,
so the finding is about the full matched recipe, not an isolated mediator.

This extends the practical schedule evidence beyond SOAP-PD in one paired
PD seed. It does not show superiority to a constant-beta-.8 PD run, which
is absent from this comparison. The new SOAP-PD reference was unfinished
at the initial PD review and has now completed as recorded above. The
earlier short-horizon SOAP-PD warmup failure remains.

Together with the observer's Q/K result, this favors keeping different
levels of evidence separate: real training transfer can coexist with an
adverse selected-step component margin. Neither that margin nor a static
gradient signal summary identifies why the schedule works. The failed
historical nesting gate in `../conditional_input/` does not erase this
independently recorded training comparison.

The [independent review](REVIEW.md) records source checks and all validation
points. [Trajectory figure](trajectory.png), `trajectory.csv`, `result.json`
and `extract.py` reproduce the scalar readout; no model or GPU call was used.
Retrospective training windows are descriptive, not added acceptance tests
or independent replications. Main checkpoints and validation records remain
at their original ordinary-NLL endpoints.
