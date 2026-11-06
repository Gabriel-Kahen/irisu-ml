# October event-search recovery

`original/` preserves the seven recovered source files used by the October
combo event search. These snapshots are evidence, not standalone entry points:
the retired training, CPU IPC and MPC dependencies are not restored.

The recovered `rl_combo_diverse_search.py` is byte-identical to the historical
runner recorded in the winning search plan, SHA-256
`a800d330a3e53c37eff711d60bcf406fe644f956ac3022d0258d05344e90b875`.
`recovery-manifest.json` identifies successful file-change records and hashes
for each recovered file. The records came from the long-running September 25
chat and its September 29 / October 1 child chats. The winning invocation is
recorded separately in
`benchmarks/results/record-search-historical-provenance-2026-11-03.json`.

`python/irisu_search/historical/` retains the proposal allocation, event-option
executors and paired guards, historical counters, value functions, and diverse
beam selection. Its small transport bridge calls the resident simulator rather
than the retired CPU IPC worker. Training, NRPA, MPC finishing, archive campaign
scheduling, and the old command-line entry points are omitted. Replay encoding,
checkpoint ownership, persistence and independent record verification belong to
the new transport driver; they do not change the recovered planner functions.

The historical regression tests cover proposal geometry, event counters,
reaiming, button release, causal guards, retirement and feeder preparation.
`test_search_historical_source.py` checks original hashes and compares the
retained function bodies with the snapshots.
