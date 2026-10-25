# 2027 prospective test

Scripts for the pre-registered prospective test of the frozen E4 detector
(`docs/prereg_2027_prospective.md`). The detector is `src/glews/e4.py` and
the crop extractor `src/glews/goff_crop.py`. Crops land in
`data/nisar/prereg_2027/` (gitignored). The public alarm log is
`alarms/e4_2027/`.

| Script | Purpose |
|--------|---------|
| `run_e4_live.py eligibility` | On or after 15 Feb 2027, from ASF metadata only: track and baseline-pair count for each `config/hkh_priority.yaml` site; writes `alarms/e4_2027/eligibility.json` |
| `run_e4_live.py run` | For each eligible site, crops new GOFF pairs, runs frozen E4 and appends one line per new cutoff to `alarms/e4_2027/alarm_log.jsonl`. Pushes nothing |
| `run_e4_live.py replay NAME GOFF_DIR LAT LON SEASON LOG` | Dry run over an existing crop directory (seen data only), written to `LOG` |
