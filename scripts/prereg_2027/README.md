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
| `daily_run.sh` | The daily job. It refuses unless tag `e4-frozen-2027` exists and the frozen code in the working tree is identical to it. Runs eligibility once from 15 Feb 2027, then `run` and `push_alarm_log.sh` until 10 Jun 2027 |
| `push_alarm_log.sh` | Commits and pushes **only** `alarms/e4_2027/`. Refuses if anything is staged, an unpushed commit touches other files, or the remote is ahead. Never force-pushes |
| `org.glews.e4-2027.plist`, `install_launchd.sh [--uninstall]` | launchd job, daily at 07:00 (a missed run fires when the Mac wakes); log in `~/Library/Logs/glews-e4-2027.log` |

## Installing the daily job (after the freeze)

1. Freeze: commit `docs/prereg_2027_prospective.md`, then tag `e4-frozen-2027` and push the tag.
2. Run the job once by hand and read its output: `zsh scripts/prereg_2027/daily_run.sh`.
3. Install it: `zsh scripts/prereg_2027/install_launchd.sh` (remove it with `--uninstall`).
4. Optional: to guarantee the Mac is awake for the run, schedule a daily wake yourself:
   `sudo pmset repeat wakeorpoweron MTWRFSU 06:55:00`.
