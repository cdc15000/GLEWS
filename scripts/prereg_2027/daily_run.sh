#!/bin/zsh
# Daily job for the 2027 prospective test (docs/prereg_2027_prospective.md),
# run by launchd (org.glews.e4-2027.plist).
#
# 1. Refuses unless tag e4-frozen-2027 exists and the frozen code in the
#    working tree is identical to it.
# 2. From 15 Feb 2027: eligibility once (ASF metadata only).
# 3. To 10 Jun 2027: a live run (new GOFF pairs -> frozen E4 -> alarm log),
#    then push_alarm_log.sh, which commits and pushes only alarms/e4_2027/.
# Failures raise a macOS notification; output goes to the launchd log.
set -euo pipefail

REPO=${GLEWS_REPO:-${0:A:h:h:h}}
PY=${GLEWS_PY:-$REPO/glews/bin/python}
TAG=e4-frozen-2027
FROZEN=(src/glews/e4.py src/glews/goff_crop.py src/glews/tm.py src/glews/nisar.py src/glews/exposure.py
        scripts/prereg_2027/run_e4_live.py scripts/prereg_2027/push_alarm_log.sh)
TODAY=${GLEWS_TODAY:-$(date +%Y%m%d)}
cd "$REPO"

notify() { osascript -e "display notification \"$1\" with title \"GLEWS E4 2027\"" 2>/dev/null || true }
stop() { print -u2 "daily_run: $1"; notify "$1"; exit 1 }
trap 'notify "Daily run failed; see ~/Library/Logs/glews-e4-2027.log"' ERR

print "daily_run: $(date -u +%Y-%m-%dT%H:%MZ)"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || stop "tag $TAG is missing: the protocol is not frozen yet"
git diff --quiet $TAG -- $FROZEN || stop "frozen code differs from $TAG; not running"

if [[ $TODAY < 20270215 ]]; then print "daily_run: before the eligibility date (15 Feb 2027)"; exit 0; fi
[[ -f alarms/e4_2027/eligibility.json ]] || $PY -u scripts/prereg_2027/run_e4_live.py eligibility
if [[ $TODAY > 20270610 ]]; then print "daily_run: live season over"; exit 0; fi
$PY -u scripts/prereg_2027/run_e4_live.py run
zsh $REPO/scripts/prereg_2027/push_alarm_log.sh
