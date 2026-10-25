#!/bin/zsh
# Install (or with --uninstall, remove) the daily launchd job for the 2027
# prospective test. Run it yourself, after the freeze and after one manual
# run of daily_run.sh.
set -euo pipefail
REPO=${0:A:h:h:h}
DEST=$HOME/Library/LaunchAgents/org.glews.e4-2027.plist
if [[ ${1:-} == --uninstall ]]; then
  launchctl bootout gui/$(id -u) "$DEST" 2>/dev/null || true
  rm -f "$DEST"
  print "removed org.glews.e4-2027"
  exit 0
fi
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
sed -e "s|__REPO__|$REPO|g" -e "s|__HOME__|$HOME|g" "$REPO/scripts/prereg_2027/org.glews.e4-2027.plist" > "$DEST"
launchctl bootout gui/$(id -u) "$DEST" 2>/dev/null || true
launchctl bootstrap gui/$(id -u) "$DEST"
print "installed org.glews.e4-2027: daily at 07:00, log ~/Library/Logs/glews-e4-2027.log"
