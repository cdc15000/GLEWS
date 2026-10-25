#!/bin/zsh
# Commit and push ONLY alarms/e4_2027/ for the 2027 prospective test
# (docs/prereg_2027_prospective.md). This is the standing, scoped push
# authorisation: it refuses to commit or push anything else.
#
# Refuses when: not on main; anything is staged; an unpushed local commit
# touches a file outside alarms/e4_2027/; or the remote has commits that the
# local branch lacks. Never force-pushes. Other modified files in the working
# tree are left alone and are not committed.
set -euo pipefail

REPO=${GLEWS_REPO:-${0:A:h:h:h}}
REMOTE=${GLEWS_REMOTE:-origin}
BRANCH=main
ALARMS=alarms/e4_2027
cd "$REPO"

fail() { print -u2 "push_alarm_log: $1"; exit 1 }

[[ $(git rev-parse --abbrev-ref HEAD) == $BRANCH ]] || fail "not on $BRANCH"
git diff --cached --quiet || fail "other changes are staged; refusing"
git fetch -q $REMOTE $BRANCH
git merge-base --is-ancestor $REMOTE/$BRANCH HEAD || fail "$REMOTE/$BRANCH has commits not in the local branch; pull first"
for c in $(git rev-list $REMOTE/$BRANCH..HEAD); do
  others=$(git diff-tree --no-commit-id --name-only -r --root $c | grep -v "^$ALARMS/" || true)
  [[ -z $others ]] || fail "unpushed commit ${c[1,8]} touches files outside $ALARMS; refusing"
done

if [[ -n $(git status --porcelain -- $ALARMS) ]]; then
  git add -- $ALARMS
  git commit -q -m "E4 2027 alarm log: run $(date -u +%Y-%m-%dT%H:%MZ)" -- $ALARMS
fi
if [[ -z $(git rev-list $REMOTE/$BRANCH..HEAD) ]]; then
  print "push_alarm_log: nothing new"
  exit 0
fi
git push -q $REMOTE HEAD:$BRANCH
print "push_alarm_log: pushed $(git rev-parse --short HEAD)"
