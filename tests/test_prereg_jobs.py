"""Guards of the 2027 scheduled job (scripts/prereg_2027/*.sh), on throwaway git repos with a local bare remote."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUSH = ROOT / "scripts/prereg_2027/push_alarm_log.sh"
DAILY = ROOT / "scripts/prereg_2027/daily_run.sh"
pytestmark = pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh not available")
FROZEN = ["src/glews/e4.py", "src/glews/goff_crop.py", "src/glews/tm.py", "src/glews/nisar.py",
          "src/glews/exposure.py", "scripts/prereg_2027/run_e4_live.py", "scripts/prereg_2027/push_alarm_log.sh"]


def git(cwd, *a, check=True):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=check).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    work = tmp_path / "work"
    git(tmp_path, "clone", "-q", str(remote), str(work))
    git(work, "config", "user.email", "test@example.com")
    git(work, "config", "user.name", "Test")
    git(work, "checkout", "-q", "-b", "main")
    for f in FROZEN + ["README.md"]:
        (work / f).parent.mkdir(parents=True, exist_ok=True)
        (work / f).write_text(f"{f}\n")
    shutil.copy(PUSH, work / "scripts/prereg_2027/push_alarm_log.sh")         # the real guard, run by daily_run
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "init")
    git(work, "push", "-q", "origin", "main")
    return work, remote


def push(work, **env):
    e = dict(os.environ, GLEWS_REPO=str(work), **env)
    return subprocess.run(["zsh", str(PUSH)], capture_output=True, text=True, env=e)


def remote_files(remote):
    return set(git(remote, "ls-tree", "-r", "--name-only", "main").splitlines())


def add_alarm(work, text="{}\n"):
    p = work / "alarms/e4_2027/alarm_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fo:
        fo.write(text)


class TestPushAlarmLog:
    def test_pushes_only_the_alarm_log(self, repo):
        work, remote = repo
        add_alarm(work)
        (work / "README.md").write_text("edited, not for pushing\n")           # unstaged change elsewhere
        r = push(work)
        assert r.returncode == 0, r.stderr
        assert "alarms/e4_2027/alarm_log.jsonl" in remote_files(remote)
        assert git(remote, "show", "main:README.md") == "README.md"          # edit stayed local
        assert git(work, "status", "--porcelain") == "M README.md"

    def test_nothing_new(self, repo):
        work, _ = repo
        r = push(work)
        assert r.returncode == 0 and "nothing new" in r.stdout

    def test_refuses_when_something_is_staged(self, repo):
        work, remote = repo
        add_alarm(work)
        (work / "README.md").write_text("staged\n")
        git(work, "add", "README.md")
        r = push(work)
        assert r.returncode != 0 and "staged" in r.stderr
        assert "alarms/e4_2027/alarm_log.jsonl" not in remote_files(remote)

    def test_refuses_unpushed_commit_outside_alarms(self, repo):
        work, remote = repo
        (work / "README.md").write_text("local commit\n")
        git(work, "commit", "-q", "-am", "local work")
        add_alarm(work)
        r = push(work)
        assert r.returncode != 0 and "outside alarms/e4_2027" in r.stderr
        assert git(remote, "show", "main:README.md") == "README.md"

    def test_refuses_when_remote_is_ahead(self, repo, tmp_path):
        work, remote = repo
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", str(remote), str(other))
        git(other, "config", "user.email", "o@example.com"); git(other, "config", "user.name", "O")
        (other / "README.md").write_text("remote change\n")
        git(other, "commit", "-q", "-am", "remote change"); git(other, "push", "-q", "origin", "main")
        add_alarm(work)
        r = push(work)
        assert r.returncode != 0 and "pull first" in r.stderr

    def test_retries_an_earlier_unpushed_alarm_commit(self, repo):
        work, remote = repo
        add_alarm(work)
        git(work, "add", "alarms"); git(work, "commit", "-q", "-m", "alarm commit whose push failed")
        r = push(work)
        assert r.returncode == 0 and "pushed" in r.stdout
        assert "alarms/e4_2027/alarm_log.jsonl" in remote_files(remote)


class TestDailyRun:
    def daily(self, work, today, py="/usr/bin/false"):
        e = dict(os.environ, GLEWS_REPO=str(work), GLEWS_PY=py, GLEWS_TODAY=today)
        return subprocess.run(["zsh", str(DAILY)], capture_output=True, text=True, env=e)

    def test_refuses_without_the_freeze_tag(self, repo):
        work, _ = repo
        r = self.daily(work, "20270301")
        assert r.returncode != 0 and "not frozen" in r.stderr

    def test_refuses_when_frozen_code_changed(self, repo):
        work, _ = repo
        git(work, "tag", "e4-frozen-2027")
        (work / "src/glews/e4.py").write_text("tampered\n")                   # uncommitted change counts
        r = self.daily(work, "20270301")
        assert r.returncode != 0 and "differs" in r.stderr

    def test_waits_for_eligibility_date_and_stops_after_season(self, repo):
        work, _ = repo
        git(work, "tag", "e4-frozen-2027")
        r = self.daily(work, "20261210")
        assert r.returncode == 0 and "before the eligibility date" in r.stdout
        (work / "alarms/e4_2027").mkdir(parents=True)
        (work / "alarms/e4_2027/eligibility.json").write_text("{}")
        r = self.daily(work, "20270701")
        assert r.returncode == 0 and "season over" in r.stdout

    def test_runs_in_season(self, repo, tmp_path):
        work, _ = repo
        git(work, "tag", "e4-frozen-2027")
        (work / "alarms/e4_2027").mkdir(parents=True)
        (work / "alarms/e4_2027/eligibility.json").write_text("{}")
        stub = tmp_path / "py"
        stub.write_text(f"#!/bin/zsh\nprint -r -- \"$*\" >> {tmp_path}/calls\n")
        stub.chmod(0o755)
        r = self.daily(work, "20270301", py=str(stub))
        assert (tmp_path / "calls").read_text().split() == ["-u", "scripts/prereg_2027/run_e4_live.py", "run"]
        assert r.returncode == 0, r.stderr                                    # then pushes the new eligibility file
