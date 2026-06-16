# Remote Git Sync Via Proxy

## 2026-06-14

When the training machine cannot reach GitHub directly, use a local proxy
forwarded over SSH and keep the repo remote unchanged.

### Formula

1. On the local machine, open a reverse tunnel to the training box:

```sh
ssh -fN -R 127.0.0.1:2080:127.0.0.1:2080 AutoDL4090
```

2. On the training machine, run normal Git commands through the forwarded proxy:

```sh
export https_proxy=http://127.0.0.1:2080
export http_proxy=http://127.0.0.1:2080
git pull --ff-only
```

### Notes

- This keeps the key sync path formulaic.
- GitHub HTTPS access from the machine itself may still fail without the proxy.
- The forwarded proxy was verified with `curl https://github.com` returning 200.

## Execution-Only Remote Policy

The training machine is an execution worker, not a second development
worktree.

Source-of-truth rules:

- Code, scripts, configs, and docs are authored in the local repository and
  synchronized through GitHub.
- The remote machine should get code by `git clone`, `git fetch`, or
  `git pull --ff-only`; do not hand-edit or patch tracked files on the remote.
- Remote outputs belong under ignored artifact paths such as `outputs/`,
  `data/processed/`, checkpoint directories, and logs.
- If a temporary diagnostic script is needed, add it to the local repository
  first and sync it through git unless the owner explicitly authorizes a one-off
  upload.
- If the remote worktree becomes dirty, do not treat it as a source of truth.
  Preserve or inspect the dirty state, then reconcile from the local committed
  branch under owner approval.

Operational consequence:

- Before a remote training or evaluation run, confirm the intended local commit
  is pushed.
- On the remote, use the proxy formula above and then `git pull --ff-only`.
- If `git pull --ff-only` fails because the remote is dirty, stop and decide
  whether to stash, archive, or discard the remote-only changes.
