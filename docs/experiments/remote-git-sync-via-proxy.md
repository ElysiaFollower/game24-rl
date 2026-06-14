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
