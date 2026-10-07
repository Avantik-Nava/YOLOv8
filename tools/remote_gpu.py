"""SSH helpers for running inference on a remote GPU box.

Local machine keeps ALL code. Only these go to the server per job:
  model.onnx + infer_onnx.py (standalone) + input video.
Result mp4 is downloaded and played locally. Nothing is git-pushed.
Needs: paramiko (local only).
"""

import io
import time

REMOTE_SUBDIR = "yolo_remote"


def _rdir(client):
    """Absolute remote work dir. (SFTP does NOT expand ~, so resolve $HOME.)"""
    code, out, _ = run(client, "echo $HOME", timeout=15)
    home = (out or "").strip()
    if code != 0 or not home:
        raise RuntimeError("could not resolve $HOME on the server")
    rdir = f"{home}/{REMOTE_SUBDIR}"
    code, _, err = run(client, f"mkdir -p {rdir}", timeout=30)
    if code != 0:
        raise RuntimeError(f"could not create {rdir}: {err.strip()}")
    return rdir


def connect(host, user, password=None, key_bytes=None, key_pass=None,
            port=22, timeout=15):
    """Returns (client, server_version). Raises with readable message on failure."""
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = None
    if key_bytes:
        for cls in ("RSAKey", "Ed25519Key", "ECDSAKey", "DSSKey"):
            try:
                kcls = getattr(paramiko, cls)
                pkey = kcls.from_private_key(io.BytesIO(key_bytes),
                                             password=key_pass or None)
                break
            except Exception:
                continue
        if pkey is None:
            raise ValueError("Could not parse the key file (tried RSA/Ed25519/ECDSA/DSS).")
    client.connect(hostname=host, port=port, username=user, password=password,
                   pkey=pkey, timeout=timeout, banner_timeout=20, auth_timeout=20,
                   look_for_keys=False, allow_agent=False)
    transport = client.get_transport()
    version = transport.remote_version if transport else "?"
    return client, version


def run(client, cmd, timeout=3600):
    """Run a remote command. Returns (exit_code, stdout, stderr)."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, stdout.read().decode(errors="ignore"), stderr.read().decode(errors="ignore")


def put(client, local, remote):
    sftp = client.open_sftp()
    try:
        sftp.put(str(local), remote)
    finally:
        sftp.close()


def get(client, remote, local):
    sftp = client.open_sftp()
    try:
        sftp.get(remote, str(local))
    finally:
        sftp.close()


def gpu_info(client):
    code, out, _ = run(client, "nvidia-smi --query-gpu=name,memory.total,driver_version "
                               "--format=csv,noheader 2>&1", timeout=30)
    return out.strip() if code == 0 and out.strip() else "no NVIDIA GPU visible"


def ensure_env(client, timeout=600):
    """Make sure python3 + onnxruntime(+gpu)/opencv/numpy exist. Returns report lines."""
    log = []
    code, out, _ = run(client, "python3 -c 'import onnxruntime as o; print(o.__version__)' 2>&1", timeout=60)
    if code == 0 and out.strip():
        log.append(f"onnxruntime {out.strip()} present")
    else:
        log.append("installing onnxruntime-gpu + opencv + numpy ...")
        code, out, err = run(client, "pip install -q onnxruntime-gpu opencv-python numpy 2>&1 "
                                     "|| pip3 install -q onnxruntime-gpu opencv-python numpy 2>&1",
                             timeout=timeout)
        log.append((out + err).strip()[-2000:] or "pip install finished")
    code, out, _ = run(client, "python3 -c 'import onnxruntime as o; print(o.get_available_providers())' 2>&1",
                       timeout=60)
    log.append("providers: " + (out.strip() or "?"))
    return log


def prepare_job(client, local_onnx, local_script="tools/infer_onnx.py"):
    """Upload model + standalone script. Returns remote paths dict."""
    rdir = _rdir(client)
    run(client, f"cp {rdir}/infer_onnx.py {rdir}/infer_onnx.py.bak 2>/dev/null; true")
    put(client, local_onnx, f"{rdir}/model.onnx")
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / local_script
    if not script.exists():
        raise RuntimeError(f"local script missing: {script}")
    put(client, script, f"{rdir}/infer_onnx.py")
    return {"dir": rdir, "model": f"{rdir}/model.onnx",
            "script": f"{rdir}/infer_onnx.py"}


def run_remote_infer(client, video_local, classes_local, conf, nms, imgsz,
                     batch, stride, remote, job="job"):
    """Upload inputs, run onnxruntime on the server (CUDA if present), return local result path + log."""
    import tempfile
    from pathlib import Path
    if not Path(video_local).exists():
        raise RuntimeError(f"local video missing: {video_local}")
    rdir = _rdir(client)
    put(client, video_local, f"{rdir}/{job}_in.mp4")
    if classes_local:
        put(client, classes_local, f"{rdir}/{job}_cls.txt")
    cls_arg = f"--classes {rdir}/{job}_cls.txt " if classes_local else ""
    cmd = (f"cd {rdir} && python3 infer_onnx.py --model {rdir}/model.onnx "
           f"--source {rdir}/{job}_in.mp4 --out {rdir}/{job}_out.mp4 {cls_arg}"
           f"--conf {conf} --nms {nms} --imgsz {imgsz} --batch {batch} --stride {stride}")
    t0 = time.time()
    code, out, err = run(client, cmd, timeout=7200)
    log = (out + ("\n" + err if err else "")).strip()
    if code != 0:
        raise RuntimeError(f"remote inference failed (exit {code}):\n{log[-3000:]}")
    local_out = Path(tempfile.mkdtemp()) / "result_remote.mp4"
    get(client, f"{rdir}/{job}_out.mp4", local_out)
    log += f"\n[remote wall time {time.time()-t0:.1f}s]"
    return local_out, log
