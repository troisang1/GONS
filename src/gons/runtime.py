"""Shared runtime helpers for bootstrap-time commands and logging."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# GONS release marker: the project root is identified by `pyproject.toml`
# (an additional docs marker is not used in this release).
REPO_MARKERS = ("pyproject.toml",)
BOOTSTRAP_DIRECTORIES = (
    "artifacts/dev",
    "artifacts/models",
    "artifacts/reports",
    "artifacts/runs",
    "configs/ablations",
    "configs/baselines",
    "configs/data",
    "configs/logging",
    "configs/model",
    "configs/suites",
    "configs/versions",
    "data/interim",
    "data/manifests",
    "data/processed",
    "logs/development",
    "logs/events",
    "scripts",
    "src/gons",
    "tests/fixtures",
    "tests/integration",
    "tests/regression",
    "tests/unit",
)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved filesystem paths used by the bootstrap layer."""

    root: Path
    artifacts_dir: Path
    dev_artifacts_dir: Path
    logs_dir: Path
    development_logs_dir: Path
    progress_log_path: Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until the repository markers are found."""

    anchor = (start or Path(__file__)).resolve()
    candidates = (anchor, *anchor.parents)
    for candidate in candidates:
        if all((candidate / marker).exists() for marker in REPO_MARKERS):
            return candidate
    raise FileNotFoundError(
        f"Could not locate the repository root from {anchor}. Expected markers: {REPO_MARKERS}."
    )


def build_project_paths(root: Path) -> ProjectPaths:
    """Create the canonical project path bundle from an explicit root."""

    artifacts_dir = root / "artifacts"
    logs_dir = root / "logs"
    return ProjectPaths(
        root=root,
        artifacts_dir=artifacts_dir,
        dev_artifacts_dir=artifacts_dir / "dev",
        logs_dir=logs_dir,
        development_logs_dir=logs_dir / "development",
        progress_log_path=artifacts_dir / "dev" / "progress.jsonl",
    )


def project_paths(root: Path | None = None) -> ProjectPaths:
    """Resolve the canonical project path bundle."""

    return build_project_paths(find_repo_root(root))


def expected_bootstrap_paths(root: Path) -> list[Path]:
    """Return the directories that must exist for a runnable checkout."""

    return [root / relative_path for relative_path in BOOTSTRAP_DIRECTORIES]


def utc_now_iso() -> str:
    """Return the current UTC timestamp in an artifact-friendly format."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_run_id(name: str) -> str:
    """Create a stable slugged run identifier."""

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "run"
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{slug}"


def current_git_commit(root: Path) -> str | None:
    """Return the current git commit if the repository is under git control."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=root,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def python_runtime() -> str:
    """Return the active Python runtime version."""

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def host_name() -> str:
    """Return the current host name for environment snapshots."""

    return socket.gethostname()


def process_rss_bytes(pid: int | None) -> int | None:
    """Return resident memory for an arbitrary process, when the platform exposes it."""

    if pid is None or pid <= 0:
        return None
    for probe in (_rss_bytes_procfs, _rss_bytes_windows, _rss_bytes_macos, _rss_bytes_ps):
        value = probe(pid)
        if value is not None:
            return value
    return None


def current_process_rss_bytes() -> int | None:
    """Return resident memory for the current process, when available."""

    return process_rss_bytes(os.getpid())


def _rss_bytes_procfs(pid: int) -> int | None:
    status_path = Path("/proc") / str(pid) / "status"
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    statm_path = Path("/proc") / str(pid) / "statm"
    if statm_path.exists():
        parts = statm_path.read_text(encoding="utf-8").split()
        if len(parts) >= 2:
            return int(parts[1]) * os.sysconf("SC_PAGE_SIZE")
    return None


def _rss_bytes_windows(pid: int) -> int | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    process_query_information = 0x0400
    process_vm_read = 0x0010
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
    except AttributeError:
        return None
    handle = kernel32.OpenProcess(process_query_information | process_vm_read, False, pid)
    if not handle:
        return None
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def _rss_bytes_macos(pid: int) -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
    except ImportError:
        return None

    class ProcTaskInfo(ctypes.Structure):
        _fields_ = [
            ("pti_virtual_size", ctypes.c_uint64),
            ("pti_resident_size", ctypes.c_uint64),
            ("pti_total_user", ctypes.c_uint64),
            ("pti_total_system", ctypes.c_uint64),
            ("pti_threads_user", ctypes.c_uint64),
            ("pti_threads_system", ctypes.c_uint64),
            ("pti_policy", ctypes.c_int32),
            ("pti_faults", ctypes.c_int32),
            ("pti_pageins", ctypes.c_int32),
            ("pti_cow_faults", ctypes.c_int32),
            ("pti_messages_sent", ctypes.c_int32),
            ("pti_messages_received", ctypes.c_int32),
            ("pti_syscalls_mach", ctypes.c_int32),
            ("pti_syscalls_unix", ctypes.c_int32),
            ("pti_csw", ctypes.c_int32),
            ("pti_threadnum", ctypes.c_int32),
            ("pti_numrunning", ctypes.c_int32),
            ("pti_priority", ctypes.c_int32),
        ]

    proc_pidtaskinfo = 4
    try:
        libproc = ctypes.CDLL("libproc.dylib")
    except OSError:
        return None
    task_info = ProcTaskInfo()
    filled = libproc.proc_pidinfo(
        pid,
        proc_pidtaskinfo,
        0,
        ctypes.byref(task_info),
        ctypes.sizeof(task_info),
    )
    if filled <= 0:
        return None
    return int(task_info.pti_resident_size)


def _rss_bytes_ps(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, PermissionError, OSError, subprocess.CalledProcessError):
        return None
    rss_text = completed.stdout.strip()
    if not rss_text:
        return None
    try:
        return int(rss_text) * 1024
    except ValueError:
        return None
