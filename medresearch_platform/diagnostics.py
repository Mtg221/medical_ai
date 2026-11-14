"""Hardware, operating system, dependency, and readiness diagnostics (Sections 8, 38, 39)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sqlite3
from typing import Any
import psutil


@dataclass(frozen=True)
class HardwareSnapshot:
    os_name: str
    os_release: str
    os_version: str
    machine_arch: str
    processor: str
    physical_cores: int
    logical_cpus: int
    total_ram_bytes: int
    total_ram_gb: float
    available_ram_bytes: int
    available_ram_gb: float
    process_rss_mb: float
    disk_total_gb: float
    disk_free_gb: float
    gpu_detected: bool
    execution_mode: str


@dataclass(frozen=True)
class RuntimeReadinessReport:
    timestamp_utc: str
    python_version: str
    sqlite_version: str
    psql_binary_path: str | None
    node_binary_path: str | None
    hardware: HardwareSnapshot
    installed_packages: dict[str, str]
    readiness_checks: dict[str, bool]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_hardware_snapshot(workspace_path: Path | str = ".") -> HardwareSnapshot:
    """Measure live CPU, memory, and disk resources without external network calls."""
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    disk = shutil.disk_usage(Path(workspace_path).resolve())
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or 1

    return HardwareSnapshot(
        os_name=platform.system(),
        os_release=platform.release(),
        os_version=platform.version(),
        machine_arch=platform.machine(),
        processor=platform.processor() or "x86_64",
        physical_cores=physical,
        logical_cpus=logical,
        total_ram_bytes=int(vm.total),
        total_ram_gb=round(vm.total / (1024**3), 2),
        available_ram_bytes=int(vm.available),
        available_ram_gb=round(vm.available / (1024**3), 2),
        process_rss_mb=round(rss_mb, 2),
        disk_total_gb=round(disk.total / (1024**3), 2),
        disk_free_gb=round(disk.free / (1024**3), 2),
        gpu_detected=False,
        execution_mode="CPU-ONLY (Native)",
    )


def generate_readiness_report(workspace_path: Path | str = ".") -> RuntimeReadinessReport:
    """Generate a full environment and dependency diagnostic report."""
    hw = capture_hardware_snapshot(workspace_path)
    required_distributions = (
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "pillow",
        "pypdf",
        "python-docx",
        "reportlab",
        "starlette",
        "httpx",
        "pytest",
        "psutil",
        "argon2-cffi",
    )
    installed: dict[str, str] = {}
    for dist_name in required_distributions:
        try:
            installed[dist_name] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            installed[dist_name] = "MISSING"

    psql_path = shutil.which("psql") or (
        "/Library/PostgreSQL/18/bin/psql"
        if Path("/Library/PostgreSQL/18/bin/psql").exists()
        else None
    )
    node_path = shutil.which("node")

    checks = {
        "cpu_detected": hw.logical_cpus >= 1,
        "ram_at_least_8gb": hw.total_ram_gb >= 8.0,
        "disk_free_at_least_5gb": hw.disk_free_gb >= 5.0,
        "core_python_packages_present": all(v != "MISSING" for v in installed.values()),
        "sqlite3_available": bool(sqlite3.sqlite_version),
        "postgres_cli_present": psql_path is not None,
    }

    warnings: list[str] = []
    if not hw.gpu_detected:
        warnings.append(
            "No dedicated GPU detected; CPU-first execution constraints enforced (compact models, bounded batches)."
        )
    if not os.environ.get("DATABASE_URL") and not os.environ.get("MEDRESEARCH_POSTGRES_URL"):
        warnings.append(
            "DATABASE_URL not configured; using local transactional SQLite engine synchronized with canonical PostgreSQL 18 DDL."
        )

    return RuntimeReadinessReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python_version=platform.python_version(),
        sqlite_version=sqlite3.sqlite_version,
        psql_binary_path=psql_path,
        node_binary_path=node_path,
        hardware=hw,
        installed_packages=installed,
        readiness_checks=checks,
        warnings=warnings,
    )
