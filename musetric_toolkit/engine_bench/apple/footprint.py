from __future__ import annotations

import ctypes
import json
import select
import subprocess
import sys
import time

FIELDS = [
    "user_time",
    "system_time",
    "pkg_idle_wkups",
    "interrupt_wkups",
    "pageins",
    "wired_size",
    "resident_size",
    "phys_footprint",
    "proc_start_abstime",
    "proc_exit_abstime",
    "child_user_time",
    "child_system_time",
    "child_pkg_idle_wkups",
    "child_interrupt_wkups",
    "child_pageins",
    "child_elapsed_abstime",
    "diskio_bytesread",
    "diskio_byteswritten",
    "cpu_time_qos_default",
    "cpu_time_qos_maintenance",
    "cpu_time_qos_background",
    "cpu_time_qos_utility",
    "cpu_time_qos_legacy",
    "cpu_time_qos_user_initiated",
    "cpu_time_qos_user_interactive",
    "billed_system_time",
    "serviced_system_time",
    "logical_writes",
    "lifetime_max_phys_footprint",
    "instructions",
    "cycles",
    "billed_energy",
    "serviced_energy",
    "interval_max_phys_footprint",
    "runnable_time",
]


class RusageInfoV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in FIELDS
    ]


LIB = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
LIB.proc_pid_rusage.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(RusageInfoV4),
]
LIB.responsibility_get_pid_responsible_for_pid.argtypes = [ctypes.c_int]
LIB.responsibility_get_pid_responsible_for_pid.restype = ctypes.c_int


def page_processes(target: int) -> list[tuple[int, str]]:
    owners = {target, LIB.responsibility_get_pid_responsible_for_pid(target)}
    listing = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,comm="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = []
    for line in listing.splitlines():
        pid_text, ppid_text, command = line.split(None, 2)
        pid, ppid = int(pid_text), int(ppid_text)
        if ("Renderer)" in command and ppid == target) or (
            "com.apple.WebKit.WebContent" in command
            and LIB.responsibility_get_pid_responsible_for_pid(pid) in owners
        ):
            found.append((pid, command.rsplit("/", 1)[-1]))
    return found


def main() -> None:
    responsible = int(sys.argv[1])
    interval = int(sys.argv[2]) / 1000
    processes: list[tuple[int, str]] = []
    listed = 0.0
    while True:
        now = time.time()
        if now - listed > 1:
            processes = page_processes(responsible)
            listed = now
        rows = []
        for pid, name in processes:
            usage = RusageInfoV4()
            if LIB.proc_pid_rusage(pid, 4, ctypes.byref(usage)) != 0:
                continue
            rows.append(
                {
                    "pid": pid,
                    "name": name,
                    "footprintMB": round(usage.phys_footprint / 2**20, 1),
                    "lifetimeMaxMB": round(
                        usage.lifetime_max_phys_footprint / 2**20, 1
                    ),
                }
            )
        sys.stdout.write(json.dumps({"t": round(now * 1000), "procs": rows}) + "\n")
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], interval)
        if ready and not sys.stdin.read(1):
            return


if __name__ == "__main__":
    main()
