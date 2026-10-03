"""Execution ledger backing the runtime idempotency check.

The CI executor can spot a duplicate transaction by reading a whole trace.
The runtime guard has no trace -- it only sees calls one at a time -- so it
needs a small ledger of what has already been executed.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol


class ExecutionStore(Protocol):
    def seen(self, key: str) -> bool: ...
    def record(self, key: str) -> None: ...
    def reserve(self, key: str) -> bool: ...
    def release(self, key: str) -> None: ...


class MemoryStore:
    """Process-local ledger.  Fine for tests and single-process agents.

    ``reserve`` is the method that matters under concurrency: it claims a key
    *and* reports whether the claim succeeded in one indivisible step.  The
    older ``seen`` + ``record`` pair is two steps, and the runtime guard used
    to sit between them for the whole duration of the tool call -- so twelve
    simultaneous charges on one order all saw an empty ledger and all went
    through.  Guarded by a lock rather than relying on the GIL: ``dict``
    operations are atomic individually, but a check-then-write pair is not,
    and the guarantee should not depend on which interpreter you are running.
    """

    def __init__(self, keys: list[str] | None = None) -> None:
        self._keys: dict[str, int] = {k: 1 for k in (keys or [])}
        self._lock = threading.Lock()

    def seen(self, key: str) -> bool:
        with self._lock:
            return key in self._keys

    def record(self, key: str) -> None:
        with self._lock:
            self._keys[key] = self._keys.get(key, 0) + 1

    def reserve(self, key: str) -> bool:
        """Claim *key* atomically.  ``True`` if this caller won the race."""
        with self._lock:
            if key in self._keys:
                return False
            self._keys[key] = 1
            return True

    def release(self, key: str) -> None:
        """Undo a reservation.  Used when the tool call itself failed, so a
        retry is legitimate rather than a duplicate."""
        with self._lock:
            self._keys.pop(key, None)

    def count(self, key: str) -> int:
        with self._lock:
            return self._keys.get(key, 0)

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return dict(self._keys)


class LedgerError(RuntimeError):
    """An existing ledger file could not be read.

    Starting over from an empty ledger would disable *every* idempotency check
    without a word -- the guard would cheerfully allow a second charge on the
    same order.  For a safety mechanism, refusing to run beats running blind.
    """


@contextmanager
def _exclusive(path: Path, timeout: float = 10.0) -> Iterator[None]:
    """Cross-process mutual exclusion for the read-modify-write on *path*.

    ``threading.Lock`` was not enough.  ``FileStore`` writes atomically, but
    atomicity of the *write* does not make the *read-modify-write* atomic: two
    processes both read an empty ledger, both decide their key is free, and both
    write.  Measured with 8 processes reserving one key: 2 believed they had
    won, with no error and a perfectly valid ledger -- a silent double charge,
    which is precisely the failure this project exists to prevent.

    Implemented with ``O_CREAT | O_EXCL``, which is atomic on POSIX and on
    Windows.  A stale lock (holder killed mid-write) would otherwise wedge every
    later caller, so it is broken after ``timeout`` seconds -- refusing to run
    forever is its own outage, and a lock file is only ever held for the
    duration of one small JSON write.

    Deliberately not ``fcntl``/``msvcrt``: those are platform-specific and this
    module must stay dependency-free and portable.
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    handle: int | None = None

    while True:
        try:
            handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                # Assume the holder died.  Breaking the lock beats deadlocking.
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
                deadline = time.monotonic() + timeout
                continue
            time.sleep(0.005)

    try:
        yield
    finally:
        if handle is not None:
            os.close(handle)
        try:
            os.unlink(lock_path)
        except OSError:
            pass


class FileStore:
    """JSON-file ledger, so idempotency survives a process restart.

    Deliberately not a database: the runtime guard must be droppable into any
    agent with zero infrastructure.

    Writes are atomic (temp file + ``os.replace``).  A plain ``write_text``
    truncates the file first, so a crash or SIGKILL mid-write leaves invalid
    JSON -- and an unreadable ledger must never be mistaken for an empty one.
    Atomic replace means readers see either the old file or the new one, never a
    half-written one; and if something *else* corrupts the file, we say so
    instead of silently forgetting every transaction we have ever seen.

    Every operation re-reads the file under an inter-process lock.  Atomic
    *writes* alone did not make the *read-modify-write* atomic: two processes
    both read an empty ledger, both concluded their key was free, and both
    charged.  The file is the source of truth, so a cached in-memory copy is
    only ever a starting point -- never something to decide on.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._keys: dict[str, int] = {}
        # Serialises read-modify-write *within this process*; `_exclusive`
        # extends that across processes.  Both are needed: the threading lock
        # keeps worker threads from interleaving, the file lock does the same
        # for separate processes.
        self._lock = threading.Lock()
        # Fail fast on an unreadable ledger rather than discovering it on the
        # first guarded call.
        self._reload()

    def seen(self, key: str) -> bool:
        with self._lock, _exclusive(self.path):
            self._reload()
            return key in self._keys

    def record(self, key: str) -> None:
        with self._lock, _exclusive(self.path):
            self._reload()
            self._keys[key] = self._keys.get(key, 0) + 1
            self._flush()

    def reserve(self, key: str) -> bool:
        """Claim *key* and persist the claim atomically.

        The whole read-modify-write happens under the inter-process lock, and
        the ledger is re-read from disk first: another process may have recorded
        this key since we last looked, and trusting our in-memory copy would
        hand the same key to both of us.
        """
        with self._lock, _exclusive(self.path):
            self._reload()
            if key in self._keys:
                return False
            self._keys[key] = 1
            self._flush()
            return True

    def release(self, key: str) -> None:
        with self._lock, _exclusive(self.path):
            self._reload()
            if self._keys.pop(key, None) is not None:
                self._flush()

    def count(self, key: str) -> int:
        with self._lock, _exclusive(self.path):
            self._reload()
            return self._keys.get(key, 0)

    def as_dict(self) -> dict[str, int]:
        with self._lock, _exclusive(self.path):
            self._reload()
            return dict(self._keys)

    def _reload(self) -> None:
        """Re-read the ledger from disk, in place.

        Must be called with both locks held.  The file is the source of truth:
        keeping a stale in-memory copy is exactly how two processes both come to
        believe a key is free.
        """
        if not self.path.exists():
            self._keys = {}
            return
        try:
            raw_text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LedgerError(f"幂等账本 {self.path} 存在但无法读取：{exc}") from exc
        if not raw_text.strip():
            self._keys = {}
            return
        try:
            raw: Any = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LedgerError(
                f"幂等账本 {self.path} 内容损坏（{exc}）。"
                "继续运行会让所有幂等检查失效，因此已中止。"
                f"确认无误后可删除该文件重新开始。"
            ) from exc
        if not isinstance(raw, dict):
            raise LedgerError(
                f"幂等账本 {self.path} 应为 JSON 对象，实际是 {type(raw).__name__}"
            )
        try:
            self._keys = {str(k): int(v) for k, v in raw.items()}
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"幂等账本 {self.path} 的计数值不是整数：{exc}") from exc

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._keys, ensure_ascii=False, indent=2)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            # Never leave the temp file behind on failure.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
