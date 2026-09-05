"""Cross-platform process locks, independent of the metadata byte range.

Windows byte locks are mandatory. Lock a byte beyond EOF so diagnostic readers
can still read the JSON metadata at offset zero while a worker owns the lease.
The operating system releases the lock if a process exits or crashes.
"""

from __future__ import annotations

import getpass
import hashlib
import os

if os.name != "nt":
    from fcntl import LOCK_EX, LOCK_NB, LOCK_SH, LOCK_UN, flock
else:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel.LockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _kernel.LockFileEx.restype = wintypes.BOOL
    _kernel.UnlockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _kernel.UnlockFileEx.restype = wintypes.BOOL

    def flock(fd: int, operation: int) -> None:
        handle = msvcrt.get_osfhandle(fd)
        offset = _Overlapped(Offset=0xFFFFFFFF, OffsetHigh=0x7FFFFFFF)
        if operation & LOCK_UN:
            ok = _kernel.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(offset))
        else:
            flags = (2 if operation & LOCK_EX else 0) | (
                1 if operation & LOCK_NB else 0
            )
            ok = _kernel.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(offset))
        if not ok:
            code = ctypes.get_last_error()
            if code in (32, 33, 158):
                raise BlockingIOError(code, "file lease is held by another process")
            raise ctypes.WinError(code)


def user_key() -> str:
    if hasattr(os, "getuid"):
        return str(os.getuid())
    return hashlib.sha256(getpass.getuser().encode()).hexdigest()[:12]
