"""Own the resident CUDA process and recycle cache directories."""

import ctypes
from ctypes import wintypes


class BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
        ("flags", wintypes.DWORD), ("minimum_working_set", ctypes.c_size_t),
        ("maximum_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
        ("scheduling", wintypes.DWORD),
    ]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", BasicLimits), ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
    ]


kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
kernel.CreateJobObjectW.restype = wintypes.HANDLE
kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
kernel.SetInformationJobObject.restype = wintypes.BOOL
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
kernel.AssignProcessToJobObject.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL


def attach(pid):
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        handle = kernel.OpenProcess(0x0101, False, pid)  # SET_QUOTA | TERMINATE.
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not kernel.AssignProcessToJobObject(job, handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)
        return job
    except Exception:
        kernel.CloseHandle(job)
        raise


def recycle(path):
    # Explorer's undoable deletion; never permanently remove a cache directory.
    class Operation(ctypes.Structure):
        _fields_ = [("window", wintypes.HWND), ("operation", wintypes.UINT), ("source", wintypes.LPCWSTR), ("destination", wintypes.LPCWSTR), ("flags", ctypes.c_ushort), ("aborted", wintypes.BOOL), ("mapping", ctypes.c_void_p), ("title", wintypes.LPCWSTR)]
    source = ctypes.create_unicode_buffer(path + "\0\0")
    operation = Operation()
    operation.operation = 3
    operation.source = ctypes.cast(source, wintypes.LPCWSTR)
    operation.flags = 0x0040 | 0x0010 | 0x0004 | 0x0400
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.SHFileOperationW.argtypes = [ctypes.POINTER(Operation)]
    shell.SHFileOperationW.restype = ctypes.c_int
    result = shell.SHFileOperationW(ctypes.byref(operation))
    if result or operation.aborted:
        raise OSError(result, "Could not move cache to the Recycle Bin", path)
