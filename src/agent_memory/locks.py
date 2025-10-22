from __future__ import annotations

import threading
from contextlib import contextmanager


class ReadWriteLock:
    def __init__(self) -> None:
        self._readers = 0
        self._readers_lock = threading.Lock()
        self._resource_lock = threading.Lock()

    @contextmanager
    def read_lock(self):
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(self):
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

    def acquire_read(self) -> None:
        with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                self._resource_lock.acquire()

    def release_read(self) -> None:
        with self._readers_lock:
            self._readers -= 1
            if self._readers == 0:
                self._resource_lock.release()

    def acquire_write(self) -> None:
        self._resource_lock.acquire()

    def release_write(self) -> None:
        self._resource_lock.release()
