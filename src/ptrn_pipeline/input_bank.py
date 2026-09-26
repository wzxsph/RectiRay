"""Bounded validation input bank with optional non-blocking preparation."""
import threading

from .cache import DiskCache, cached_scene, digest, scene_key


class InputBank:
    """Keep deterministic validation inputs in RAM, within ``budget`` bytes.

    The eager default preserves standalone evaluation. ``eager=False`` starts
    an ordered producer immediately so training can continue while inputs are
    prepared. Validation uses ``ready_batch`` rather than blocking on an item.
    """

    def __init__(self, data, root, count=None, budget=8 * 2**30, *, eager=True, prefetch=512):
        if count is not None and count < 1:
            raise ValueError('Positive sample count required')
        if budget < 1 or prefetch < 1:
            raise ValueError('Positive input bank budget and prefetch required')
        self.count = min(count, len(data)) if count is not None else len(data)
        self.data = data
        self.cache = DiskCache(root, 'dev-input-' + digest(data.fingerprint)[:16], budget=2 * 2**30)
        self.full = DiskCache(root, 'dev-full-' + digest(data.fingerprint)[:16], budget=10 * 2**30)
        self.bytes = 0
        self.budget = budget
        self.rebuilt = 0
        self.prefetch = prefetch
        self.items = [] if eager else [None] * self.count
        self._condition = threading.Condition()
        self._prepared = 0
        self._consumed = 0
        self._error = None
        self._closed = False
        self._producer = None
        if eager:
            for index in range(self.count):
                self._append(index, self._load(index))
        else:
            self._producer = threading.Thread(target=self._produce, name='ptrn-dev-inputs', daemon=True)
            self._producer.start()

    def _load(self, index):
        cached = self.cache.get(scene_key(self.data, index, full=False))
        if cached is None:
            _, arrays, _, _ = cached_scene(self.data, index, self.full)
            inputs = {key[6:]: value for key, value in arrays.items() if key.startswith('input_')}
            self.cache.put(scene_key(self.data, index, full=False), inputs, {'index': index})
            self.rebuilt += 1
        else:
            inputs, metadata = cached
            if metadata.get('index', index) != index:
                raise ValueError('Validation input cache index mismatch')
        return {key: value for key, value in inputs.items() if key not in ('mask', 'photo_valid')}

    def _append(self, index, item):
        size = sum(value.nbytes for value in item.values())
        with self._condition:
            if self.bytes + size > self.budget:
                raise MemoryError('Validation RAM bank exceeds its budget')
            if self._producer is None:
                self.items.append(item)
            else:
                self.items[index] = item
            self.bytes += size
            self._prepared = index + 1
            self._condition.notify_all()

    @property
    def prepared(self):
        with self._condition:
            return self._prepared

    @property
    def error(self):
        with self._condition:
            return self._error

    def _produce(self):
        try:
            for index in range(self.count):
                with self._condition:
                    while not self._closed and self._prepared - self._consumed >= self.prefetch:
                        self._condition.wait()
                    if self._closed:
                        return
                self._append(index, self._load(index))
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def ready_batch(self, start, size):
        """Return ordered inputs, or ``None`` without waiting for the producer."""
        if start < 0 or size < 1 or start + size > self.count:
            raise IndexError('Validation batch outside input bank')
        with self._condition:
            if self._error is not None:
                raise RuntimeError('Validation input preparation failed') from self._error
            if self._prepared < start + size:
                return None
            return self.items[start:start + size]

    def acknowledge(self, end):
        """Permit the producer to move its bounded lookahead past ``end``."""
        if not 0 <= end <= self.count:
            raise IndexError(end)
        with self._condition:
            self._consumed = max(self._consumed, end)
            self._condition.notify_all()

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        with self._condition:
            while self._prepared <= index and self._error is None and not self._closed:
                self._condition.wait()
            if self._error is not None:
                raise RuntimeError('Validation input preparation failed') from self._error
            if self._prepared <= index:
                raise RuntimeError('Validation input bank closed before item was ready')
            return self.items[index]

    def __len__(self):
        return self.count

    def close(self, timeout=5):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._producer is not None:
            self._producer.join(timeout=timeout)
