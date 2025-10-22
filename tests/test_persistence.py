import json
import tempfile
from pathlib import Path

from agent_memory import TimeToLiveCollector
from agent_memory.memory_manager import MemoryManager
from agent_memory.persistence import CheckpointManager, WriteAheadLog


def read_json_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def test_write_ahead_log_and_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        wal_path = Path(tmpdir) / "wal.log"
        checkpoint_path = Path(tmpdir) / "checkpoint.pkl"
        wal = WriteAheadLog(str(wal_path))
        checkpoint = CheckpointManager(str(checkpoint_path))

        manager = MemoryManager(
            capacity=512,
            collectors=[TimeToLiveCollector()],
            wal=wal,
            checkpoint_manager=checkpoint,
        )

        obj = manager.store({"text": "persist me"})
        manager.free_object(obj.id)
        manager.checkpoint()

        entries = list(read_json_lines(wal_path))
        assert entries[0]["op"] == "store"
        assert entries[1]["op"] == "free"

        checkpoint_data = checkpoint.load()
        assert isinstance(checkpoint_data, dict)
        assert len(checkpoint_data) == 0
