import unittest

from agent_memory.memory_object import MemoryObject
from agent_memory.pager import PageTable


class PagerTests(unittest.TestCase):
    def test_lfu_eviction_prefers_low_frequency(self) -> None:
        pager = PageTable(page_size=64, strategy="lfu")
        max_pages = 2
        obj1 = MemoryObject(payload={"text": "alpha"}, size=5)
        obj2 = MemoryObject(payload={"text": "beta"}, size=5)
        obj3 = MemoryObject(payload={"text": "gamma"}, size=5)

        pager.ensure_resident(obj1, max_pages)
        pager.ensure_resident(obj2, max_pages)
        pager.read(obj1.id)
        pager.read(obj1.id)

        pager.ensure_resident(obj3, max_pages)

        self.assertEqual(pager.stats.page_faults, 3)
        self.assertEqual(pager.stats.evictions, 1)
        remaining_ids = {page_id[0] for page_id in pager._table.keys()}
        self.assertIn(obj1.id, remaining_ids)
        self.assertIn(obj3.id, remaining_ids)


if __name__ == "__main__":
    unittest.main()
