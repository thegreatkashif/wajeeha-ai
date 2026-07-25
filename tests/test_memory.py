import tempfile
from pathlib import Path

from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory


def test_short_term_rolls_window():
    stm = ShortTermMemory(max_turns=3)
    for i in range(5):
        stm.add_user(f"msg {i}")
    turns = stm.as_list()
    assert len(turns) == 3
    assert turns[0].content == "msg 2"
    assert turns[-1].content == "msg 4"


def test_short_term_as_llm_messages_shape():
    stm = ShortTermMemory(max_turns=10)
    stm.add_user("hello")
    stm.add_assistant("hi there")
    messages = stm.as_llm_messages()
    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_long_term_upsert_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite3")
        ltm = LongTermMemory(db_path)
        ltm.upsert("device", "wifi_adapter", "TP-Link Archer T3U")
        fact = ltm.get("device", "wifi_adapter")
        assert fact is not None
        assert fact.value == "TP-Link Archer T3U"

        # upsert again should update, not duplicate
        ltm.upsert("device", "wifi_adapter", "TP-Link Archer T4U")
        fact = ltm.get("device", "wifi_adapter")
        assert fact.value == "TP-Link Archer T4U"
        assert len(ltm.list_by_category("device")) == 1
        ltm.close()


def test_long_term_delete():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite3")
        ltm = LongTermMemory(db_path)
        ltm.upsert("preference", "theme", "dark")
        assert ltm.delete("preference", "theme") is True
        assert ltm.get("preference", "theme") is None
        ltm.close()