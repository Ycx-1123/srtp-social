"""Explicit isolated, file-only cross-process diagnostics; never real sessions."""
import json
import os
from pathlib import Path

from .history_store import HistoryStore


def self_test_history(phase):
    directory = os.environ.get('SOCI_AI_TEST_DATA_DIR', '')
    if not directory or not Path(directory).is_absolute():
        raise ValueError('历史自检必须设置绝对路径 SOCI_AI_TEST_DATA_DIR。')
    root = Path(directory)
    store = HistoryStore(root / 'history.sqlite3')
    try:
        if phase == 'write':
            for key, score, cue in (('a', 70, '疑似学历偏见'), ('b', 40, '眉部紧张增强')):
                report = dict(elapsed_seconds=30, average_sbi=score, peak_sbi=score,
                    history=[dict(at_ms=1000, sbi=score, cues=[cue])],
                    transcripts=[dict(at_ms=1000, text='仅用于文件自检的测试字幕')],
                    provenance='isolated_test_fixture')
                store.create_session(key, '2026-09-27T09:00:00+08:00', report, 'diagnostic')
                assert store.finish(key, report, '2026-09-27T09:00:30+08:00')
        elif phase == 'delete':
            assert store.delete_session('a')
        if phase in ('write', 'read'):
            assert store.load_session('a')['report']['history'][0]['cues'] == ['疑似学历偏见']
            assert len(store.list_sessions()) == 2
        else:
            assert store.load_session('a') is None
            assert len(store.list_sessions()) == 1
        b = store.load_session('b')['report']
        assert b['average_sbi'] == 40 and b['history'][0]['cues'] == ['眉部紧张增强']
        payload = dict(ok=True, phase=phase, file_only=True, camera_opened=False,
                       microphone_opened=False, remaining_ids=[item['id'] for item in store.list_sessions()])
        (root / f'history-self-test-{phase}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    finally:
        store.close()
