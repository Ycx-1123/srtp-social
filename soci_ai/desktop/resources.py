from pathlib import Path
import os
import sys


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))


def model_root() -> Path:
    return resource_root() / "models"


def output_root() -> Path:
    diagnostic = any(flag in sys.argv for flag in ('--self-test-models', '--smoke-test', '--self-test-history'))
    if diagnostic:
        override = os.environ.get('SOCI_AI_TEST_DATA_DIR', '')
        if not override or not Path(override).is_absolute():
            raise ValueError('自检必须指定绝对路径 SOCI_AI_TEST_DATA_DIR，不能使用真实用户历史。')
        return Path(override)
    local = os.environ.get('LOCALAPPDATA')
    return (Path(local) if local else Path.home() / 'AppData' / 'Local') / 'SOCI-AI-Desktop'


def log_root() -> Path:
    return output_root() / 'logs'


def export_root() -> Path:
    return output_root() / 'exports'
