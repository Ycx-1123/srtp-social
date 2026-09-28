"""Exercise ONLY the copied EXE in fresh directories, no hardware/user DB."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='单文件隔离验收-', dir=args.output.resolve()))
    first, second, cwd, data = [root / item for item in ('仅有程序','移动后的程序','独立工作目录','隔离用户数据')]
    for folder in (first, second, cwd, data):
        folder.mkdir()
    exe = first / 'SOCI-AI.exe'
    shutil.copy2(args.exe.resolve(), exe)
    windows = Path(os.environ.get('SystemRoot', 'C:/Windows'))
    env = {**os.environ, 'SOCI_AI_TEST_DATA_DIR': str(data),
           'PATH': os.pathsep.join((str(windows / 'System32'), str(windows)))}
    for key in ('PYTHONPATH','PYTHONHOME','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH'):
        env.pop(key, None)
    runs = []
    def run(arguments, artifact):
        target = data / artifact
        if target.exists():
            raise RuntimeError('验收不允许使用旧诊断结果。')
        started = time.perf_counter()
        result = subprocess.run([str(exe), *arguments], cwd=cwd, env=env, capture_output=True,
            text=True, encoding='utf-8', errors='replace', timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        elapsed = round(time.perf_counter()-started, 3)
        if result.returncode != 0 or not target.is_file():
            raise RuntimeError(f'{arguments}: exit={result.returncode}; {result.stderr}; 日志：{data / "logs/desktop.log"}')
        if target.suffix == '.json':
            payload = json.loads(target.read_text(encoding='utf-8'))
            assert payload['ok'] and not payload['camera_opened'] and not payload['microphone_opened']
        else:
            assert target.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
        runs.append(dict(arguments=arguments, seconds=elapsed, artifact=str(target)))
    run(['--self-test-models'], 'model-self-test.json')
    run(['--smoke-test'], 'native-smoke.png')
    for phase in ('write','read','delete','read-after-delete'):
        run(['--self-test-history','--history-test-phase',phase], f'history-self-test-{phase}.json')
    shutil.copy2(exe, second / 'SOCI-AI.exe')
    exe = second / 'SOCI-AI.exe'
    # Distinct proof artifact; leave the first read-after-delete evidence intact.
    shutil.move(str(data / 'history-self-test-read-after-delete.json'), str(data / 'before-move-read-after-delete.json'))
    run(['--self-test-history','--history-test-phase','read-after-delete'], 'history-self-test-read-after-delete.json')
    assert [f.name for f in first.iterdir()] == ['SOCI-AI.exe']
    assert [f.name for f in second.iterdir()] == ['SOCI-AI.exe']
    assert not list(cwd.iterdir())
    output = dict(ok=True, file_only=True, camera_opened=False, microphone_opened=False,
                  executable=str(args.exe.resolve()), bytes=exe.stat().st_size,
                  runs=runs, test_root=str(root))
    # Explicit close of file used for hashing avoids a Windows deletion/rename lock.
    with exe.open('rb') as handle:
        output['sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    report = root / 'acceptance.json'
    report.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
