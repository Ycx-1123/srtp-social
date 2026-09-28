"""Build the portable native Qt desktop app with the current Python runtime."""
from __future__ import annotations

import argparse
import importlib.util
from importlib.metadata import distribution, PackageNotFoundError
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import tempfile


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"
NAME = "SOCI-AI-Desktop"
EXCLUDED_MODULES = (
    "torch", "tensorflow", "funasr", "ultralytics", "pandas", "IPython",
    "scipy", "PyQt5", "PyQt6", "PySide2", "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "soci_ai.api", "soci_ai.runtime", "soci_ai.live.runtime", "soci_ai.live.vision",
    "soci_ai.live.audio", "fastapi", "uvicorn", "websockets",
)


def resource_files(root: Path) -> tuple[Path, ...]:
    model = root / "models" / MODEL_NAME
    return (
        root / "models" / "face_landmarker.task",
        model / "model.int8.onnx", model / "tokens.txt", model / "bbpe.model",
        model / "test_wavs" / "0.wav",
        root / "soci_ai" / "live" / "bias_patterns.json",
        root / "resources" / "language" / "social_bias_examples.json",
        root / "resources" / "language" / "social_bias_examples.sqlite",
        root / "resources" / "language" / "README.md",
        root / 'resources' / 'licenses' / 'THIRD-PARTY-NOTICES.txt',
        root / 'docs' / 'desktop-quickstart.md',
        root / 'docs' / 'desktop-packaging.md',
    )


def build_command(root: Path = ROOT, *, clean: bool = False, mode: str = 'onedir') -> list[str]:
    if mode not in ('onedir', 'onefile'):
        raise ValueError('mode must be onedir or onefile')
    root = root.resolve()
    name = 'SOCI-AI' if mode == 'onefile' else NAME
    dist = root / 'dist' / 'SOCI-AI-Onefile' if mode == 'onefile' else root / 'dist'
    cache = root / 'build' / ('native-onefile' if mode == 'onefile' else 'native')
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", '--' + mode, "--windowed",
        "--noupx", "--name", name,
        "--distpath", str(dist),
        "--workpath", str(cache),
        "--specpath", str(cache),
        "--paths", str(root),
        "--hidden-import", "sherpa_onnx",
        "--hidden-import", "sherpa_onnx.lib._sherpa_onnx",
        "--collect-binaries", "sherpa_onnx",
        "--hidden-import", "mediapipe",
        "--hidden-import", "mediapipe.tasks.c",
        "--collect-binaries", "mediapipe",
        "--hidden-import", "sounddevice",
        "--collect-binaries", "_sounddevice_data",
    ]
    if clean:
        command.append("--clean")
    for package in EXCLUDED_MODULES:
        command.extend(("--exclude-module", package))
    for source in resource_files(root):
        relative = source.relative_to(root)
        command.extend(("--add-data", f"{source}:{relative.parent.as_posix()}"))
    command.append(str(root / "desktop_launcher.py"))
    return command


def write_notices(target: Path):
    """Assemble installed dependency notices, offline, not any user files."""
    parts = ['SOCI-AI 第三方软件与模型声明\n\n'
             '本程序使用 PySide6/Qt (LGPLv3)、MediaPipe 与 sherpa-onnx；第三方模型并非本项目训练。\n'
             '允许按各许可证对相关库进行研究、修改、替换和重新链接；不施加禁止此类调试的条款。\n'
             'Qt/PySide 对应源码及构建信息：https://download.qt.io/official_releases/qt/6.11/6.11.2/\n'
             'https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.2-src/\n'
             '许可证义务：https://www.qt.io/development/open-source-lgpl-obligations\n'
             'Face Landmarker：https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker\n'
             '面部模型说明：https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf\n'
             '中文语音模型：sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01\n'
             '模型来源：https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models\n'
             '代码来源：https://github.com/k2-fsa/sherpa-onnx/tree/v1.13.8\n'
             '运行时第三方声明不等于模型权重再分发授权核验；公开或商业发布前需单独核对。\n'
             '软件不是加密封装。可用 PyInstaller 官方 archive_viewer 提取字节码与资源，'
             '修改动态库后用同版本 PyInstaller 重新封装；开发方应提供对应库源码与重链接材料。\n']
    names = ('pyside6-essentials','shiboken6','sherpa-onnx','sherpa-onnx-core','mediapipe',
             'opencv-contrib-python','numpy','sounddevice','matplotlib','pydantic','pydantic-core',
             'cffi','pycparser','absl-py','flatbuffers','protobuf','pillow','contourpy','cycler',
             'fonttools','kiwisolver','packaging','pyparsing','python-dateutil','six',
             'annotated-types','typing-extensions','typing-inspection','PyInstaller','altgraph','pywin32-ctypes')
    for name in names:
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            continue
        parts.append(f'\n===== {name} {dist.version} =====\n' +
                     str(dist.metadata.get('License-Expression') or dist.metadata.get('License') or '详见上游声明') + '\n')
        for file in dist.files or ():
            if any(word in Path(str(file)).name.lower() for word in ('license','copying','notice')):
                path = Path(dist.locate_file(file))
                if path.is_file() and path.suffix.lower() not in ('.dll','.pyd','.py','.pyc'):
                    parts.append('\n--- ' + str(file) + ' ---\n' + path.read_text(encoding='utf-8', errors='replace'))
    for name in ('LGPL-3.0.txt','GPL-3.0.txt','sherpa-onnx-LICENSE.txt'):
        parts.append('\n===== ' + name + ' =====\n' + (ROOT / 'resources/licenses' / name).read_text(encoding='utf-8'))
    python_license = Path(sys.executable).parent / 'LICENSE.txt'
    if python_license.is_file():
        parts.append('\n===== Python =====\n' + python_license.read_text(encoding='utf-8'))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(parts), encoding='utf-8')


def validate_inputs(root: Path = ROOT) -> list[str]:
    errors = []
    for path in (root / "desktop_launcher.py", *resource_files(root)):
        if not path.is_file():
            errors.append(f"Missing build input: {path}")
    for package in ("PyInstaller", "PySide6", "sherpa_onnx", "mediapipe", "matplotlib", "sounddevice", "cv2", "pydantic"):
        if importlib.util.find_spec(package) is None:
            errors.append(f"Missing package: {package}; install requirements-desktop.txt")
    return errors


def build_environment() -> dict[str, str]:
    """Keep unrelated tool DLLs out of native dependency resolution.

    For example, a Poppler ICU DLL on the host PATH has a different API from
    Windows ICU used by Qt 6.11. Package hooks already add their own DLL paths.
    """
    environment = dict(os.environ)
    python_root = Path(sys.executable).resolve().parent
    windows_root = Path(environment.get("SystemRoot", r"C:\Windows"))
    directories = (
        python_root, python_root / "DLLs", python_root / "Scripts",
        python_root / "Library" / "bin", windows_root / "System32", windows_root,
    )
    environment["PATH"] = os.pathsep.join(str(path) for path in directories if path.is_dir())
    return environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without building or opening devices")
    parser.add_argument("--print-command", action="store_true", help="Print the argument list without running PyInstaller")
    parser.add_argument("--clean", action="store_true", help="Refresh dependency analysis after Python/runtime changes")
    parser.add_argument("--skip-smoke", action="store_true", help="Build without running the no-device model and GUI self-tests")
    parser.add_argument('--mode', choices=('onedir','onefile'), default='onedir')
    args = parser.parse_args(argv)
    write_notices(ROOT / 'resources/licenses/THIRD-PARTY-NOTICES.txt')
    errors = validate_inputs()
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 2
    command = build_command(clean=args.clean, mode=args.mode)
    if args.check_only or args.print_command:
        print(json.dumps({"ok": True, "command": command, "output": str(ROOT / 'dist' / ('SOCI-AI-Onefile' if args.mode == 'onefile' else NAME))}, ensure_ascii=False, indent=2))
        return 0
    # PyInstaller replaces only its named output bundle. No broad cleanup or
    # deletion of the project, dist directory, or build directory is performed.
    (ROOT / 'build' / ('native-onefile' if args.mode == 'onefile' else 'native')).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=ROOT, env=build_environment())
    if result.returncode:
        return result.returncode
    bundle = ROOT / 'dist' / ('SOCI-AI-Onefile' if args.mode == 'onefile' else NAME)
    executable = bundle / ('SOCI-AI.exe' if args.mode == 'onefile' else f'{NAME}.exe')
    if not args.skip_smoke:
        qa = ROOT / 'artifacts/onefile-qa'
        qa.mkdir(parents=True, exist_ok=True)
        test_root = Path(tempfile.mkdtemp(prefix='build-smoke-', dir=qa))
        test_env = {**build_environment(), 'SOCI_AI_TEST_DATA_DIR': str(test_root)}
        tested_at = time.time()
        smoke = subprocess.run([str(executable), "--self-test-models"], cwd=bundle, env=test_env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        if smoke.stderr:
            print(smoke.stderr, file=sys.stderr)
        if smoke.returncode:
            print(f"Self-test failed; inspect {test_root / 'logs/desktop.log'}", file=sys.stderr)
            return smoke.returncode
        report_file = test_root / 'model-self-test.json'
        if not report_file.is_file() or report_file.stat().st_mtime < tested_at - 1:
            print(f"Self-test did not write a fresh report: {report_file}", file=sys.stderr)
            return 3
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("Executable self-test did not return valid JSON", file=sys.stderr)
            return 3
        if report.get("ok") is not True:
            print(f"Executable model self-test reported failure: {report}", file=sys.stderr)
            return 3
        print(json.dumps(report, ensure_ascii=False))
        gui_started = time.time()
        gui_smoke = subprocess.run([str(executable), "--smoke-test"], cwd=bundle, env=test_env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        screenshot = test_root / 'native-smoke.png'
        if gui_smoke.returncode or not screenshot.is_file() or screenshot.stat().st_mtime < gui_started - 1:
            print(f"Frozen GUI self-test failed; inspect {test_root / 'logs/desktop.log'}", file=sys.stderr)
            return gui_smoke.returncode or 4
        if screenshot.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            print(f"GUI self-test produced an invalid PNG: {screenshot}", file=sys.stderr)
            return 4
        print(json.dumps({"ok": True, "offscreen_gui_smoke": str(screenshot), "camera_opened": False, "microphone_opened": False}))
    if args.mode == 'onefile':
        print(json.dumps({'ok':True, 'executable':str(executable), 'bytes':executable.stat().st_size}, ensure_ascii=False))
        return 0
    shutil.copy2(ROOT / "docs" / "desktop-packaging.md", bundle / "README-packaging.md")
    quickstart = ROOT / "docs" / "desktop-quickstart.md"
    if quickstart.is_file():
        shutil.copy2(quickstart, bundle / "使用说明.md")
    shutil.copy2(ROOT / "docs" / "desktop-system-math.md", bundle / "指标与实现说明.md")
    # User-facing database copy; the runtime's identical snapshot is bundled
    # under _internal/resources/language for relocatable resource loading.
    corpus_dir = bundle / "语料库"
    corpus_dir.mkdir(exist_ok=True)
    for name in ("social_bias_examples.json", "social_bias_examples.sqlite", "README.md"):
        shutil.copy2(ROOT / "resources" / "language" / name, corpus_dir / name)
    shutil.copy2(ROOT / 'docs' / 'desktop-language-labels.md', corpus_dir / '标签与提示说明.md')
    size = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
    print(json.dumps({"ok": True, "executable": str(executable), "bundle_bytes": size, "bundle_mib": round(size / 1024 ** 2, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
