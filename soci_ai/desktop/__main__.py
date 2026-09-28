from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime


def initial_geometry(available):
    """Use Qt's logical work area, including taskbar and per-screen DPI."""
    x, y, available_width, available_height = available
    width = min(1360, max(1, available_width - 64))
    height = min(860, max(1, available_height - 64))
    return x + (available_width - width) // 2, y + (available_height - height) // 2, width, height


def self_test_models():
    """File-only portable runtime verification. Never opens webcam/microphone."""
    import wave
    import numpy as np
    from .resources import model_root
    from .asr import MODEL_NAME, StreamingChineseASR
    from .capture import _create_landmarker
    from .core import classify_text
    from .language import corpus_status
    started = time.perf_counter()
    mp, detector = _create_landmarker(model_root() / "face_landmarker.task")
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((128, 128, 3), dtype=np.uint8))
    result = detector.detect_for_video(image, 1)
    detector.close()
    recognizer = StreamingChineseASR(model_root() / MODEL_NAME).load()
    sample = model_root() / MODEL_NAME / "test_wavs/0.wav"
    with wave.open(str(sample), "rb") as wav:
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    updates = []
    for block in np.array_split(samples, max(1, int(np.ceil(samples.size / (rate / 10))))):
        updates.extend(recognizer.feed(block, rate))
    for _ in range(20):
        updates.extend(recognizer.feed(np.zeros(rate // 10, dtype=np.float32), rate))
    recognizer.close()
    finals = [update.text for update in updates if update.is_final]
    if not finals:
        raise RuntimeError("Bundled recording did not produce a final transcript")
    payload = {"ok": True, "file_only": True, "camera_opened": False, "microphone_opened": False,
                      "landmarker_loaded": True, "sample_transcript": finals[-1], "partials": len(updates),
                      "semantic_rule_loaded": classify_text("女生不适合学工科")[0] > .9,
                      "language_corpus": corpus_status(),
                      "seconds": round(time.perf_counter() - started, 2)}
    if not payload["semantic_rule_loaded"] or classify_text("女生就是学不好工科吧")[0] <= .5:
        raise RuntimeError("Bundled language corpus did not load or match the required phrase")
    from .resources import output_root
    root = output_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "model-self-test.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="SOCI-AI native desktop")
    parser.add_argument("--self-test-models", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="File-only offscreen GUI smoke; no devices")
    parser.add_argument('--self-test-history', action='store_true')
    parser.add_argument('--history-test-phase', choices=('write', 'read', 'delete', 'read-after-delete'), default='read')
    options = parser.parse_args(argv)
    if options.self_test_models or options.smoke_test or options.self_test_history:
        if sys.platform == "win32":
            import ctypes
            # Suppress dialogs in this verification process only, not Windows settings.
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    if options.self_test_models:
        return self_test_models()
    if options.self_test_history:
        from .history_diagnostics import self_test_history
        return self_test_history(options.history_test_phase)
    if options.smoke_test:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QLockFile, QTimer, QStandardPaths
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog, QProgressDialog
    from .ui import StudioWindow
    from .session import DesktopSession
    from .resources import output_root, export_root
    from .history_worker import HistoryWorker
    from .history_controller import HistoryController
    from .lifecycle import SessionHistoryLifecycle

    app = QApplication(sys.argv[:1])
    # Last window closing must not unload DLLs while native workers still run.
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("SOCI-AI-Desktop")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    lock = QLockFile(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.TempLocation) + "/soci-ai-desktop.lock")
    if not options.smoke_test and not lock.tryLock(0):
        QMessageBox.information(None, "SOCI-AI", "桌面窗口已经在运行。请使用已打开的窗口，避免重复占用摄像头。")
        return 0
    window = StudioWindow()
    screen = app.primaryScreen()
    if screen is not None and not options.smoke_test:
        area = screen.availableGeometry()
        geometry = initial_geometry((area.x(), area.y(), area.width(), area.height()))
        window.setMinimumSize(min(window.minimumWidth(), geometry[2]), min(window.minimumHeight(), geometry[3]))
        window.setGeometry(*geometry)
    session = DesktopSession()
    worker = HistoryWorker(output_root() / 'history.sqlite3')
    history = HistoryController(worker, auto_save=False)
    runtime = SessionHistoryLifecycle(session, history)
    if not options.smoke_test:
        worker.start()
        history._send('recover')
    window.set_deferred_close(True)
    preview_sequence = [-1]
    report_updated = [0.0]
    shutdown_started = [None]
    shutdown_dialog = [None]
    worker_stopping = [False]

    def paint_history(force=False):
        changed = history.handle_results()
        if changed or force:
            window.set_history_items(history.display_items(), history.selected_id, has_more=history.has_more)
            window.set_report(history.selected_report)
        error = history.save_error or history.load_error
        message = ('未能保存或读取 · ' + error) if error else history.save_status + ' · 本机历史' if history.items or history.active_id else '结束后点击“保存到历史”，下次打开仍可查看。'
        window.set_history_status(message, error=bool(error))
        window.set_history_save_available(history.needs_save_choice())

    def resolve_choice():
        if not history.needs_save_choice():
            return True
        box = QMessageBox(window)
        box.setWindowTitle('保存本次会话？')
        box.setText('保存后，下次打开可在“会话回顾”按日期查看完整曲线、原因和建议。\n不保存则仅保留本次窗口中的回顾。')
        save = box.addButton('保存到历史', QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton('不保存', QMessageBox.ButtonRole.DestructiveRole)
        box.addButton('返回', QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save)
        box.exec()
        if box.clickedButton() == save:
            history.save_current()
        elif box.clickedButton() == discard:
            history.discard_current()
        else:
            return False
        return True

    def start():
        if not resolve_choice():
            return
        if not runtime.start(window.selected_camera(), window.selected_microphone()):
            QMessageBox.information(window, "设备释放中", session.message or "会话已经运行。")
        else:
            history.select(None)
            preview_sequence[0] = -1
            paint_history(True)
            window.set_state({"status": "starting", "suggestion": session.message})

    def stop():
        window.set_state(runtime.stop())
        window.set_frame(None)
        paint_history(True)
        if history.needs_save_choice():
            history.select(None)
            window._show_page(1)
            paint_history(True)

    def refresh():
        try:
            state = session.poll()
            window.set_state(state)
            if session.running:
                history.update_current(session.latest_report)
            paint_history()
            if session.running and time.monotonic() - report_updated[0] > 1:
                report_updated[0] = time.monotonic()
                # Avoid rebuilding thousands of report rows during live capture.
                if window.pages.currentIndex() == 1 and history.selected_id is None:
                    window.set_report(history.selected_report)
        except Exception as exc:
            stop()
            QMessageBox.warning(window, "检测异常", str(exc))

    def preview():
        frame = session.frame()
        if frame and time.monotonic() - frame[1] < .7 and frame[0] != preview_sequence[0]:
            preview_sequence[0] = frame[0]
            vision = session.latest_state.get("vision", {})
            window.set_frame(frame[2], vision.get("box"))
        elif (frame is None or time.monotonic() - frame[1] >= .7) and preview_sequence[0] != -1:
            preview_sequence[0] = -1
            window.set_frame(None)

    def export():
        report = window.displayed_report()
        if not report:
            return
        root = export_root()
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(window, '导出目录不可用', str(exc) + '\n可以在保存对话框中选择其他位置。')
        default = str(root / ("会话报告-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".html"))
        path, selected_filter = QFileDialog.getSaveFileName(window, "导出会话报告", default,
            "可阅读报告 (*.html);;结构化数据 (*.json)")
        if path:
            from pathlib import Path
            try:
                from .report_export import export_payload
                path, content = export_payload(report, path, selected_filter)
                Path(path).write_text(content, encoding="utf-8")
            except OSError as exc:
                QMessageBox.warning(window, '导出失败', str(exc))

    def devices():
        try:
            import sounddevice as sd
            microphones = [("系统默认麦克风", None)] + [(f"{index} · {device['name']}", index) for index, device in enumerate(sd.query_devices()) if device["max_input_channels"] > 0]
        except Exception:
            microphones = [("系统默认麦克风（设备列表不可用）", None)]
        window.set_devices([(f"摄像头 {index}（尝试此索引）", index) for index in range(3)], microphones)

    def closing():
        refresh_timer.stop()
        preview_timer.stop()
        runtime.begin_close()
        window.set_state(session.latest_state)
        window.set_frame(None)
        if not resolve_choice():
            runtime.cancel_close()
            window.set_deferred_close(True)
            refresh_timer.start(100)
            preview_timer.start(33)
            paint_history(True)
            return
        paint_history(True)
        shutdown_started[0] = time.monotonic()
        shutdown_timer.start(100)

    def finish_closing():
        paint_history()
        if history.has_unsaved_sessions() and not history.pending_writes() and not runtime.discarding:
            if shutdown_dialog[0] is not None:
                shutdown_dialog[0].close()
                shutdown_dialog[0] = None
            box = QMessageBox(window)
            box.setWindowTitle('会话尚未保存')
            box.setText('最后的会话快照未保存：\n' + history.save_error + '\n\n已保存的进度仍保留。请选择重试、保留窗口导出，或放弃未保存内容后退出。')
            retry = box.addButton('重试保存', QMessageBox.ButtonRole.AcceptRole)
            keep = box.addButton('保留窗口', QMessageBox.ButtonRole.RejectRole)
            discard = box.addButton('放弃并退出', QMessageBox.ButtonRole.DestructiveRole)
            box.setDefaultButton(keep)
            shutdown_timer.stop()
            box.exec()
            if box.clickedButton() == retry:
                history.retry_failed()
                shutdown_timer.start(100)
            elif box.clickedButton() == discard:
                runtime.discard_unsaved()
                shutdown_timer.start(100)
            else:
                runtime.cancel_close()
                window.set_deferred_close(True)
                refresh_timer.start(100)
                preview_timer.start(33)
            return
        if runtime.can_shutdown() and not worker_stopping[0]:
            worker_stopping[0] = True
            worker.request_shutdown()
        if runtime.can_shutdown() and not worker.is_alive():
            shutdown_timer.stop()
            if shutdown_dialog[0] is not None:
                shutdown_dialog[0].close()
            window.allow_close()
            app.quit()
        elif time.monotonic() - shutdown_started[0] > 2 and shutdown_dialog[0] is None:
            dialog = QProgressDialog("正在保存会话并释放本应用的摄像头、麦克风与模型，请稍候。\n不会结束其他程序或修改系统设置。", "", 0, 0, window)
            dialog.setWindowTitle("SOCI-AI · 安全结束")
            dialog.setCancelButton(None)
            dialog.setMinimumDuration(0)
            dialog.show()
            shutdown_dialog[0] = dialog

    window.start_requested.connect(start)
    window.stop_requested.connect(stop)
    window.calibrate_requested.connect(session.calibrate)
    window.export_requested.connect(export)
    def save_history():
        if history.selected_id is None:
            history.save_current()
            paint_history(True)
    window.history_save_requested.connect(save_history)
    def select_history(key):
        history.select(key)
        paint_history(True)
    window.history_selected.connect(select_history)
    window.history_refresh_requested.connect(history.refresh_history)
    window.history_more_requested.connect(lambda: history.refresh_history(len(history.items)))
    def delete_history(key):
        if history.selected_id == key:
            history.delete_selected()
    window.history_delete_requested.connect(delete_history)
    window.history_retry_requested.connect(history.retry_failed)
    window.closing.connect(closing)
    shutdown_timer = QTimer()
    shutdown_timer.timeout.connect(finish_closing)
    refresh_timer = QTimer()
    refresh_timer.timeout.connect(refresh)
    refresh_timer.start(100)
    preview_timer = QTimer()
    preview_timer.timeout.connect(preview)
    preview_timer.start(33)
    window.show()
    if options.smoke_test:
        # Offscreen plugin does not enumerate Windows fonts; register for QA only.
        for font in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf"):
            if os.path.isfile(font):
                QFontDatabase.addApplicationFont(font)
        def smoke():
            root = output_root()
            root.mkdir(parents=True, exist_ok=True)
            path = root / "native-smoke.png"
            if not window.grab().save(str(path)):
                raise RuntimeError("Native GUI screenshot failed")
            print(json.dumps({"ok": True, "file_only": True, "screenshot": str(path), "camera_opened": False, "microphone_opened": False}))
            window.close()
        QTimer.singleShot(400, smoke)
    else:
        QTimer.singleShot(100, devices)
    result = app.exec()
    # finish_closing only quits after actual worker completion; no timeout can
    # silently tear down active native resources.
    return result


if __name__ == "__main__":
    raise SystemExit(main())
