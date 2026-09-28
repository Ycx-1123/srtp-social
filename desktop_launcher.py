"""PyInstaller entry: native window, not a web server."""
import faulthandler
import io
import sys

from soci_ai.desktop.resources import log_root


if __name__ == "__main__":
    try:
        root = log_root()
        root.mkdir(parents=True, exist_ok=True)
        log = (root / "desktop.log").open("a", encoding="utf-8", buffering=1)
        faulthandler.enable(log)
    except OSError:
        # Unwritable user data must not prevent the window and save error UI.
        log = io.StringIO()
    with log:
        # Windowed EXE has no console; retain diagnostic evidence locally.
        if sys.stdout is None:
            sys.stdout = log
        if sys.stderr is None:
            sys.stderr = log
        try:
            from soci_ai.desktop.__main__ import main
            raise SystemExit(main())
        except Exception:
            import traceback
            traceback.print_exc(file=log)
            raise
