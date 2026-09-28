"""Session/history lifecycle with injectable devices, no GUI dependencies."""
from datetime import datetime

APP_VERSION = '2026.09.27-layered-guidance'


class SessionHistoryLifecycle:
    def __init__(self, session, history):
        self.session, self.history = session, history
        self.closing = self.discarding = False

    def _report(self):
        return self.session.latest_report or dict(elapsed_seconds=round(self.session.elapsed, 1),
            average_sbi=None, peak_sbi=None, history=[], transcripts=[], events=[],
            provenance='measured_inputs_heuristic_assessment')

    def start(self, camera, microphone):
        if self.closing or self.history.needs_save_choice() or not self.session.start(camera, microphone):
            return False
        self.history.begin(self._report(), datetime.now().astimezone().isoformat(), APP_VERSION)
        return True

    def poll(self):
        state = self.session.poll()
        if self.session.running:
            self.history.update_current(self._report())
        self.history.handle_results()
        return state

    def stop(self):
        if not self.session.running:
            return getattr(self.session, 'latest_state', {'status': 'stopped'})
        state = self.session.stop()
        self.history.finish_current(self._report(), datetime.now().astimezone().isoformat())
        return state

    def begin_close(self):
        self.closing = True
        self.stop()

    def cancel_close(self):
        self.closing = self.discarding = False

    def discard_unsaved(self):
        self.discarding = True

    def can_shutdown(self):
        return (self.session.shutdown_finished() and not self.history.pending_writes()
                and not self.history.needs_save_choice()
                and (self.discarding or not self.history.has_unsaved_sessions()))
