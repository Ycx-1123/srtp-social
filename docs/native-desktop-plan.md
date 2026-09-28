# Native desktop implementation

The desktop request replaces the interrupted scripted-demo request. No fabricated transcription, random score changes, or invented validation metrics are introduced.

1. Preserve the web prototype; fix its confirmed background-inference blocking regression.
2. Create a standalone PySide6/QtWidgets window with native camera, microphone, living-tree feedback, and measured-session review.
3. Separate capture, latest-frame landmarker, bounded audio capture, streaming CTC recognition, and UI refresh. Slow processing must discard old work rather than accumulate minutes of latency.
4. Use genuine landmarks and personally calibrated facial action changes. Distinguish observable tension from proven offence or psychological state. Combine recent visual, acoustic, and rule-based semantic evidence with fast attack and slower release. Missing evidence yields no score.
5. Test scheduling, overflow, freshness, score response, endpointing, bounding-box mapping, and start/stop lifecycle before integration. Benchmark local models using file fixtures, without opening devices.
6. Package a portable one-folder EXE including small local models. Verify the executable and rendered native window. Physical webcam/microphone accuracy and end-to-end latency still require user testing.

Acceptance: no browser/server dependency; latest camera frame is independent of inference; face box derives from current landmarks; ASR emits real partial hypotheses; bounded audio backlog is observable; stop removes live scores and releases devices; reports describe actual recorded session data and clearly identify unvalidated prototype rules.
