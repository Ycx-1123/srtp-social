# SOCI-AI 实时自检与生命树 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个默认使用电脑真实摄像头和麦克风的单人微冒犯自检系统，以真实人脸框、中文语音转写、可解释多模态指标和动态生命树提供实时反馈。

**Architecture:** 浏览器负责媒体授权、自拍预览、音频特征提取和 Canvas 动画；本地 FastAPI 接收低频视频帧与短 PCM 音频，懒加载 YOLOv8 和 SenseVoice，并将视觉、声学、语义事件送入独立的实时运行时。现有 ShowcaseRuntime 与模拟场景保留为“备用回放”，实时主链路使用独立 API、独立会话状态和 measured provenance。

**Tech Stack:** Python 3.10+、FastAPI、Pydantic 2、OpenCV、Ultralytics YOLOv8、FunASR/SenseVoice、NumPy、SQLite、原生 JavaScript、MediaDevices、Web Audio API、Canvas 2D、unittest/pytest

**Spec:** `docs/superpowers/specs/2026-09-25-soci-ai-live-tree-design.md`

## Global Constraints

- 默认入口必须是真实单人自检；模拟场景只能以“备用回放”出现。
- 摄像头预览必须显示真实自拍和实际检测到的人脸边界框。
- “正在监听/正在转写/正在分析”只能反映真实处理状态；SenseVoice 不可用时显示“转写不可用”。
- 基础运行完全本地化，不调用云端 API。
- 原始视频帧和音频窗口不得写入会话数据库；只保存结构化特征、转录和融合事件。
- 第一版默认近距离主讲者就是摄像头前使用者；声纹注册与多人说话人分离仅预留接口。
- 默认视频分析 3 FPS、最长边约 480 像素；过期帧丢弃，不排队积压。
- 树动画与模型推理解耦，目标 50–60 FPS；低性能设备自动降低粒子量。
- 无人脸、静音、权限拒绝或模型缺失时保持中性并明确降级，不注入模拟证据。
- 不把结果描述为人格、道德、心理或医学诊断。

## Review Focus

- 浏览器重复点击开始或结束：`tests/test_live_api.py::test_start_and_stop_are_idempotent` 必须证明不会创建重复模型任务或留下活跃会话。
- 超大、损坏或非图像帧：`tests/test_live_vision.py::test_rejects_invalid_and_oversized_frames` 必须返回结构化错误且不触发模型。
- 音频块乱序、过短或静音：`tests/test_live_audio.py::test_silent_short_and_stale_audio_are_not_transcribed` 必须保持中性且不伪造转录。
- 摄像头镜像及画布缩放：`tests/test_frontend_live_contract.py::test_face_overlay_uses_source_dimensions_and_mirror_flag` 必须固定边界框坐标契约。
- 模型推理慢于采样：`tests/test_live_runtime.py::test_busy_runtime_drops_stale_frames` 必须证明只保留最新帧并维持可查询状态。

---

### Task 1: 实时领域模型、配置与能力检测

**Files:**
- Create: `soci_ai/live/__init__.py`
- Create: `soci_ai/live/domain.py`
- Create: `soci_ai/live/capabilities.py`
- Modify: `soci_ai/config.py`
- Test: `tests/test_live_capabilities.py`

**Interfaces:**
- Consumes: `Settings.root: Path` 与现有 `discover_optional_adapters()`。
- Produces: `LiveCapabilities`, `LiveState`, `FaceBox`, `VisionObservation`, `AudioObservation`, `TranscriptObservation`, `TreeState`；`Settings.yolo_weights`, `Settings.sensevoice_model_dir`, `Settings.live_video_fps`；`detect_live_capabilities(settings) -> LiveCapabilities`；`print_capabilities(settings) -> None`。

- [ ] **Step 1: 写失败测试，固定本地模型路径、缺失状态和 measured 数据契约**

```python
def test_capabilities_resolve_existing_models_without_importing_them(tmp_path):
    yolo = tmp_path / "best.pt"
    sense = tmp_path / "SenseVoiceSmall"
    yolo.write_bytes(b"weights")
    sense.mkdir()
    settings = Settings(
        root=tmp_path,
        yolo_weights=yolo,
        sensevoice_model_dir=sense,
    )
    result = detect_live_capabilities(settings)
    assert result.camera_capture == "browser"
    assert result.yolo.status == "ready"
    assert result.sensevoice.status in {"ready", "package_missing"}
    assert "ultralytics" not in sys.modules

def test_live_state_starts_neutral_and_measured():
    state = LiveState.neutral("session-1")
    assert state.provenance == "measured"
    assert state.evidence_status == "insufficient"
    assert state.tree.mode == "observing"
```

- [ ] **Step 2: 运行测试并确认因实时模块不存在而失败**

Run: `python -m pytest tests/test_live_capabilities.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'soci_ai.live'`。

- [ ] **Step 3: 实现不可变领域模型与环境变量配置**

```python
@dataclass(frozen=True)
class ModelCapability:
    name: str
    status: Literal["ready", "package_missing", "model_missing", "disabled"]
    detail: str

class TreeState(BaseModel):
    mode: Literal["observing", "friendly", "signal", "risk", "recovering"]
    health: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    bloom: float = Field(ge=0, le=1)
    wind: float = Field(ge=0, le=1)

class LiveState(BaseModel):
    session_id: str
    status: Literal["ready", "running", "stopped", "error"]
    listening_state: Literal["idle", "listening", "transcribing", "analyzing", "unavailable"]
    sbi: float = Field(ge=0, le=100)
    friendliness: float = Field(ge=0, le=100)
    evidence_status: EvidenceStatus
    provenance: Literal["measured"] = "measured"
    tree: TreeState
    vision: VisionObservation
    audio: AudioObservation
    transcript: TranscriptObservation
    metrics: dict[str, float]
```

`Settings.from_env()` 同时解析 `SOCI_YOLO_WEIGHTS`、`SOCI_SENSEVOICE_MODEL_DIR` 与 `SOCI_LIVE_VIDEO_FPS`，未配置时回退到 SRTP 现有权重目录。

- [ ] **Step 4: 运行测试确认领域模型与能力检测通过**

Run: `python -m pytest tests/test_live_capabilities.py -v`

Expected: PASS，且导入能力检测不会加载 Torch、Ultralytics 或 FunASR。

- [ ] **Step 5: 提交**

```bash
git add soci_ai/live/__init__.py soci_ai/live/domain.py soci_ai/live/capabilities.py soci_ai/config.py tests/test_live_capabilities.py
git commit -m "feat: define realtime self-monitoring capabilities"
```

### Task 2: 真实人脸定位与 YOLOv8 表情适配器

**Files:**
- Create: `soci_ai/live/vision.py`
- Test: `tests/test_live_vision.py`

**Interfaces:**
- Consumes: `Settings.yolo_weights`；Task 1 的 `VisionObservation`。
- Produces: `YoloVisionAdapter.analyze(image_bytes: bytes, captured_at_ms: int) -> VisionObservation`；归一化边界框 `{x, y, width, height}` 和 `features={"negative": float, "expression": str}`。

- [ ] **Step 1: 写失败测试，覆盖最大人脸、真实框、风险映射和损坏输入**

```python
def test_adapter_returns_largest_face_box_and_negative_risk(fake_cv2, fake_yolo):
    adapter = YoloVisionAdapter(Path("best.pt"), cv2_module=fake_cv2, model_factory=fake_yolo)
    result = adapter.analyze(JPEG_BYTES, 1200)
    assert result.face_detected is True
    assert result.box.model_dump() == {"x": .1, "y": .2, "width": .5, "height": .5}
    assert result.expression == "angry"
    assert result.features["negative"] > .7
    assert result.provenance == "measured"

def test_rejects_invalid_and_oversized_frames(fake_cv2, fake_yolo):
    adapter = YoloVisionAdapter(Path("best.pt"), cv2_module=fake_cv2, model_factory=fake_yolo)
    with pytest.raises(FrameValidationError):
        adapter.analyze(b"not-an-image", 1)
    with pytest.raises(FrameValidationError):
        adapter.analyze(b"x" * 2_000_001, 2)
    assert fake_yolo.calls == 0
```

- [ ] **Step 2: 运行测试并确认因适配器不存在而失败**

Run: `python -m pytest tests/test_live_vision.py -v`

Expected: FAIL importing `YoloVisionAdapter`。

- [ ] **Step 3: 实现懒加载适配器**

```python
class YoloVisionAdapter:
    MAX_FRAME_BYTES = 2_000_000

    def analyze(self, image_bytes: bytes, captured_at_ms: int) -> VisionObservation:
        frame = self._decode_and_resize(image_bytes, longest_edge=480)
        face = self._largest_face(frame)
        if face is None:
            return VisionObservation.no_face(captured_at_ms)
        prediction = self._predict(self._crop(frame, face))
        return self._observation(frame, face, prediction, captured_at_ms)
```

模型只在首次有效人脸帧到来时加载并固定 `device="cpu"`。类别名采用归一化映射；`angry/disgust/fear/sad` 进入 negative 风险，`happy/neutral/surprise` 不直接判为微冒犯。只处理面积最大的人脸。

- [ ] **Step 4: 运行视觉测试**

Run: `python -m pytest tests/test_live_vision.py -v`

Expected: PASS，包括无人脸返回 `face_detected=False`、损坏帧不调用模型。

- [ ] **Step 5: 提交**

```bash
git add soci_ai/live/vision.py tests/test_live_vision.py
git commit -m "feat: add lazy local vision perception"
```

### Task 3: 真实声学窗口、SenseVoice 转写与微冒犯语义分析

**Files:**
- Create: `soci_ai/live/audio.py`
- Create: `soci_ai/live/semantics.py`
- Create: `soci_ai/live/bias_patterns.json`
- Create: `tests/test_live_audio.py`
- Create: `tests/test_live_semantics.py`

**Interfaces:**
- Consumes: Task 1 的 `AudioObservation`, `TranscriptObservation` 与 `Settings.sensevoice_model_dir`。
- Produces: `validate_audio_features(payload: AudioFeaturePayload) -> AudioObservation`；`SenseVoiceTranscriber.transcribe_pcm16(pcm: bytes, sample_rate: int, at_ms: int) -> TranscriptObservation`；`MicroaggressionAnalyzer.analyze(text: str, at_ms: int) -> PerceptionEvent`。

- [ ] **Step 1: 写失败测试，覆盖真实状态、静音、过期块与中文偏见语句**

```python
def test_sensevoice_result_strips_tags_and_keeps_emotion(fake_model):
    transcriber = SenseVoiceTranscriber(Path("SenseVoiceSmall"), model_factory=lambda **_: fake_model)
    result = transcriber.transcribe_pcm16(LOUD_PCM, 16000, 3000)
    assert result.text == "女生不适合学工科"
    assert "ANGRY" in result.tags
    assert result.status == "completed"

def test_silent_short_and_stale_audio_are_not_transcribed(fake_model):
    transcriber = SenseVoiceTranscriber(Path("SenseVoiceSmall"), model_factory=lambda **_: fake_model)
    assert transcriber.transcribe_pcm16(SILENT_PCM, 16000, 1000).status == "silent"
    assert transcriber.transcribe_pcm16(b"\0\0", 16000, 2000).status == "too_short"
    transcriber.transcribe_pcm16(LOUD_PCM, 16000, 3000)
    assert transcriber.transcribe_pcm16(LOUD_PCM, 16000, 2500).status == "stale"
    assert fake_model.calls == 1

def test_semantic_analyzer_explains_stereotype():
    event = MicroaggressionAnalyzer.default().analyze("女生不适合学工科", 3000)
    assert event.features["microbias"] >= .9
    assert event.features["category"] == "gender_stereotype"
    assert "女生不适合" in event.evidence
```

- [ ] **Step 2: 运行测试并确认模块缺失失败**

Run: `python -m pytest tests/test_live_audio.py tests/test_live_semantics.py -v`

Expected: FAIL importing audio/semantic modules。

- [ ] **Step 3: 实现 PCM 校验、内存推理与可解释规则**

```python
class SenseVoiceTranscriber:
    def transcribe_pcm16(self, pcm: bytes, sample_rate: int, at_ms: int) -> TranscriptObservation:
        samples = np.frombuffer(pcm, dtype="<i2")
        if sample_rate != 16000 or samples.size < 3200:
            return TranscriptObservation.rejected(at_ms, "too_short_or_bad_rate")
        peak = int(np.max(np.abs(samples.astype(np.int32))))
        if peak < 500:
            return TranscriptObservation.silent(at_ms)
        result = self._model().generate(input=np.ascontiguousarray(samples / 32768.0), is_final=True)
        return self._parse_result(result, at_ms, peak, samples.size / sample_rate)

class MicroaggressionAnalyzer:
    def analyze(self, text: str, at_ms: int) -> PerceptionEvent:
        matches = self._match_patterns(text)
        return PerceptionEvent(
            at_ms=at_ms,
            modality="semantic",
            features={"microbias": self._score(matches), "category": self._category(matches)},
            confidence=self._confidence(matches, text),
            source="live_semantics",
            evidence=self._evidence(matches),
        )
```

`bias_patterns.json` 至少包含 gender stereotype、regional stereotype、ability dismissal、exclusion、belittling 五类中文模式，每条具有 phrase、weight、explanation。未命中时返回零风险而不是默认负面。

- [ ] **Step 4: 运行音频与语义测试**

Run: `python -m pytest tests/test_live_audio.py tests/test_live_semantics.py -v`

Expected: PASS；测试使用注入假模型，不加载 936 MB 权重。

- [ ] **Step 5: 提交**

```bash
git add soci_ai/live/audio.py soci_ai/live/semantics.py soci_ai/live/bias_patterns.json tests/test_live_audio.py tests/test_live_semantics.py
git commit -m "feat: add local speech and microaggression analysis"
```

### Task 4: 实时自检运行时、融合与状态平滑

**Files:**
- Create: `soci_ai/live/runtime.py`
- Modify: `soci_ai/fusion.py`
- Modify: `soci_ai/interventions.py`
- Test: `tests/test_live_runtime.py`
- Modify: `tests/test_fusion.py`

**Interfaces:**
- Consumes: Task 2 `VisionObservation`；Task 3 `AudioObservation`, `TranscriptObservation`, `MicroaggressionAnalyzer`；现有 `FusionEngine.update(events, now_ms)` 与 `InterventionPolicy.decide(...)`。
- Produces: `LiveSessionRuntime.start() -> LiveState`、`accept_vision(negative, confidence, at_ms) -> LiveState`、`accept_audio(arousal, confidence, at_ms) -> LiveState`、`accept_transcript(text, at_ms) -> LiveState`、`submit_frame(...) -> LiveState`、`submit_audio_features(...) -> LiveState`、`submit_audio_chunk(...) -> LiveState`、`stop() -> LiveState`、`snapshot() -> dict[str, object]`。

- [ ] **Step 1: 写失败测试，固定 measured provenance、丢弃过期帧、恢复与幂等生命周期**

```python
async def test_live_multimodal_events_raise_self_risk_and_then_recover(runtime):
    await runtime.start()
    await runtime.accept_vision(negative=.72, confidence=.9, at_ms=1000)
    await runtime.accept_audio(arousal=.74, confidence=.88, at_ms=1200)
    await runtime.accept_transcript("女生不适合学工科", at_ms=1500)
    high = runtime.state()
    assert high.provenance == "measured"
    assert high.evidence_status == "sufficient"
    assert high.sbi >= 45
    assert high.tree.mode in {"signal", "risk"}
    await runtime.tick(12000)
    assert runtime.state().tree.mode == "recovering"

async def test_busy_runtime_drops_stale_frames(runtime):
    first = asyncio.create_task(runtime.submit_frame(b"first", 1000))
    await runtime.vision.started.wait()
    await runtime.submit_frame(b"stale-middle", 1100)
    await runtime.submit_frame(b"latest", 1300)
    runtime.vision.release.set()
    await first
    assert runtime.vision_calls == [b"first", b"latest"]
    assert runtime.state().last_frame_at_ms == 1300

async def test_start_and_stop_are_idempotent(runtime):
    first = await runtime.start()
    assert (await runtime.start()).session_id == first.session_id
    await runtime.stop()
    assert (await runtime.stop()).status == "stopped"
```

- [ ] **Step 2: 运行测试并确认实时运行时缺失失败**

Run: `python -m pytest tests/test_live_runtime.py tests/test_fusion.py -v`

Expected: FAIL importing `LiveSessionRuntime`；现有模拟融合测试保持可运行。

- [ ] **Step 3: 实现带锁的实时运行时和 measured 融合入口**

```python
class LiveSessionRuntime:
    async def submit_frame(self, image: bytes, captured_at_ms: int) -> LiveState:
        self._pending_frame = (image, captured_at_ms)
        if self._vision_lock.locked():
            return self._state
        async with self._vision_lock:
            while self._pending_frame is not None:
                newest, newest_at = self._pending_frame
                self._pending_frame = None
                observation = await asyncio.to_thread(self.vision.analyze, newest, newest_at)
                self._accept_events(observation.events(), newest_at)
            return self._publish_state()

    def _publish_state(self) -> LiveState:
        snapshot = self.fusion.update(self._pending_events, self.elapsed_ms)
        tree = self.tree_mapper.map(snapshot, previous=self._state.tree)
        self._state = LiveState.from_fusion(snapshot, tree, provenance="measured")
        return self._state
```

`FusionEngine.update()` 增加可选 `provenance` 参数，默认仍为 `simulated`，实时调用传入 `measured`。干预文案改为面向使用者自身，例如视觉主导时提示“放松眉间和下颌，再继续表达观点”。

- [ ] **Step 4: 运行运行时与融合回归测试**

Run: `python -m pytest tests/test_live_runtime.py tests/test_fusion.py tests/test_interventions.py -v`

Expected: PASS；模拟测试仍返回 simulated，实时测试返回 measured。

- [ ] **Step 5: 提交**

```bash
git add soci_ai/live/runtime.py soci_ai/fusion.py soci_ai/interventions.py tests/test_live_runtime.py tests/test_fusion.py
git commit -m "feat: fuse measured self-monitoring signals"
```

### Task 5: 实时 API、隐私持久化与错误语义

**Files:**
- Create: `soci_ai/live/api.py`
- Modify: `soci_ai/api.py`
- Modify: `soci_ai/store.py`
- Create: `tests/test_live_api.py`
- Modify: `tests/test_store_reporting.py`

**Interfaces:**
- Consumes: Task 1 `detect_live_capabilities`；Task 4 `LiveSessionRuntime`。
- Produces: `/api/live/capabilities`、`POST /api/live/sessions`、`POST /api/live/sessions/{id}/frame`、`POST /api/live/sessions/{id}/audio-features`、`POST /api/live/sessions/{id}/audio-chunk`、`GET /api/live/sessions/{id}/state`、`POST /api/live/sessions/{id}/stop`。

- [ ] **Step 1: 写失败 API 测试，覆盖生命周期、原始媒体不落库、错误和幂等**

```python
def test_live_session_accepts_real_inputs_and_never_persists_raw_media(client, live_stubs):
    session = client.post("/api/live/sessions").json()
    sid = session["session_id"]
    frame = client.post(f"/api/live/sessions/{sid}/frame?at_ms=1000", content=JPEG_BYTES, headers={"content-type": "image/jpeg"})
    audio = client.post(f"/api/live/sessions/{sid}/audio-chunk?sample_rate=16000&at_ms=2000", content=LOUD_PCM)
    assert frame.status_code == audio.status_code == 200
    stored = client.app.state.store.get_session(sid)
    assert "JPEG_BYTES" not in json.dumps(stored)
    assert "raw_audio" not in json.dumps(stored)

def test_start_and_stop_are_idempotent(client):
    first = client.post("/api/live/sessions").json()
    second_stop = client.post(f"/api/live/sessions/{first['session_id']}/stop")
    third_stop = client.post(f"/api/live/sessions/{first['session_id']}/stop")
    assert second_stop.status_code == third_stop.status_code == 200
```

- [ ] **Step 2: 运行测试并确认路由不存在**

Run: `python -m pytest tests/test_live_api.py tests/test_store_reporting.py -v`

Expected: FAIL with live endpoints returning 404。

- [ ] **Step 3: 实现路由和结构化错误，不让实时模型阻塞事件循环**

```python
@router.post("/sessions/{session_id}/frame")
async def submit_frame(session_id: str, request: Request, at_ms: int) -> dict[str, object]:
    content_type = request.headers.get("content-type", "")
    if content_type not in {"image/jpeg", "image/webp"}:
        raise LiveApiError(415, "unsupported_frame_type")
    return (await registry.get(session_id).submit_frame(await request.body(), at_ms)).model_dump(mode="json")

@router.post("/sessions/{session_id}/audio-chunk")
async def submit_audio(session_id: str, request: Request, sample_rate: int, at_ms: int):
    return (await registry.get(session_id).submit_audio_chunk(await request.body(), sample_rate, at_ms)).model_dump(mode="json")
```

`SessionStore.start_session("live_self_check")` 复用现有表；frame payload 仅写特征、转录、融合与建议。FastAPI lifespan 关闭实时运行时并释放模型引用。

- [ ] **Step 4: 运行 API 与存储测试**

Run: `python -m pytest tests/test_live_api.py tests/test_store_reporting.py tests/test_runtime_api.py -v`

Expected: PASS；展示模式 API 回归通过，实时媒体字节不出现在 SQLite payload。

- [ ] **Step 5: 提交**

```bash
git add soci_ai/live/api.py soci_ai/api.py soci_ai/store.py tests/test_live_api.py tests/test_store_reporting.py
git commit -m "feat: expose private realtime sensing API"
```

### Task 6: 浏览器真实媒体采集、自拍框与语音状态机

**Files:**
- Create: `web/js/media.js`
- Create: `web/js/live-api.js`
- Create: `web/js/live.js`
- Create: `tests/test_frontend_live_contract.py`

**Interfaces:**
- Consumes: Task 5 实时 API；DOM ids `camera-video`, `face-overlay`, `audio-wave`, `speech-state`, `live-transcript`, `start-live`, `stop-live`（Task 8 创建）。
- Produces: `MediaCapture.start()`、`MediaCapture.stop()`、`LiveController.start()`、`LiveController.stop()`、`LiveController.applyState(state)`；以 3 FPS 发送镜像自拍帧，以 16 kHz PCM16 发送断句音频。

- [ ] **Step 1: 写失败契约测试，锁定真实 getUserMedia、框坐标和语音状态**

```python
def test_live_media_module_uses_real_camera_and_microphone(client):
    source = client.get("/assets/js/media.js").text
    assert "navigator.mediaDevices.getUserMedia" in source
    assert "video: true" in source
    assert "audio: true" in source
    assert "MediaRecorder" not in source
    assert "encodePcm16" in source

def test_face_overlay_uses_source_dimensions_and_mirror_flag(client):
    source = client.get("/assets/js/live.js").text
    assert "state.vision.box" in source
    assert "video.videoWidth" in source
    assert "data-mirrored" in source
```

- [ ] **Step 2: 运行测试并确认新资源返回 404**

Run: `python -m pytest tests/test_frontend_live_contract.py -v`

Expected: FAIL because `/assets/js/media.js` and `/assets/js/live.js` do not exist。

- [ ] **Step 3: 实现真实媒体采集与状态机**

```javascript
export class MediaCapture {
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 }
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.startFrameLoop(1000 / 3);
    await this.startAudioGraph(16000);
  }
}

export function drawFaceBox(canvas, video, box) {
  const x = (1 - box.x - box.width) * canvas.clientWidth;
  const y = box.y * canvas.clientHeight;
  canvas.getContext("2d").strokeRect(x, y, box.width * canvas.clientWidth, box.height * canvas.clientHeight);
}
```

音频使用 AudioContext/AudioWorklet 可用时优先，兼容路径使用 ScriptProcessor；浏览器端 VAD 在连续静音后把内存中的 PCM16 语句块发送后端。UI 状态顺序为 listening → transcribing → analyzing → listening；请求失败进入 unavailable。

- [ ] **Step 4: 运行前端契约测试**

Run: `python -m pytest tests/test_frontend_live_contract.py tests/test_frontend_contract.py -v`

Expected: PASS，且既有无整页滚动、轮询降级契约仍通过。

- [ ] **Step 5: 提交**

```bash
git add web/js/media.js web/js/live-api.js web/js/live.js tests/test_frontend_live_contract.py
git commit -m "feat: capture real browser media for self-check"
```

### Task 7: 程序化动态生命树渲染器

**Files:**
- Create: `web/js/tree.js`
- Create: `tests/test_tree_contract.py`

**Interfaces:**
- Consumes: Task 1/4 的 `TreeState {mode, health, risk, bloom, wind}`。
- Produces: `LivingTree(canvas, options)`、`setTarget(treeState)`、`start()`、`stop()`、`resize()`；无图片素材依赖。

- [ ] **Step 1: 写失败测试，固定树的程序化结构、平滑过渡与性能降级**

```python
def test_tree_is_procedural_and_state_driven(client):
    source = client.get("/assets/js/tree.js").text
    for token in ["requestAnimationFrame", "setTarget", "drawBranch", "drawLeaf", "drawBloom", "fallingLeaves"]:
        assert token in source
    assert "new Image(" not in source
    assert "prefers-reduced-motion" in source
    assert "devicePixelRatio" in source
```

- [ ] **Step 2: 运行测试并确认树模块不存在**

Run: `python -m pytest tests/test_tree_contract.py -v`

Expected: FAIL because `/assets/js/tree.js` returns 404。

- [ ] **Step 3: 实现确定性枝干、粒子叶片、花苞、落叶与恢复生长**

```javascript
export class LivingTree {
  setTarget(next) { this.target = sanitizeTreeState(next); }
  frame(now) {
    this.current.health = damp(this.current.health, this.target.health, .055);
    this.current.risk = damp(this.current.risk, this.target.risk, .045);
    this.current.bloom = damp(this.current.bloom, this.target.bloom, .035);
    this.drawGlow();
    this.drawBranch(this.root, -Math.PI / 2, this.depth, now);
    this.updateFallingLeaves(now);
    this.raf = requestAnimationFrame(t => this.frame(t));
  }
}
```

树结构使用固定 seed，避免每帧重建抖动。颜色在银蓝、翡翠、琥珀、暖红和恢复青绿之间插值；风险上升增加枝条收拢和落叶，恢复时沿枝干重新点亮嫩芽。根据最近 60 帧平均耗时把粒子数从 1200 降为 700 或 360。

- [ ] **Step 4: 运行树与前端契约测试**

Run: `python -m pytest tests/test_tree_contract.py tests/test_frontend_live_contract.py -v`

Expected: PASS；树不加载外部图片或网络资源。

- [ ] **Step 5: 提交**

```bash
git add web/js/tree.js tests/test_tree_contract.py
git commit -m "feat: render adaptive living feedback tree"
```

### Task 8: 重构实时主界面并保留备用回放与复盘

**Files:**
- Modify: `web/index.html`
- Modify: `web/styles.css`
- Modify: `web/js/app.js`
- Modify: `web/js/render.js`
- Modify: `web/js/state.js`
- Modify: `tests/test_frontend_contract.py`
- Modify: `tests/test_frontend_live_contract.py`

**Interfaces:**
- Consumes: Task 6 `LiveController` 与 Task 7 `LivingTree`。
- Produces: 默认实时自检视图；主树舞台；自拍卡片；语音卡片；指标条；折叠详情；明确标注的备用回放入口。

- [ ] **Step 1: 写失败前端测试，固定默认入口和关键可视区域**

```python
def test_default_view_is_real_self_check_and_replay_is_labeled(client):
    html = client.get("/").text
    assert 'id="living-tree"' in html
    assert 'id="camera-video"' in html
    assert 'id="face-overlay"' in html
    assert 'id="audio-wave"' in html
    assert 'id="speech-state"' in html
    assert 'id="live-transcript"' in html
    assert 'id="start-live"' in html
    assert "备用回放" in html
    assert "播放场景" not in html.split('id="view-live"', 1)[1].split("</section>", 1)[0]
```

- [ ] **Step 2: 运行测试并确认旧命令中心布局失败**

Run: `python -m pytest tests/test_frontend_contract.py tests/test_frontend_live_contract.py -v`

Expected: FAIL because live tree/media regions are absent and scenario controls are still primary。

- [ ] **Step 3: 实现美观的低认知负担布局**

主舞台采用三层结构：全幅深色渐变与星尘背景；中央 16:10 发光树画布；左右悬浮玻璃卡片。左侧自拍卡保持 4:3、镜像显示，绿色/琥珀人脸框随状态变化；右侧包含综合 SBI、社交友好度、语气压力、视觉紧张度和语义偏差。底部语音卡显示波形、三阶段状态、最新转录与一句修正建议。

```javascript
const tree = new LivingTree(document.getElementById("living-tree"));
const live = new LiveController({
  tree,
  video: document.getElementById("camera-video"),
  overlay: document.getElementById("face-overlay"),
  onState: state => renderLiveState(state, tree)
});
document.getElementById("start-live").addEventListener("click", () => live.start());
document.getElementById("stop-live").addEventListener("click", () => live.stop());
```

导航顺序调整为“实时自检、会后复盘、机制解释、研究证据、备用回放”。复盘复用现有图表和证据组件；模拟状态所有标签继续显示 SIMULATED，实时状态显示 MEASURED。

- [ ] **Step 4: 运行前端全套契约测试**

Run: `python -m pytest tests/test_frontend_contract.py tests/test_frontend_live_contract.py tests/test_tree_contract.py -v`

Expected: PASS，页面没有远程资源，实时入口无模拟自动播放。

- [ ] **Step 5: 提交**

```bash
git add web/index.html web/styles.css web/js/app.js web/js/render.js web/js/state.js tests/test_frontend_contract.py tests/test_frontend_live_contract.py
git commit -m "feat: make living tree the realtime primary interface"
```

### Task 9: 依赖、启动器、端到端检查与答辩说明

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-live.txt`
- Modify: `start_showcase.ps1`
- Modify: `start_showcase.bat`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Create: `docs/live-demo-checklist.md`
- Modify: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: Tasks 1–8 的完整实时链路。
- Produces: 可在现有 YOLOv8 Python 环境安装的依赖清单、默认 edge 启动命令、模型自检输出、人工硬件验收清单。

- [ ] **Step 1: 写失败端到端测试，固定默认启动模式与能力说明**

```python
def test_launchers_start_edge_mode_and_document_real_capabilities():
    ps1 = Path("start_showcase.ps1").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "$env:SOCI_MODE = \"edge\"" in ps1
    assert "单人自检" in readme
    assert "声纹注册与多人分离：扩展接口已设计，当前未实现" in readme
    assert "python -m pip install -r requirements-live.txt" in readme

def test_full_live_api_smoke(client_with_stubs):
    sid = client_with_stubs.post("/api/live/sessions").json()["session_id"]
    assert client_with_stubs.post(f"/api/live/sessions/{sid}/frame?at_ms=1000", content=JPEG_BYTES, headers={"content-type": "image/jpeg"}).status_code == 200
    assert client_with_stubs.get(f"/api/live/sessions/{sid}/state").json()["provenance"] == "measured"
    assert client_with_stubs.post(f"/api/live/sessions/{sid}/stop").json()["status"] == "stopped"
```

- [ ] **Step 2: 运行端到端测试并确认文档/启动模式不符合**

Run: `python -m pytest tests/test_end_to_end.py -v`

Expected: FAIL because launchers still start showcase mode and live installation docs are absent。

- [ ] **Step 3: 添加分层依赖和真实能力自检**

`requirements.txt` 保持 Web 服务基础依赖；`requirements-live.txt` 增加以下本地推理依赖：

```text
-r requirements.txt
numpy>=1.24,<3
opencv-python>=4.8,<5
ultralytics>=8.0,<9
funasr>=1.1,<2
```

启动器先运行 `python -c "from soci_ai.config import Settings; from soci_ai.live.capabilities import print_capabilities; print_capabilities(Settings.from_env(__import__('pathlib').Path.cwd()))"`，随后设置 `SOCI_MODE=edge` 并启动 Uvicorn。缺少 FunASR 时仍启动基础模式并给出安装提示。

- [ ] **Step 4: 补充真实答辩演示清单并运行完整测试**

`docs/live-demo-checklist.md` 明确列出：摄像头授权、人脸框、麦克风波形、中文转写、测试语句、树状态变化、停止与复盘、拒绝权限降级、模型缺失降级，以及声纹/多人分离为下一阶段工作。

Run: `python -m pytest -q`

Expected: PASS all tests with no failures；测试运行不需要物理摄像头或麦克风，也不加载重模型。

- [ ] **Step 5: 在目标环境安装实时依赖并运行只读能力检测**

Run: `python -m pip install -r requirements-live.txt`

Expected: 安装成功或已满足依赖；不升级/替换现有 PyTorch CUDA 构建。

Run: `python -c "from pathlib import Path; from soci_ai.config import Settings; from soci_ai.live.capabilities import print_capabilities; print_capabilities(Settings.from_env(Path.cwd()))"`

Expected: YOLOv8 权重为 ready；OpenCV/Ultralytics 为 ready；SenseVoice 在 FunASR 可导入且模型目录存在时为 ready。

- [ ] **Step 6: 启动服务并完成人工浏览器验收**

Run: `powershell -ExecutionPolicy Bypass -File .\start_showcase.ps1`

Expected: `http://127.0.0.1:8001/` 可访问；浏览器授权后出现真实自拍、人脸框、真实波形与真实状态指示；讲话后出现转录；树随 measured 状态平滑变化；停止后可查看复盘。

- [ ] **Step 7: 提交**

```bash
git add requirements.txt requirements-live.txt start_showcase.ps1 start_showcase.bat README.md docs/architecture.md docs/live-demo-checklist.md tests/test_end_to_end.py
git commit -m "docs: package realtime self-check demonstration"
```

## Completion Verification

- [ ] Run: `python -m pytest -q`
  - Expected: all tests pass, zero failures.
- [ ] Run: `git diff --check`
  - Expected: no whitespace errors.
- [ ] Run: `python -m soci_ai --host 127.0.0.1 --port 8001`
  - Expected: server starts in edge mode through launcher and `/api/health` responds.
- [ ] Browser check at 1366×768 and 1920×1080.
  - Expected: tree remains the dominant visual; selfie, face box, speech state, transcript and primary controls remain visible without horizontal overflow.
- [ ] Manual privacy check.
  - Expected: `data/` contains SQLite structure only; no `.jpg`, `.webp`, `.wav`, `.pcm` or other raw media files are created.
- [ ] Manual capability truth check.
  - Expected: actual model/permission failures are visibly reported; no unavailable capability is shown as active.
