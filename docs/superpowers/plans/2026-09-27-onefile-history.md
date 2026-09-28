# SOCI-AI 单文件分发与本机历史 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有原生桌面软件上实现持久化会话历史、选择查看与逐条删除，并交付不依赖相邻文件的单个 Windows EXE。

**Architecture:** 模型/语料保持只读资源，用户数据迁到固定的本机账号目录。SQLite 存储、后台队列和会话协调分层，与实时采集线程解耦；回顾页复用现有报告和图表。最后用独立输出目录构建单文件，并做跨进程验证。

**Tech Stack:** Windows x64、既有 `D:\minicoda3\python.exe`、Python 标准库 sqlite3/threading/queue/json、现有 PySide6 QtWidgets、PyInstaller 6.22.3；不新增云服务或模型依赖。

**Spec:** `D:/desktop_3/SRTP/SOCI-AI-Studio/docs/superpowers/specs/2026-09-27-onefile-history-design.md`

## Global Constraints

- 运行中每 5 秒保存一次进度；结束检测或正常关闭软件时保存最终快照。
- 生产数据目录固定为 `%LOCALAPPDATA%/SOCI-AI-Desktop`；数据库 `history.sqlite3`，日志 `logs`，默认导出 `exports`。
- 不保存原始视频、照片、录音或未确认字幕；不上传，不混用不同 Windows 账号。
- 不修改视觉算法、语义库、融合权重和生命树评分驱动逻辑，不删除旧安装或原有导出报告。
- 保留三项导航、实时一屏布局和固定反馈卡；只在回顾页加入历史控件。
- 已结束/中断会话才可删除；必须确认，只删单个 ID，导出文件不受影响。
- 单文件自检不打开摄像头/麦克风，必须使用隔离数据目录；交付不含开发者数据。
- 不承诺防反编译、取证级安全擦除、断电零丢失或未经验证的硬件兼容性。
- 延续用户此前选择的 Native 执行方式，在当前工程基础上修改；不得将当前大量已有修改清理、回退或丢弃。需要隔离时先遵循工作区技能，不直接创建不含既有桌面代码的空 worktree。

## Review Focus

- EXE 被移到下载目录、中文路径或只读目录：依然从内置资源加载，历史仍在固定账号目录，不向 EXE 旁写数据（任务 1、6）。
- 历史很多或用户连续快速切换：所有记录可访问，旧异步加载不能覆盖新选择，导出必须是当前所选报告（任务 4）。
- 磁盘满/锁定后用户又开新会话：失败快照按会话 ID 保留，不被下一次会话覆盖；明确显示未保存（任务 2、3、5）。
- 结束、立即关闭、删除和晚到检查点交错：只保存一条、不复活、不把已结束改回运行中（任务 2、3、5）。
- 有新版本/损坏数据库或单条 JSON 损坏：保留原数据，不自动重建覆盖，其他可用功能仍能用（任务 2、4）。

---

## 文件责任与接口边界

- `resources.py`：资源根、固定数据根、日志/导出根和仅自检可用的数据目录覆盖。
- 新增 `history_store.py`：单线程 SQLite CRUD、schema 和报告字段过滤，不导入 Qt/采集/模型。
- 新增 `history_worker.py`：单个工作线程、检查点合并、控制任务排序、结果队列与错误返回。
- 新增 `history_controller.py`：会话 ID/5 秒间隔、待保存快照、当前/历史选择令牌及导出目标，不直接持有 Qt 控件。
- `ui.py`：历史选择和删除交互、保存状态、异步关闭入口；继续复用 `set_report()`。
- `__main__.py`：连接生命周期/计时器/历史服务，协调安全退出及隔离诊断。
- `build_desktop.py`：分别支持 onedir 诊断构建和独立 onefile 交付。
- 新增 `tools/check_onefile_history.py`：只通过 EXE 诊断命令做跨进程读写、删除、迁移路径验证。
- `docs/desktop-quickstart.md`、`docs/desktop-packaging.md`：真实使用提示和打包限制。

## Task 1: 固定用户数据路径与自检隔离

**Files:** Modify `soci_ai/desktop/resources.py`, `desktop_launcher.py`; Create `tests/test_desktop_resources.py`。

**Interfaces:** 保留 `resource_root() -> Path`、`model_root() -> Path`、`output_root() -> Path`；新增 `log_root() -> Path`、`export_root() -> Path`。`output_root()` 只解析根目录，不在导入时创建目录。诊断覆盖使用 `SOCI_AI_TEST_DATA_DIR`，仅当启动参数包含 `--self-test-models`、`--smoke-test` 或 `--self-test-history` 时生效；正常运行忽略该变量。

- [ ] **Step 1: 写失败测试。** 固定案例：模拟两个不同 `LOCALAPPDATA`，`output_root()` 分别等于 `<账号目录>/SOCI-AI-Desktop`；模拟冻结程序和 `_MEIPASS`，`model_root()` 指向解压资源但 `output_root()` 不受其影响；正常 argv 下忽略测试变量，自检 argv 下采用显式绝对测试路径；缺少 `LOCALAPPDATA` 时使用用户 `AppData/Local`，拒绝相对测试路径；仅调用路径函数不得创建文件。
- [ ] **Step 2: 运行红灯。** `& 'D:\minicoda3\python.exe' -X utf8 -m unittest tests.test_desktop_resources -v`，应因旧同目录行为失败，而不是缺少测试运行环境。
- [ ] **Step 3: 实现以上签名。** 使用 stdlib 路径解析；日志入口改为 `log_root()/desktop.log`。目录不可写不能阻止程序显示保存故障提示：日志打开失败时采用不落盘的诊断输出，不能回退写到未知目录或让程序在窗口出现前直接崩溃。
- [ ] **Step 4: 验证绿灯与回归。** 运行任务测试及 `tests.test_desktop_geometry`，确认启动尺寸行为没变；在测试目录模拟日志创建失败，启动入口仍能进入主程序。
- [ ] **Step 5: 保存进度。** 新测试文件可单独提交；既有未提交文件先保存局部备份和本次差异，禁止 `git add .` 或将不相关改动混入提交。

## Task 2: SQLite 历史存储与防复活

**Files:** Create `soci_ai/desktop/history_store.py`, `tests/test_desktop_history_store.py`。

**Interfaces:** `HistoryStore(path: Path)`；`create_session(session_id: str, started_at: str, report: dict, app_version: str) -> None`；`checkpoint(session_id: str, report: dict) -> bool`；`finish(session_id: str, report: dict, ended_at: str) -> bool`；`list_sessions(offset: int = 0, limit: int = 100) -> list[dict]`；`load_session(session_id: str) -> dict | None`；`delete_session(session_id: str) -> bool`；`recover_interrupted() -> int`；`close() -> None`。

`list_sessions` 元数据含 `id/started_at/ended_at/updated_at/status/elapsed_seconds/average_sbi/peak_sbi/app_version`，不带报告正文。`load_session` 返回这些字段及 `report`。状态字符串 `running/completed/interrupted`；DB `PRAGMA user_version=1`，报告版本 `1`。时间采用带时区 ISO 字符串，列表按开始时间与 ID 倒序，带分页。

- [ ] **Step 1: 写失败测试。** 用临时 SQLite 文件实现以下断言（测试必须关闭连接后重新打开）：

```python
store.create_session('a', '2026-09-27T09:00:00+08:00', report_a, 'test')
store.finish('a', report_a, '2026-09-27T09:01:00+08:00')
store.create_session('b', '2026-09-27T09:02:00+08:00', report_b, 'test')
store.finish('b', report_b, '2026-09-27T09:03:00+08:00')
assert store.delete_session('a') is True
assert store.checkpoint('a', report_a) is False
assert store.finish('a', report_a, '2026-09-27T09:04:00+08:00') is False
assert store.load_session('a') is None
assert store.load_session('b')['report']['history'] == report_b['history']
```

同时固定：第二次 finish 不覆盖终态快照；running 不可删除；空信号分数为 None；中断恢复只更改 running；分页可访问 205 条而不漏重复；schema=2 原文件不变并报不支持；单条坏 JSON 不影响元数据列表和另一条报告；非 SQLite 文件不被覆盖。
- [ ] **Step 2: 运行红灯。** `& 'D:\minicoda3\python.exe' -X utf8 -m unittest tests.test_desktop_history_store -v`，应明确缺少新模块/行为。
- [ ] **Step 3: 实现存储 API。** 参数化 SQL、事务建表/升级、索引、合理的短锁等待（250 毫秒）。仅 create 执行 INSERT；checkpoint/finish 用 `UPDATE ... WHERE id=? AND status='running'`，不 UPSERT。重复已存在 create 不覆盖该记录。完成/删除不保存永久的含敏感正文备份。
- [ ] **Step 4: 实现报告白名单与失败测试。** 保留规格列出的报告字段和来源信息；未知媒体字段过滤，含嵌套媒体对象的非法报告拒绝；确认 transcripts 只保存已经确认的记录。锁定/不可写/磁盘满注入应回滚并抛可展示错误，不显示写入成功；无法识别的旧 schema 不自动迁移猜测。
- [ ] **Step 5: 运行绿灯并提交新模块。** 运行该测试文件，确认所有断言通过。`git add -- soci_ai/desktop/history_store.py tests/test_desktop_history_store.py`；仅提交这两个新文件。

## Task 3: 非阻塞保存队列与会话协调

**Files:** Create `soci_ai/desktop/history_worker.py`, `soci_ai/desktop/history_controller.py`, `tests/test_desktop_history_worker.py`, `tests/test_desktop_history_controller.py`。

**Interfaces:** `HistoryWorker(path: Path, store_factory=HistoryStore)`；`start() -> None`；`submit(operation: str, request_id: int, **payload) -> None`；`drain_results() -> list[dict]`；`request_shutdown() -> None`；`is_alive() -> bool`。操作固定为 `create/checkpoint/finish/list/load/delete/recover/retry`，payload 对应任务 2 的方法参数；结果字典含 `operation/request_id/session_id/ok/value/error`。

`list` 操作接收 `offset` 和 `limit=100`，实际向存储读取 `limit+1` 条，结果 `value={"items": 前 limit 条, "has_more": 是否有额外一条, "offset": offset}`。`retry` 仅重新打开失败的存储连接，不隐式重放操作；协调器收到成功结果后按原 ID 重发尚未保存的 create/final 快照。

`HistoryController(worker: HistoryWorker, clock=monotonic)`；`begin(report: dict, started_at: str, app_version: str) -> str`；`update_current(report: dict) -> None`；`finish_current(report: dict, ended_at: str) -> None`；`select(session_id: str | None) -> None`（None 为当前）；`refresh_history(offset: int = 0) -> None`；`delete_selected() -> None`；`retry_failed() -> None`；`handle_results() -> bool`；`export_report() -> dict`；`pending_writes() -> bool`。只读状态：`active_id/selected_id/items/has_more/selected_report/save_status/save_error/load_error`。分页 offset=0 替换列表，offset>0 按 ID 去重追加；“更多记录”请求 offset=len(items)。

- [ ] **Step 1: 写失败测试。** 用事件控制假存储阻塞写盘；submit 返回且主线程能继续更新假 session，不等待阻塞解除。提交同会话 100 次检查点，等待槽只有最新普通快照；create 必须先于更新，finish 不被丢弃。重复 finish 不创建第二行，完成后晚到普通快照无效，删除后不恢复。
- [ ] **Step 2: 加入协调红灯。** 假 clock 在 4.9 秒时无第二次检查点，5.0 秒提交最新报告；新 begin 得到不同 ID；旧会话 finish 失败后新 begin 不覆盖待重试的旧快照；两个失败的最终快照按各自 ID 重试。快速 select('a')/select('b') 时，a 的晚到结果不能覆盖 b；更新当前报告不覆盖选中的历史；export_report 返回所选快照。
- [ ] **Step 3: 运行红灯。** `& 'D:\minicoda3\python.exe' -X utf8 -m unittest tests.test_desktop_history_worker tests.test_desktop_history_controller -v`。
- [ ] **Step 4: 实现工作线程。** 数据库连接和 JSON 编码只在该线程内执行；普通检查点每个活动会话一个合并槽；有序控制任务单独保留，finish 消除该 ID 的旧检查点，终态由存储再次防护。转移已稳定的报告快照，不能跨线程共享会继续修改的 dict/list。异常变成结果而不是杀死线程；初始化失败时 retry 可重新打开，不丢本次等待的快照。
- [ ] **Step 5: 实现协调器。** 生成 UUID，5.0 秒间隔，控制请求序号与选择令牌，保留按 ID 分开的失败最终快照，删除前再核查非当前状态。中文状态为“保存中/已保存/保存失败”；重试必须区分 create 失败与最终更新失败，不能显示不存在记录已保存。首次列表加载且没有当前会话时选最近记录；后续刷新保留选择。删除成功后选择列表中相邻一条，最后一条删除后清空；新增会话或停止会话不抢占已选历史。
- [ ] **Step 6: 验证绿灯并提交新模块。** 运行本任务与任务 2 测试；确认 slow-store 测试使用同步事件而不是易波动的微秒性能断言。只暂存本任务四个新文件提交。

## Task 4: 回顾页历史选择、导出与逐条删除

**Files:** Modify `soci_ai/desktop/ui.py`; Create `tests/test_desktop_history_ui.py`, `tools/check_history_presentation.py`。

**Interfaces:** 新信号 `history_selected = Signal(object)`（ID 或 None）、`history_refresh_requested = Signal()`、`history_more_requested = Signal()`、`history_delete_requested = Signal(str)`、`history_retry_requested = Signal()`。新增 `set_history_items(items: list[dict], selected_id: str | None, *, has_more: bool = False) -> None`、`set_history_status(text: str, *, error: bool = False) -> None`、`set_history_selection(session_id: str | None) -> None`、`displayed_report() -> dict`。`set_report(report: dict)` 继续负责现有图表和建议。

- [ ] **Step 1: 写失败离屏 Qt 测试。** 日期/时长/均值/峰值/状态的中文列表；空历史禁用删除和导出；切换 ID 发出对应信号，不因为重新填充列表重复加载；running 项不可删除；取消确认不发信号；确认仅发选中 ID。先显示 report_a，再切换并加载 report_b，displayed_report 必须是 b。
- [ ] **Step 2: 加入长列表/错误测试。** 有 205 条记录时用“更多记录”按 100 条追加并可选择最后一条，三页的 has_more 依次为 True/True/False；首次有历史默认显示最近记录，无历史明确空状态；快速选择在控制层通过令牌验证；选中新条但加载失败时清空旧报告并禁用导出，不误导出旧内容；删除后选择邻近记录，最后一条删除后空图、空建议、无字幕。保留第三任务失败保存的“重试保存”入口。
- [ ] **Step 3: 运行红灯。** `& 'D:\minicoda3\python.exe' -X utf8 -m unittest tests.test_desktop_history_ui -v`；使用 `QT_QPA_PLATFORM=offscreen` 子进程，禁止打开摄像头和麦克风。
- [ ] **Step 4: 实现控件。** 历史区放在回顾页顶部，自适应宽窄窗口；不新增主导航，不挤占实时页。列表显示摘要而非原句长文；确认框含日期和时长，并说明“仅删除本条本机历史，不删除已导出的报告，删除后无法在应用内撤销”。UI 删除成功与失败只接受存储实际返回。
- [ ] **Step 5: 绿灯、视觉 QA 和局部差异保存。** 运行新 UI 测试与 `tests.test_desktop_ui`、`tests.test_desktop_compact_ui`、`tests.test_desktop_hover_ui`。`tools/check_history_presentation.py` 在 `artifacts/native-history-ui` 输出中文字体截图，检查 1440×840、1280×720、960×600 的回顾与实时页；用 view_image 查看实际图片。保留未修改的实时布局，既有未提交 UI 不混入不相关提交。

## Task 5: 入口生命周期、正常退出与历史自检

**Files:** Modify `soci_ai/desktop/__main__.py`, `soci_ai/desktop/ui.py`, `soci_ai/desktop/session.py`（仅在快照元数据有必要时）; Create `tests/test_desktop_history_lifecycle.py`。

**Interfaces:** UI 新增 `set_deferred_close(enabled: bool) -> None`、`allow_close() -> None`；默认不开启延迟关闭以兼容独立 UI 测试，真实入口开启。协调器使用任务 3 API；export 使用 window.displayed_report()。增加 `--self-test-history` 及 `--history-test-phase`，阶段固定 `write/read/delete/read-after-delete`；必须有隔离的 `SOCI_AI_TEST_DATA_DIR`，无设备和真实用户数据访问。

- [ ] **Step 1: 写生命周期失败测试。** 注入假 session/worker，成功 start 后 create 一次；5 秒提交；stop 后最终保存；stop 后 close 不重复创建。关闭事件先 ignore，完成存储和设备释放后 allow_close；慢存储期间不能提前退出。正常无检测关闭不创建空会话。保存失败时保持可操作窗口，可重试或确认放弃；设备停止后不会偷偷重开。
- [ ] **Step 2: 写当前/历史/导出失败测试。** 实时 refresh 只在选择当前会话时更新回顾；停止当前会话不把用户正在看的历史换掉；导出 b 时不是活动会话 a。初始化 DB 失败仍能自检，但显示不能自动保存；关闭时任何强制放弃不删除已经保存的数据。
- [ ] **Step 3: 运行红灯。** `& 'D:\minicoda3\python.exe' -X utf8 -m unittest tests.test_desktop_history_lifecycle -v`；所有注入设备不得接触真实硬件。
- [ ] **Step 4: 连接入口。** 单实例锁成功后初始化历史服务与恢复，100 毫秒现有刷新循环排空结果，历史列表读取按需触发而不是每帧重建。start/stop 保存同一 ID；关闭等待两个条件（工作线程已退出、摄像头/音频已释放），由原有进度对话框显示过程。重试与放弃提示分别处理保存和设备状态，不能通过结束其他 Python/应用进程解决。
- [ ] **Step 5: 实现隔离诊断。** 用固定测试 ID a/b 和带不同 cue 的两个明确标记为测试的报告；write 保存两条，read 重开验证两条，delete 只删 a，read-after-delete 重开确认 b 完整且 a 不存在。每阶段写新的 JSON 诊断结果，包含 phase、ok、camera_opened=false、microphone_opened=false；不能借由诊断创建正常历史。
- [ ] **Step 6: 验证绿灯与全桌面回归。** `& 'D:\minicoda3\python.exe' -X utf8 -m unittest discover -s tests -p 'test_desktop_*.py' -v`。记录实际测试数，不沿用旧版本的 222 条作为本次结果；失败归因并修复，不通过删断言掩盖问题。

## Task 6: 单文件构建和真正单 EXE 验收

**Files:** Modify `build_desktop.py`, `docs/desktop-packaging.md`, `docs/desktop-quickstart.md`; Create `tests/test_desktop_packaging.py`, `tools/check_onefile_history.py`。

**Interfaces:** `build_command(root: Path = ROOT, *, clean: bool = False, mode: str = 'onedir') -> list[str]`；`build_desktop.py --mode onefile --clean`；onefile 名称 `SOCI-AI.exe`，输出 `dist/SOCI-AI-Onefile/SOCI-AI.exe`，独立缓存 `build/native-onefile`。onedir 原路径/名称保留。诊断统一输出 `artifacts/onefile-qa`，不把说明书、语料副本或 data 当作单文件运行必需品。

- [ ] **Step 1: 写打包失败测试。** onefile 命令含 `--onefile` 不含 `--onedir`，所有现有 resource_files 仍内置，明确排除旧网页/研究大依赖；onedir 命令保持兼容；不同 mode 的 dist/work/spec 路径不重叠。打包脚本只验证/更新自身产物，不递归清理项目和旧安装。
- [ ] **Step 2: 运行红灯。** `& 'D:\minicoda3\python.exe' -X utf8 -m unittest tests.test_desktop_packaging -v`。
- [ ] **Step 3: 实现构建分支与诊断路径。** 使用已存在的冻结 hook 和受限 DLL 搜索 PATH；onefile 后置模型/GUI 自检读取隔离目录中本次新生成的结果，不能读到旧的同名文件当作成功。CLI mode 非法值报错。第三方需要保留的许可证与声明打包并在“系统介绍”中提供查看入口；不得声称第三方代码/模型均由本团队原创。
- [ ] **Step 4: 绿灯后构建。** 先运行 `build_desktop.py --mode onefile --check-only` 和 `--print-command`，确认依赖、资源与输出；然后执行 `& 'D:\minicoda3\python.exe' -X utf8 build_desktop.py --mode onefile --clean`。只在新输出目录构建，不覆盖当前用户正在使用的 EXE。
- [ ] **Step 5: 实现并运行独立 EXE 检查器。** `tools/check_onefile_history.py --exe <绝对EXE路径> --output <artifacts/onefile-qa路径>`：创建 workspace 内独立中文测试目录，只复制 EXE，设置隔离数据根，以非项目 cwd 分别运行模型自检、离屏 GUI 自检和四个 history 阶段，要求退出码/新 JSON/断言均成功。再将 EXE 复制到另一目录并运行 read-after-delete，证明移动 EXE 后读取同一数据根；不复制 `_internal` 或 models。报告实际启动耗时、字节数和 SHA-256。
- [ ] **Step 6: 验证与文档。** 运行打包单测、完整桌面回归、独立 EXE 检查器，并查看冻结 GUI 截图。文档说明单文件启动解压、当前账号固定目录、自动保存/删除/导出、原始媒体不保存、本地未加密、旧内存历史不可恢复、Windows 版本兼容需实机测试。删除仅为应用记录删除，不能写成安全擦除。
- [ ] **Step 7: 交付新版本，不自动替换旧包。** 给出 `dist/SOCI-AI-Onefile/SOCI-AI.exe` 的绝对可点击路径、大小与校验摘要。旧 `D:/desktop_3/SRTP/SOCI-AI-Desktop` 保留；用户无需停止其他应用或交付整个项目。正常摄像头/麦克风试用和另一台电脑测试与文件自检结果分开报告。

## 执行前检查与提交策略

- [ ] 用户审阅本计划并确认执行；延续 Native，不主动派发实现子任务。
- [ ] 重新检查 AGENTS.md、Git 状态、工作区隔离状态和既有用户改动。尊重“在现有工程基础上修改”的偏好；禁止创建丢失未提交桌面代码的 checkout。
- [ ] 在 workspace 内创建明确命名的任务备份，仅复制本计划要修改的已有文件，不移动或删除原文件；记录初始差异。
- [ ] 使用现有 Python 运行桌面基线测试，若已有失败先报告原因，未获方向前不将其当作本次造成的问题修复或跳过。
- [ ] 每个任务先运行新失败测试，再实现、跑绿灯，完成后审阅差异；只提交自己新建/本任务明确修改的内容，Git 所有权例外仅用单条命令的精确 safe.directory，不改全局配置。
- [ ] 若新版本许可、依赖或单文件兼容性受阻，保留原版，报告实际阻碍，不宣称已交付“单文件可用”。

## 计划自审

- 覆盖规格全部功能：固定目录（1）、快照和中断（2/3/5）、历史选择/删除/导出（4/5）、不阻塞实时（3）、单文件与迁移路径（6）、数据保护和回退（全部任务）。
- 后续任务只使用上文定义的方法和字段；时间间隔 5.0 秒、schema/report 版本 1、中文提示及数据目录保持一致。
- 五项 Review Focus 均有归属任务及具体测试；不加入云、媒体录制、平台适配或评分修改。
- 当前只有设计和计划文档，尚未运行新的功能测试、修改程序或生成新的 EXE。
