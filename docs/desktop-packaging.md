# 原生 Windows 桌面版打包与历史

本次分层提示版交付在 `dist/SOCI-AI-LayeredGuidance/SOCI-AI.exe`，只发送这一个 EXE 即可，旧会话总结版 `dist/SOCI-AI-SavedReview/SOCI-AI.exe` 保留不覆盖。通用打包脚本的默认输出仍是 `dist/SOCI-AI-Onefile/SOCI-AI.exe`。Python、Qt、模型、语言样例库及说明封装在程序内，运行时自动解压到临时目录；不需要目标电脑安装 Python、浏览器或项目源码。单文件封装不是加密或防逆向措施。目标为现代 Windows 10/11 x64，具体驱动与权限需现场验证。

## 构建

在 Windows x64 Python 3.13 环境安装 `requirements-desktop.txt`。模型必须预先放在 `models`，脚本检查完整性，不下载模型。

```powershell
& 'D:\minicoda3\python.exe' build_desktop.py --mode onefile --check-only
& 'D:\minicoda3\python.exe' build_desktop.py --mode onefile --print-command
& 'D:\minicoda3\python.exe' build_desktop.py --mode onefile --clean
```

单文件缓存位于 `build/native-onefile`。仅修改源码时可省略 `--clean`。兼容的 `--mode onedir` 输出 `dist/SOCI-AI-Desktop`，该旧模式仍需要完整文件夹。两个构建目录相互独立，不会覆盖 SRTP 下已经交付的旧安装目录，也不清空整个 `build` 或 `dist`。重建同名 EXE 前先关闭它。

构建子进程限制 DLL 搜索路径为 Python 和 Windows 系统目录，避免混入其他工具的同名 DLL。收集 Qt 平台插件、MediaPipe 原生 DLL、Sherpa ONNX 与 PortAudio；排除 QtWebEngine、Torch、TensorFlow、FunASR、Ultralytics 和旧网页服务等无关大型依赖。MediaPipe 视觉模块使用 Matplotlib，因此保留此依赖。

内置资源包含 `face_landmarker.task`、中文流式识别的 `model.int8.onnx`、词表、官方测试录音、语义规则、300 条原创风险样例 + 60 条对照的 JSON/SQLite 数据库、来源说明、使用说明和第三方声明。中文语音模型来自 [Sherpa 官方发布](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01.tar.bz2)。这些样例并非已验证的训练集，打包成功不证明算法准确率。

## 数据与生命周期

正常运行的可写数据固定在当前 Windows 账户的 `%LOCALAPPDATA%/SOCI-AI-Desktop`：

- `history.sqlite3`：会话历史。
- `logs/desktop.log`：诊断日志。
- `exports`：默认报告导出目录。

只读资源使用 `sys._MEIPASS`；历史不放在该临时目录、当前工作目录或 EXE 旁。移动 EXE、替换版本不影响同一 Windows 账户的历史；不同电脑和账户不共享记录。没有云同步、账户识别或数据库加密。

SQLite 连接只由专用后台线程持有。正常桌面会话在结束后由用户选择“保存到历史”，再提交完整快照；未选择保存时只保留在本次窗口内，关窗或下一次检测前提醒保存、不保存或返回。突然退出前尚未选择保存的会话无法恢复。当前会话的派生记录保留整场曲线、文字与事件，不因超过一小时或条数上限截掉开头；原始音视频处理队列仍保持有界。只记录确认转录、分数、时间、线索、事件及建议，不保存原始音视频、照片或临时字幕。异常重启后将已经写入但未完成的记录标为异常结束，保留已落盘内容。写入错误可重试；删除仅作用于用户确认的单条已结束记录。导出支持独立的 HTML 可读报告或 JSON 数据；重开应用直接恢复图表与总结，不需要导入报告文件。

## 隔离验收

构建后默认做真实模型加载、官方文件录音识别及 offscreen 窗口自检，输出在新建隔离测试目录，不开摄像头/麦克风，不接触真实用户历史。最终交付还应执行：

```powershell
& 'D:\minicoda3\python.exe' tools/check_onefile_history.py `
  --exe 'D:\desktop_3\SRTP\SOCI-AI-Studio\dist\SOCI-AI-LayeredGuidance\SOCI-AI.exe' `
  --output 'D:\desktop_3\SRTP\SOCI-AI-Studio\artifacts\layered-guidance-qa'
```

验收把 EXE 复制到只含该文件的中文路径，在独立工作目录与仅 Windows 系统 PATH 下启动，不使用源码或已安装模型。分别验证模型加载、GUI、跨进程历史写入/读取、单条删除及再次读取；然后移动 EXE 并验证历史仍在。报告记录实际大小、SHA256 和每次耗时。它不能代替另一台电脑的实机或现场设备测试。

手动运行 `--self-test-models`、`--smoke-test`、`--self-test-history` 前，必须将 `SOCI_AI_TEST_DATA_DIR` 设为独立测试目录的绝对路径。正常启动忽略该变量。`--skip-smoke` 仅用于构建调试，不意味着已经验收。

## 第三方许可与发布边界

打包脚本从本机依赖元数据收集许可文本，并附 Qt 官方 LGPL/GPL 文本与 Sherpa ONNX 的 Apache 许可，生成 `resources/licenses/THIRD-PARTY-NOTICES.txt`。声明和使用说明嵌入 EXE，可在“系统介绍”内查看。模型不是本项目自行训练，代码许可也不能自动推定为所有模型权重的许可。

Qt 开源版本的分发还涉及告知、对应库源码获取、替换或重新链接等义务，详见 [Qt 官方 LGPL 义务](https://www.qt.io/development/open-source-lgpl-obligations)。PyInstaller 资源归档可提取，不等于已提供完整合规材料。公开或商业发布前仍需核对实际依赖和模型权重授权、提供所需材料；本次技术原型构建不是法律合规结论。

资源路径依据 [PyInstaller 运行时路径说明](https://pyinstaller.org/en/stable/runtime-information.html)。
