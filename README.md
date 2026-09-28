# SOCI AI Studio

面向细微社交偏差的本地多模态自检原型。程序通过电脑摄像头和麦克风观察使用者自己的表达状态，将面部动作、声音压力和中文微偏见语义线索融合为 SBI 指标，并用生命树、友好度圆环、实时建议和会话回顾展示变化。

本项目用于东南大学 SRTP 结项展示与原型研究。它帮助使用者在面试练习、模拟答辩、团队沟通训练、会议复盘等场景中发现可能让人不适的表达方式；它不是人格、心理、医学、招聘或纪律判断工具。

## 当前版本

- 原生 Windows 桌面应用，不需要网页或本地服务器。
- 实时摄像头预览和跟随人脸移动的面部轮廓。
- 眉部紧张、嘴角下压、声音压力和语言偏见的可解释提示。
- 内置中文社交微偏见样例库，包含 300 条风险表达和 60 条友好/否定/引用对照。
- SBI 曲线、异常原因、总结建议、历史会话保存、单条删除和 HTML/JSON 报告导出。
- 处理过程默认在本机完成，不保存原始视频、照片、音频或临时字幕。

## 快速运行源码

推荐环境：Windows 10/11 x64，Python 3.13。

```powershell
git clone https://github.com/Ycx-1123/srtp-social.git
cd srtp-social
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-desktop.txt
python desktop_launcher.py
```

也可以在安装依赖后双击 `start_desktop.bat`。如果电脑上有多个 Python，可以先设置：

```powershell
$env:SOCI_PYTHON="C:\Path\To\Python.exe"
.\start_desktop.bat
```

首次启动会加载本地人脸和中文语音模型，可能比之后稍慢。请先关闭正在占用摄像头的会议软件或旧网页。

## 模型与数据

仓库包含运行原型所需的小型本地模型和语言样例库：

- `models/face_landmarker.task`：MediaPipe 人脸关键点模型。
- `models/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/`：sherpa-onnx 中文流式识别模型。
- `resources/language/social_bias_examples.json` 与 `.sqlite`：中文微偏见样例库。

语言库条目是结合 COLD、SWSR、CHBias 等公开研究类别后整理的原创合成样例，用于原型规则检索和答辩展示，不是经过独立验证的真实会议训练集。

## 使用方式

1. 打开程序，点击“开始检测”。
2. 保持自然表情约 1 秒，让系统记录个人基线。
3. 对着摄像头进行答辩、面试或会议表达练习。
4. 观察生命树、SBI、友好度和顶部建议。
5. 点击“结束会话”，在“会话回顾”保存、查看历史或导出报告。

更多操作说明见 [docs/desktop-quickstart.md](docs/desktop-quickstart.md)。

## 兼容旧演示模式

当前推荐使用原生桌面版。仓库仍保留旧网页实时单人自检原型，便于对照早期 SRTP 答辩材料。如需运行旧演示依赖：

```powershell
python -m pip install -r requirements-live.txt
```

旧模式采用单人自检假设：近距离主讲者就是摄像头前的本人。声纹注册与多人分离：扩展接口已设计，当前未实现。

## 重新打包 EXE

安装桌面依赖后运行：

```powershell
python build_desktop.py --mode onefile --clean
```

输出文件位于 `dist/SOCI-AI-Onefile/SOCI-AI.exe`。`dist/`、`build/`、`artifacts/` 和 `data/` 是本机构建或运行产物，不进入 GitHub 源码仓库。

## 测试

```powershell
python -W error::ResourceWarning -m unittest discover -s tests -v
```

测试不要求真实摄像头或麦克风。模型自检、窗口自检和历史自检说明见 [docs/desktop-packaging.md](docs/desktop-packaging.md)。

## 隐私与边界

程序只保存分数、时间、线索、确认转写、事件和建议。历史数据库保存在当前 Windows 用户的 `%LOCALAPPDATA%/SOCI-AI-Desktop/history.sqlite3`，不会随源码上传。系统不进行身份识别，不上传云端，不保存原始音视频。

SBI 是未校准的原型评分。单次皱眉、大音量或一句转写不能单独证明冒犯意图；界面给出的提示应作为自我练习和复盘参考。

## 第三方声明

项目使用 PySide6、MediaPipe、sherpa-onnx、OpenCV、NumPy 等开源组件和本地模型。相关声明保存在 `resources/licenses/`。公开发布、商业分发或申请软件著作权/专利前，应继续核对依赖和模型权重的许可义务。
