import pyaudio
import numpy as np
import time
import os
import re
from funasr import AutoModel

print("--- SOCI-AI 语音模块 V6.2：工业级封板内核 ---")

# ==========================================
# ⚙️ SOCI-AI 引擎雷达配置中心
# ==========================================
SOCI_FORCE_GPU = os.environ.get("SOCI_FORCE_GPU", "1") not in {"0", "false", "False"}

AUDIO_RATE = 16000                # 工业级标准采样率
AUDIO_CHUNK = 1024                # 内存切片大小
MAX_RECORD_SECONDS = 15           # 最大单次录音时长 (秒)
SILENCE_LIMIT_SECONDS = 1.5       # 断句雷达：容忍停顿时间 (秒)
SILENCE_THRESHOLD = 500           # 底噪雷达：低于此值判定为静音 (可随麦克风灵敏度调整)
SHOUTING_THRESHOLD = 18000        # 声学雷达：高于此值判定为音量过载/吼叫
FAST_SPEECH_CPS = 5.0             # 节奏雷达：语速过快阈值 (字/秒)


# ==========================================
# 🔧 核心初始化与环境探测
# ==========================================
def _resolve_model_dir() -> str:
    env_path = os.environ.get("SOCI_MODEL_DIR")
    if env_path and os.path.isdir(env_path):
        return env_path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_model_dir = os.path.abspath(os.path.join(current_dir, "..", "models", "SenseVoiceSmall"))
    if os.path.isdir(repo_model_dir):
        return repo_model_dir
    return r"D:\SOCI-AI\models\SenseVoiceSmall"

def _resolve_device() -> str:
    device = os.environ.get("SOCI_CUDA_DEVICE", "cuda:0")
    if not SOCI_FORCE_GPU:
        try:
            import torch  # type: ignore
            return device if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("已设置强制 GPU，但未能导入 torch；请安装带 CUDA 的 PyTorch。") from e
    if not torch.cuda.is_available():
        raise RuntimeError("已设置强制 GPU (SOCI_FORCE_GPU=1)，但当前环境未检测到 CUDA。")
    return device

def load_bias_vocabulary(filepath="bias_words.txt"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, filepath)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return ["偏见", "女生不适合", "书呆子"]

GLOBAL_BIAS_WORDS = load_bias_vocabulary()

print("[1/3] 正在探测显卡与模型环境...")
try:
    model_dir = _resolve_model_dir()
    device = _resolve_device()
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        device=device,
        disable_update=True
    )
    if device.startswith("cuda"):
        print(f"✅ 引擎载入成功！当前使用 {device}（显存直连已就绪）。")
    else:
        print("✅ 引擎载入成功！当前使用 CPU。")
except Exception as e:
    print(f"❌ 引擎载入失败: {e}")
    exit()

# ==========================================
# 🎙️ 多模态监听主循环
# ==========================================
def start_sensing():
    p = pyaudio.PyAudio()
    stream = None
    frames = []
    speaking = False
    try:
        print("\n>>> 正在监听社交信号 (动态捕捉中)，请说话...")
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK,
        )

        silent_chunks = 0

        SILENCE_LIMIT_CHUNKS = int(AUDIO_RATE / AUDIO_CHUNK * SILENCE_LIMIT_SECONDS)
        MAX_CHUNKS = int(AUDIO_RATE / AUDIO_CHUNK * MAX_RECORD_SECONDS)

        for _ in range(MAX_CHUNKS):
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            frames.append(data)

            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.int32, copy=False)
            volume = int(np.max(np.abs(audio_chunk)))

            if volume > SILENCE_THRESHOLD:
                speaking = True
                silent_chunks = 0
            elif speaking:
                silent_chunks += 1

            if speaking and silent_chunks > SILENCE_LIMIT_CHUNKS:
                print("💡 [检测到语句停顿，显存直连调度中...]")
                break
    except Exception as e:
        print(f"❌ 音频采集失败: {e}")
        return
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        try:
            p.terminate()
        except Exception:
            pass

    # 静音与防爆音过滤
    if not frames:
        print("⚠️ [未采集到音频数据]")
        return

    audio_data_int16 = np.frombuffer(b"".join(frames), dtype=np.int16)
    if audio_data_int16.size == 0:
        print("⚠️ [音频数据为空，已跳过]")
        return

    peak_volume = int(np.max(np.abs(audio_data_int16.astype(np.int32, copy=False))))
    if peak_volume < SILENCE_THRESHOLD:  
        print("💤 [环境静音，已过滤]")
        return

    # 显存直连 (0 磁盘开销)
    start_time = time.time()
    audio_data_float32 = np.ascontiguousarray(audio_data_int16.astype(np.float32) / 32768.0)

    try:
        res = model.generate(input=audio_data_float32, is_final=True)
        raw_text = res[0]['text'].strip() if res and len(res) > 0 and 'text' in res[0] else ""
    except Exception as e:
        print(f"❌ 推理中断: {e}")
        return
    
    latency = (time.time() - start_time) * 1000
    
    if raw_text:
        tags = re.findall(r'<\|.*?\|>', raw_text)
        pure_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()

        # 计算真实语速
        total_seconds = (len(frames) * AUDIO_CHUNK) / float(AUDIO_RATE)
        effective_seconds = max(0.5, total_seconds - SILENCE_LIMIT_SECONDS) if speaking else total_seconds
        
        pure_text_len = len([char for char in pure_text if char.isalnum()])
        cps = pure_text_len / effective_seconds 

        is_shouting = peak_volume > SHOUTING_THRESHOLD 
        is_sigh = any(tag in tags for tag in ["<|Sigh|>", "<|sigh|>"])
        is_sneer = any(tag in tags for tag in ["<|Laughter|>", "<|Laugh|>", "<|laugh|>"])
        is_angry = "<|ANGRY|>" in tags
        detected = [w for w in GLOBAL_BIAS_WORDS if pure_text and w in pure_text]

        print("-" * 50)
        print(f"【捕捉内容】：{pure_text if pure_text else '(无有效文字)'}")
        print(f"【多模态标签】：{' '.join(tags)}")
        print(f"【物理指标】：真实语速 {cps:.1f} 字/秒 | 峰值音量 {peak_volume}")
        print(f"【系统延迟】：{latency:.2f} ms")
        print("-" * 50)

        # 🎯 综合诊断引擎
        if detected:
            print("\n" + "!" * 40)
            print(f"🚨 红色警报：捕捉到潜在社交偏差词汇！")
            print(f"🚨 触发标签：{', '.join(detected)}")
            if is_angry or is_shouting:
                print("💥 攻击性叠加：伴随高压音量或愤怒情绪，此言论具有极强杀伤力！")
            print("!" * 40 + "\n")
            
        elif is_sigh or is_sneer:
            print("\n" + "⚠️ 橙色警报：捕捉到非语言维度的【微小攻击】！")
            if is_sigh:
                print("👉 分析：重重叹气。可能传达了不耐烦、轻视或拒绝沟通的防御姿态。")
            if is_sneer:
                print("👉 分析：冷笑/嘲笑。极易破坏社交安全感，请注意态度。")
            print("⚠️" * 20 + "\n")

        elif is_angry or is_shouting or cps > FAST_SPEECH_CPS:
            print("\n💡 社交压迫感提醒：未检测到敏感词，但您的表达方式存在隐患：")
            if is_angry:
                print("   - 情绪雷达：您的语调听起来处于【愤怒】状态。")
            if is_shouting:
                print("   - 声学雷达：音量异常飙升，容易给对方造成生理性压迫感。")
            if cps > FAST_SPEECH_CPS:
                print("   - 节奏雷达：语速过快，容易传递焦躁情绪。")
            print("💡 建议：深呼吸，放慢语速，降低音量。")
            
    else:
        print("⚠️ [未提取到有效语义]")

if __name__ == "__main__":
    while True:
        try:
            start_sensing()
        except KeyboardInterrupt:
            print("\n[系统退出] SOCI-AI 模块已安全关闭。")
            break