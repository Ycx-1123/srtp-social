import whisper
import numpy as np
import sounddevice as sd
import torch
import time

print("--- SOCI-AI 语音模块 V2：高精度过滤模式 ---")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[1/3] 正在载入 RTX 4050 加速引擎... (设备: {device})")

# 【修改1：将 'tiny' 升级为 'base'，准确率大幅提升，第一次运行会下载约 140MB】
try:
    model = whisper.load_model("base", device=device)
    print("✅ 引擎启动成功！")
except Exception as e:
    print(f"❌ 引擎启动失败: {e}")
    exit()

def start_sensing():
    fs = 16000
    duration = 5  
    print(f"\n>>> 正在监听社交信号，请说话...")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio_data = recording.flatten()

    # 音量检测（VAD），过滤环境底噪
    volume_norm = np.linalg.norm(audio_data) * 10
    if volume_norm < 1.0:  
        print(f"💤 [静音跳过] 音量过小 ({volume_norm:.2f})，未检测到有效人声。")
        return 

    print(">>> 正在进行神经元转换（GPU 加速中）...")
    start_time = time.time()

    # ==========================================
    # 【升级 1：防幻觉核心参数注入】
    # ==========================================
    result = model.transcribe(
        audio_data, 
        language='zh', 
        fp16=True, 
        condition_on_previous_text=False,
        no_speech_threshold=0.8,     # 强力过滤：要求模型有 80% 以上把握确定这是人声，否则丢弃 (默认 0.6)
        logprob_threshold=-1.0,      # 信心过滤：丢弃模型自己都不确定的乱码猜测
        initial_prompt="以下是一段中文普通话对话：" # 语言锚定：强行把模型的注意力拉到简体中文上，大幅减少繁体字和无意义语气词的幻觉
    )
    
    end_time = time.time()
    latency = (end_time - start_time) * 1000
    
    text = result['text'].strip()

    # ==========================================
    # 【升级 2：经典幻觉黑名单拦截】
    # ==========================================
    # Whisper 在中文环境下如果实在听不清，最爱脑补以下词汇。如果完全命中，直接物理抹除。
    hallucination_blacklist = ["在那裡", "字幕", "提供", "感谢", "收看", "未经允许", "请不吝赐教"]
    for bad_word in hallucination_blacklist:
        if bad_word in text and len(text) < 15: # 如果识别结果很短且包含黑名单词，说明是纯幻觉
            text = "" 
            break
    
    # 防止空字符串打印
    if not text:
        print("⚠️ [已过滤无意义噪音，继续监听...]")
        return

    print("-" * 30)
    print(f"【识别文字】：{text}")
    print(f"【音量大小】：{volume_norm:.2f}")
    print(f"【实时耗时】：{latency:.2f} ms") 
    print("-" * 30)

    # 社交偏差检测逻辑
    bias_words = ["偏见", "女生不适合", "某地人", "书呆子"]
    for word in bias_words:
        if word in text:
            print(f"⚠️ 警报：检测到社交偏差词汇【{word}】！")
            
if __name__ == "__main__":
    while True:
        try:
            start_sensing()
        except KeyboardInterrupt:
            break