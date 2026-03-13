import whisper
import numpy as np
import sounddevice as sd
import torch
import time
import os

# ==========================================
# 【新增：动态加载外部词表】
# ==========================================
def load_bias_vocabulary(filepath="bias_words.txt"):
    # 获取当前脚本所在目录，确保路径绝对正确
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, filepath)
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            # 逐行读取，去除空格，并过滤掉空行
            words = [line.strip() for line in f if line.strip()]
        print(f"✅ 成功加载社交偏差词表，共包含 {len(words)} 个触发词！")
        return words
    except FileNotFoundError:
        print("⚠️ 未找到 bias_words.txt，已启用后备默认词库。")
        return ["偏见", "女生不适合", "书呆子"]

# 在启动时，把词表加载到内存中（全局变量，随时调用）
GLOBAL_BIAS_WORDS = load_bias_vocabulary()


print("--- SOCI-AI 语音模块 V3：全环境极速融合版 ---")

# 硬件自适应启动
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[1/3] 正在载入硬件加速引擎... (当前设备: {device})")

try:
    # 使用 base 模型兼顾准确率与加载速度
    model = whisper.load_model("base", device=device)
    print("✅ AI 大脑引擎启动成功！")
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

    # ==========================================
    # 【终极融合 1：防底噪核爆的第一道防线 (VAD)】
    # ==========================================
    volume_norm = np.linalg.norm(audio_data) * 10
    if volume_norm < 1.0:  
        print(f"💤 [静音跳过] 音量过小 ({volume_norm:.2f})，未检测到有效人声。")
        return 

    # ==========================================
    # 【终极融合 2：安全触发音频归一化 (清洗信号)】
    # ==========================================
    # 只有在确认有声音后，才把波形拉满，喂给 AI 最干净的音频
    max_amp = np.max(np.abs(audio_data))
    if max_amp > 0:
        audio_data = audio_data / max_amp

    print(">>> 正在进行神经元转换（GPU 加速中）...")
    start_time = time.time()

    # ==========================================
    # 【终极融合 3：锁死极速与防幻觉组合拳】
    # ==========================================
    result = model.transcribe(
        audio_data, 
        language='zh', 
        fp16=torch.cuda.is_available(), 
        condition_on_previous_text=False,
        temperature=0.0,             # [核心] 锁死重试机制！保证在嘈杂环境下也能 1 秒出结果！
        no_speech_threshold=0.8,     # [防幻觉] 要求模型有 80% 把握确定是人声
        logprob_threshold=-1.0,      # [防幻觉] 丢弃低信心的乱码猜测
        initial_prompt="以下是一段中文普通话对话：" # [语言锚定] 强制聚焦简体中文
    )
    
    end_time = time.time()
    latency = (end_time - start_time) * 1000
    
    text = result['text'].strip()

    # ==========================================
    # 【终极融合 4：物理级幻觉黑名单】
    # ==========================================
    hallucination_blacklist = ["在那裡", "字幕", "提供", "感谢", "收看", "未经允许", "请不吝赐教"]
    for bad_word in hallucination_blacklist:
        if bad_word in text and len(text) < 15: 
            text = "" 
            break
    
    if not text:
        print("⚠️ [已过滤无意义噪音，继续监听...]")
        return

    print("-" * 30)
    print(f"【识别文字】：{text}")
    print(f"【音量大小】：{volume_norm:.2f}")
    print(f"【实时耗时】：{latency:.2f} ms") 
    print("-" * 30)

    # 社交偏差检测闭环
    # ==========================================
    # 【升级：精准多词汇匹配与警报】
    # ==========================================
    detected_biases = []
    for word in GLOBAL_BIAS_WORDS:
        if word in text:
            detected_biases.append(word)
            
    if detected_biases:
        # 如果检测到任何一个词，拉响最高级警报
        print("\n" + "!" * 40)
        print(f"🚨 红色警报：系统捕捉到潜在的社交偏差！")
        print(f"🚨 触发标签：{', '.join(detected_biases)}")
        print("!" * 40 + "\n")
            
if __name__ == "__main__":
    while True:
        try:
            start_sensing()
        except KeyboardInterrupt:
            print("\n[系统退出] 测试已安全终止。")
            break