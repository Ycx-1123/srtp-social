import pyaudio
import wave
import numpy as np
import time
import os
import re
from funasr import AutoModel

print("--- SOCI-AI 语音模块 V6.0：SenseVoice 情感多模态版 ---")

# 0. 加载偏差词库
def load_bias_vocabulary(filepath="bias_words.txt"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, filepath)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return ["偏见", "女生不适合", "书呆子"]

GLOBAL_BIAS_WORDS = load_bias_vocabulary()

# 1. 载入 SenseVoice 多模态引擎 (纯本地直连模式)
print("[1/3] 正在从本地纯英文特区载入 SenseVoiceSmall...")
try:
    model = AutoModel(
        model=r"D:\SOCI-AI\models\SenseVoiceSmall", # 【核心修改】：直接指向你搬家后的纯英文路径！
        trust_remote_code=True,
        device="cuda:0",
        disable_update=True # 【核心修改】：彻底断开网络，开启绝对物理离线模式！
    )
    print("✅ 引擎载入成功！RTX 4050 情感分析节点已就绪。")
except Exception as e:
    print(f"❌ 引擎载入失败: {e}")
    exit()

def start_sensing():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    WAVE_OUTPUT_FILENAME = "temp_soci_audio.wav"

    p = pyaudio.PyAudio()

    print("\n>>> 正在监听社交信号 (动态捕捉中)，请说话...")
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []               # 存储音频数据块的列表
    silent_chunks = 0         # 静音块计数器（检测停顿）
    speaking = False          # 说话状态标志（True=正在说话）
    
    # 动态断句雷达参数
    MAX_RECORD_SECONDS = 15 #最长单次录音时间：最大录音：15秒
    SILENCE_THRESHOLD = 500  
    SILENCE_LIMIT_CHUNKS = int(RATE / CHUNK * 2.0) # 容忍 2.0 秒停顿
    MAX_CHUNKS = int(RATE / CHUNK * MAX_RECORD_SECONDS)

    for i in range(MAX_CHUNKS):
        data = stream.read(CHUNK)
        frames.append(data)
        
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        volume = np.max(np.abs(audio_chunk))
        
        if volume > SILENCE_THRESHOLD:
            speaking = True
            silent_chunks = 0 
        elif speaking:
            silent_chunks += 1
            
        if speaking and silent_chunks > SILENCE_LIMIT_CHUNKS:
            print("💡 [检测到语句停顿，瞬间截取]")
            break

    # 物理归还麦克风句柄
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 静音过滤
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    if np.max(np.abs(audio_data)) < SILENCE_THRESHOLD:  
        print("💤 [环境静音，已过滤]")
        return

    # 生成临时文件桥
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(">>> 正在进行情感与语义双重解析...")
    start_time = time.time()

    try:
        # SenseVoice 推理
        res = model.generate(input=WAVE_OUTPUT_FILENAME, is_final=True)
        raw_text = res[0]['text'].strip() if res and len(res) > 0 and 'text' in res[0] else ""
    except Exception as e:
        print(f"❌ 推理中断: {e}")
        return
    finally:
        if os.path.exists(WAVE_OUTPUT_FILENAME):
            os.remove(WAVE_OUTPUT_FILENAME) # 删除临时文件，保护隐私
    
    latency = (time.time() - start_time) * 1000
    
    if raw_text:
        # ==========================================
        # 【核心解析：剥离情绪与事件标签】
        # ==========================================
        tags = re.findall(r'<\|.*?\|>', raw_text)
        pure_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()

        # ==========================================
        # 1. 动态计算语速 (Words Per Second)
        # ==========================================
        actual_seconds = (len(frames) * 1024) / 16000.0 
        pure_text_len = len([char for char in pure_text if char.isalnum()])
        cps = pure_text_len / actual_seconds if actual_seconds > 0 else 0

        # ==========================================
        # 2. 提取声学压迫感 (Volume Peak)
        # ==========================================
        # int16 格式的最大振幅是 32767，我们设定 18000 为“突然大声/吼叫”的阈值
        peak_volume = np.max(np.abs(audio_data))
        is_shouting = peak_volume > 18000 

        # ==========================================
        # 3. 解析微小攻击事件 (Microaggressions)
        # ==========================================
        # SenseVoice 原生支持识别叹气和笑声
        is_sigh = any(tag in tags for tag in ["<|Sigh|>", "<|sigh|>"])
        is_sneer = any(tag in tags for tag in ["<|Laughter|>", "<|Laugh|>", "<|laugh|>"])
        is_angry = "<|ANGRY|>" in tags

        # ==========================================
        # 4. 社交偏差词交叉比对
        # ==========================================
        detected = [w for w in GLOBAL_BIAS_WORDS if pure_text and w in pure_text]

        print("-" * 50)
        print(f"【捕捉内容】：{pure_text if pure_text else '(无有效文字)'}")
        print(f"【多模态标签】：{' '.join(tags)}")
        print(f"【物理指标】：语速 {cps:.1f} 字/秒 | 峰值音量 {peak_volume}")
        print(f"【系统延迟】：{latency:.2f} ms")
        print("-" * 50)

        # 🎯 综合诊断引擎（按危险等级从高到低输出）
        
        # 危险等级 1：直接的语言偏见
        if detected:
            print("\n" + "!" * 40)
            print(f"🚨 红色警报：捕捉到潜在社交偏差词汇！")
            print(f"🚨 触发标签：{', '.join(detected)}")
            if is_angry or is_shouting:
                print("💥 攻击性叠加：伴随高压音量或愤怒情绪，此言论具有极强杀伤力！")
            print("!" * 40 + "\n")
            
        # 危险等级 2：非语言的冷暴力/微小攻击
        elif is_sigh or is_sneer:
            print("\n" + "⚠️ 橙色警报：捕捉到非语言维度的【微小攻击】！")
            if is_sigh:
                print("👉 分析：重重叹气。可能传达了不耐烦、轻视或拒绝沟通的防御姿态。")
            if is_sneer:
                print("👉 分析：冷笑/嘲笑。极易破坏社交安全感，请注意态度。")
            print("⚠️" * 20 + "\n")

        # 危险等级 3：高压迫感的声学与情绪特征
        elif is_angry or is_shouting or cps > 5.0:
            print("\n💡 社交压迫感提醒：未检测到敏感词，但您的表达方式存在隐患：")
            if is_angry:
                print("   - 情绪雷达：您的语调听起来处于【愤怒】状态。")
            if is_shouting:
                print("   - 声学雷达：音量异常飙升，容易给对方造成生理性压迫感。")
            if cps > 5.0:
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