import pyaudio
import wave
import numpy as np
import time
import os
from funasr import AutoModel

print("--- SOCI-AI 语音模块 V5.0：PyAudio 工业流式斩龙版 ---")

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

# 1. 载入模型
print("[1/3] 正在载入 FunASR 核心引擎...")
cache_path = r"C:\Users\杨雨馨\.cache\modelscope\hub\damo"
try:
    model = AutoModel(
        model=f"{cache_path}\\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model=f"{cache_path}\\speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model=f"{cache_path}\\punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        device="cuda:0",
        disable_update=True
    )
    print("✅ 引擎载入成功！")
except Exception as e:
    print(f"❌ 引擎载入失败: {e}")
    exit()

def start_sensing():
    # PyAudio 核心参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "temp_soci_audio.wav"

    p = pyaudio.PyAudio()

    print("\n>>> 正在监听社交信号，请说话...")
    # 打开音频流
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    silent_chunks = 0
    speaking = False
    
    # 【核心升级：动态断句雷达】
    # 只要开始说话，就一直录，直到对方停顿 1 秒钟，或者最多录满 10 秒强制切断
    MAX_RECORD_SECONDS = 10
    SILENCE_THRESHOLD = 500  # 音量低于这个值认为是静音
    SILENCE_LIMIT_CHUNKS = int(RATE / CHUNK * 1.0) # 容忍 1.0 秒的停顿
    MAX_CHUNKS = int(RATE / CHUNK * MAX_RECORD_SECONDS)

    for i in range(MAX_CHUNKS):
        data = stream.read(CHUNK)
        frames.append(data)
        
        # 将二进制音频转为数字以计算音量
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        volume = np.max(np.abs(audio_chunk))
        
        if volume > SILENCE_THRESHOLD:
            speaking = True
            silent_chunks = 0 # 一旦有声音，重置静音计数器
        elif speaking:
            silent_chunks += 1
            
        # 如果已经开始说话，且停顿超过了 1 秒，认为本句话结束，立刻跳出录音！
        if speaking and silent_chunks > SILENCE_LIMIT_CHUNKS:
            print("💡 [检测到语句停顿，瞬间截取]")
            break

    # ==========================================
    # 【核心防御】：录完立刻物理掐断麦克风，归还 Windows 句柄！
    # ==========================================
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 瞬间写成干净的文件桥
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # 静音检测 (简单粗暴的振幅判定)
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    if np.max(np.abs(audio_data)) < 500:  
        print("💤 [环境静音，已过滤]")
        return

    print(">>> 正在进行神经元转换...")
    start_time = time.time()

    try:
        # 直接让模型读取这个被彻底释放的物理文件
        res = model.generate(input=WAVE_OUTPUT_FILENAME, is_final=True)
        text = res[0]['text'].strip() if res and len(res) > 0 and 'text' in res[0] else ""
    except Exception as e:
        print(f"❌ 推理中断: {e}")
        return
    finally:
        # 阅后即焚
        if os.path.exists(WAVE_OUTPUT_FILENAME):
            os.remove(WAVE_OUTPUT_FILENAME)
    
    latency = (time.time() - start_time) * 1000
    
    if text:
        print("-" * 30)
        print(f"【捕捉内容】：{text}")
        print(f"【系统延迟】：{latency:.2f} ms") 
        print("-" * 30)

        # ==========================================
        # 【新增：社交压迫感（语速）检测】
        # ==========================================
        # 刨去标点符号，计算纯汉字数量
        pure_text_len = len([char for char in text if char.isalnum()])
        # 我们录音固定是 5 秒
        cps = pure_text_len / 5.0  
        
        print(f"【实时语速】：{cps:.1f} 字/秒")
        if cps > 4.5:
            print("⚠️ 情绪提醒：您的语速偏快，可能存在社交紧张或给对方造成压迫感，请深呼吸放缓节奏。")
        elif cps < 1.0 and pure_text_len > 0:
            print("💡 社交提醒：您的语速较慢，可以适当提高音量和节奏哦。")

        # 偏差词检测
        detected = [w for w in GLOBAL_BIAS_WORDS if w in text]
        if detected:
            print(f"🚨 警报：检测到社交偏差词汇 {detected}")
    else:
        print("⚠️ [未提取到有效语义]")

if __name__ == "__main__":
    while True:
        try:
            start_sensing()
        except KeyboardInterrupt:
            print("\n[系统退出] SOCI-AI 模块已安全关闭。")
            break