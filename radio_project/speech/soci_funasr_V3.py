import pyaudio
import numpy as np
import time
import os
import re
from funasr import AutoModel

print("--- SOCI-AI 语音模块 V6.2：工业级封板内核 ---")

# ==========================================
# ⚙️ SOCI-AI 引擎雷达配置中心 (已适配 CPU)
# ==========================================
SOCI_FORCE_GPU = False

AUDIO_RATE = 16000                
AUDIO_CHUNK = 1024                
MAX_RECORD_SECONDS = 15           
SILENCE_LIMIT_SECONDS = 1.5       
SILENCE_THRESHOLD = 500           
SHOUTING_THRESHOLD = 18000        
FAST_SPEECH_CPS = 5.0             

# ==========================================
# 🔧 核心初始化与环境探测
# ==========================================
def _resolve_model_dir() -> str:
    # 路径已根据您的文件夹截图 (image_9bfc8e.png) 修正
    return r"D:\desktop_3\SRTP\radio_project\models\SenseVoiceSmall"

def _resolve_device() -> str:
    return "cpu"

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
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        device="cpu",
        disable_update=True
    )
    print("✅ 引擎载入成功！当前使用 CPU。")
except Exception as e:
    print(f"❌ 引擎载入失败: {e}")
    exit()

# 用于控制监听状态的全局变量
IS_LISTENING = True

def stop_sensing():
    global IS_LISTENING
    IS_LISTENING = False

# ==========================================
# 🎙️ 多模态监听生成器 (适配 Gradio)
# ==========================================
def start_sensing():
    global IS_LISTENING
    IS_LISTENING = True
    p = pyaudio.PyAudio()
    
    while IS_LISTENING:
        stream = None
        frames = []
        speaking = False
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=AUDIO_RATE, 
                           input=True, frames_per_buffer=AUDIO_CHUNK)

            silent_chunks = 0
            SILENCE_LIMIT_CHUNKS = int(AUDIO_RATE / AUDIO_CHUNK * SILENCE_LIMIT_SECONDS)

            while IS_LISTENING:
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
                    break
            
            if not frames or not speaking: continue

            audio_data_int16 = np.frombuffer(b"".join(frames), dtype=np.int16)
            peak_volume = int(np.max(np.abs(audio_data_int16.astype(np.int32, copy=False))))
            audio_data_float32 = np.ascontiguousarray(audio_data_int16.astype(np.float32) / 32768.0)

            res = model.generate(input=audio_data_float32, is_final=True)
            raw_text = res[0]['text'].strip() if res else ""
            
            if raw_text:
                tags = ' '.join(re.findall(r'<\|.*?\|>', raw_text))
                pure_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()
                
                # 计算 CPS 和诊断 (简化逻辑供 UI 演示)
                cps = len(pure_text) / (len(frames) * AUDIO_CHUNK / AUDIO_RATE)
                
                # 构造返回给 soci_final_hub.py 的数据格式
                yield pure_text, tags, {"语速": f"{cps:.1f}字/秒", "音量": peak_volume}, None, "🟢 正常", "正在分析社交信号..."
                
        except Exception as e:
            print(f"监听循环异常: {e}")
            break
        finally:
            if stream: stream.stop_stream(); stream.close()
    
    p.terminate()

if __name__ == "__main__":
    for data in start_sensing():
        print(data)