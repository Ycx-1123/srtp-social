import sys
import re
import signal
import cv2
import numpy as np
import threading
import queue
import time
from typing import Tuple, Dict
import gradio as gr
from ultralytics import YOLO
import os

# ==========================================
# 📦 音频模块导入 + 全局配置
# ==========================================
try:
    from soci_funasr_V3 import (
        start_sensing, stop_sensing, GLOBAL_BIAS_WORDS,
        SHOUTING_THRESHOLD, FAST_SPEECH_CPS
    )
except ImportError as e:
    print(f"❌ 导入音频模块失败：{e}")
    sys.exit(1)

# 视频模型配置
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制CPU
MODEL_PATH = r"D:\desktop_3\SRTP\video_project\runs\classify\runs\cls\yolov8n_cls_train\weights\best.pt"

# 全局控制
IS_RUNNING = False
audio_queue = queue.Queue(maxsize=1)  # 音频结果队列
last_video_frame = None  # 缓存上一帧
# 人脸信息缓存（确保框持续显示）
face_cache = {
    "x": 0, "y": 0, "w": 0, "h": 0,
    "emotion": "neutral", "confidence": 0.0,
    "color": (0, 255, 255)  # 默认neutral黄色
}

# ==========================================
# 🧠 社交信号分析核心逻辑
# ==========================================
def detect_bias_content(text: str) -> Tuple[bool, list]:
    """检测文本中的偏见词汇"""
    matched_biases = []
    for bias_word in GLOBAL_BIAS_WORDS:
        if re.search(re.escape(bias_word), text, re.IGNORECASE):
            matched_biases.append(bias_word)
    return len(matched_biases) > 0, matched_biases

def analyze_social_signals(
    text: str, 
    metrics: Dict[str, str], 
    peak_volume: int
) -> Tuple[str, str, str]:
    """分析社交信号：情绪/互动状态/风险等级"""
    # 解析语速和音量（兼容字符串/数字）
    try:
        cps = float(metrics["语速"].replace("字/秒", "") if isinstance(metrics["语速"], str) else metrics["语速"])
        volume = int(peak_volume)
    except (ValueError, KeyError):
        cps = 0.0
        volume = 0
    
    # 情绪判断（基于音量+语速）
    if volume > SHOUTING_THRESHOLD:
        emotion = "🔴 愤怒/激动"
    elif cps > FAST_SPEECH_CPS * 1.2:
        emotion = "🟡 紧张/急促"
    elif cps < FAST_SPEECH_CPS * 0.8:
        emotion = "🟢 放松/缓慢"
    else:
        emotion = "🟢 正常"
    
    # 互动状态
    has_bias, _ = detect_bias_content(text)
    if has_bias:
        interaction_state = "⚠️ 含偏见表述"
    elif len(text) < 2:
        interaction_state = "🟡 有效内容不足"
    else:
        interaction_state = "✅ 正常交流"
    
    # 风险等级
    if volume > SHOUTING_THRESHOLD and has_bias:
        risk_level = "🔴 高风险"
    elif has_bias or volume > SHOUTING_THRESHOLD or cps > FAST_SPEECH_CPS * 1.5:
        risk_level = "🟡 中风险"
    else:
        risk_level = "🟢 低风险"
    
    return emotion, interaction_state, risk_level

# ==========================================
# 🤖 AI建议生成器（融合视频+音频信息）
# ==========================================
def generate_ai_suggestion(facial_emotion: str, audio_emotion: str, speed: float, volume: int, interaction_state: str) -> str:
    """融合面部表情和音频特征，生成AI建议"""
    suggestions = []
    
    # 1. 基于面部表情的建议
    if facial_emotion == "happy":
        suggestions.append("😊 你看起来心情不错，保持这种积极的状态！")
    elif facial_emotion == "neutral":
        suggestions.append("😐 表情比较平静，这有助于你清晰地表达观点。")
    elif facial_emotion == "surprise":
        suggestions.append("😮 你似乎对当前话题感到意外，可以尝试更深入地探索。")
    elif facial_emotion in ["sad", "angry", "fear"]:
        suggestions.append(f"😔 检测到你可能有些{facial_emotion}，注意调节情绪，深呼吸。")
    
    # 2. 基于音频特征的建议
    if speed > FAST_SPEECH_CPS * 1.2:
        suggestions.append("🗣️ 语速稍快，适当放缓节奏，能让听众更好地跟上你的思路。")
    elif speed < FAST_SPEECH_CPS * 0.8:
        suggestions.append("🐢 语速偏慢，这显得很沉稳，但可以适当加快以增强表达力。")
    
    if volume > SHOUTING_THRESHOLD:
        suggestions.append("🔊 音量偏高，注意控制，避免给对方造成压迫感。")
    elif volume < 20:
        suggestions.append("🔈 音量偏低，适当提高音量，确保对方能清晰听到。")
    
    # 3. 基于互动状态的建议
    if "偏见" in interaction_state:
        suggestions.append("⚠️ 注意措辞，避免使用可能引起误解的表述。")
    elif "内容不足" in interaction_state:
        suggestions.append("💡 可以尝试丰富你的表达，让沟通更有深度。")
    
    # 4. 融合多模态的综合建议
    if facial_emotion in ["sad", "angry", "fear"] and audio_emotion != "🟢 正常":
        suggestions.append("🔄 你的情绪和语音状态都显示出一些压力，建议暂停一下，整理思绪。")
    elif facial_emotion == "happy" and audio_emotion == "🟢 放松/缓慢":
        suggestions.append("🌟 状态非常好！自信且从容，继续保持。")
    
    # 如果没有具体建议，返回默认鼓励
    if not suggestions:
        return "✅ 你的表达状态良好，继续保持自然的沟通方式。"
    
    # 将所有建议拼接成一段通顺的话
    return " ".join(suggestions)

# ==========================================
# 🎥 视频情绪识别线程（核心修复：稳定显示人脸框）
# ==========================================
def video_emotion_thread():
    """视频情绪识别：人脸框持续显示，仅更新必要内容"""
    global last_video_frame, face_cache
    # 初始化模型
    try:
        video_model = YOLO(MODEL_PATH)
        video_model.to('cpu')
        video_model.eval()
        FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        print(f"❌ 视频模型初始化失败：{e}")
        return
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    empty_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    last_video_frame = empty_frame
    
    # 颜色映射表（固定）
    emotion_colors = {
        "happy": (0, 255, 0),      # 绿色
        "neutral": (0, 255, 255),  # 黄色
        "surprise": (255, 0, 0),   # 蓝色
        "sad": (0, 0, 255),        # 红色
        "angry": (0, 0, 255),      # 红色
        "fear": (0, 0, 255)        # 红色
    }
    
    while IS_RUNNING:
        ret, frame = cap.read()
        if not ret:
            frame = empty_frame
        else:
            # 镜像翻转
            frame = cv2.flip(frame, 1)
            # 人脸检测（每帧都检测，确保框不丢失）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # 取第一个检测到的人脸
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                
                # 每3帧更新一次情绪和置信度（降低计算量）
                if int(time.time() * 10) % 3 == 0:
                    results = video_model(face_roi, device='cpu', verbose=False, imgsz=224)
                    emotion = results[0].names[results[0].probs.top1]
                    confidence = results[0].probs.top1conf.item()
                    
                    # 更新缓存
                    face_cache["x"] = x
                    face_cache["y"] = y
                    face_cache["w"] = w
                    face_cache["h"] = h
                    face_cache["emotion"] = emotion
                    face_cache["confidence"] = confidence
                    face_cache["color"] = emotion_colors.get(emotion, (0, 255, 255))
            
            # 核心：无论是否检测到新人脸，都绘制缓存中的框（确保框持续显示）
            if face_cache["w"] > 0 and face_cache["h"] > 0:
                # 绘制矩形框（持续显示，不闪烁）
                cv2.rectangle(
                    frame,
                    (face_cache["x"], face_cache["y"]),
                    (face_cache["x"] + face_cache["w"], face_cache["y"] + face_cache["h"]),
                    face_cache["color"],
                    2  # 线宽固定，避免粗细变化
                )
                # 绘制情绪+置信度文字（仅更新数字，位置固定）
                text = f"{face_cache['emotion']} ({face_cache['confidence']:.2f})"
                cv2.putText(
                    frame,
                    text,
                    (face_cache["x"], face_cache["y"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    face_cache["color"],
                    1  # 文字线宽固定
                )
        
        # 缓存帧数据
        last_video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        time.sleep(0.033)  # 30fps稳定帧率
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# 🎙️ 音频分析线程
# ==========================================
def audio_analysis_thread():
    """音频社交信号分析"""
    stop_sensing()  # 先停止旧的监听
    time.sleep(0.1)
    try:
        for pure_text, tags, metrics, _, peak_volume, alert_msg in start_sensing():
            if not IS_RUNNING:
                break
            
            # 基础兜底
            pure_text = pure_text or ""
            metrics = metrics or {"语速": "0.0字/秒", "音量": "0"}
            peak_volume = peak_volume or 0
            
            # 完整分析
            emotion, interaction_state, risk_level = analyze_social_signals(
                pure_text, metrics, peak_volume
            )
            has_bias, matched_biases = detect_bias_content(pure_text)
            
            # 封装结果
            audio_result = {
                "text": pure_text,
                "tags": tags or "无",
                "speed": metrics["语速"],
                "volume": str(peak_volume),
                "emotion": emotion,
                "bias": f"发现[{','.join(matched_biases)}]" if has_bias else "无",
                "interaction": interaction_state,
                "risk": risk_level,
                "alert": alert_msg or "正常"
            }
            
            # 放入队列
            if not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
            audio_queue.put(audio_result)
    
    except Exception as e:
        print(f"❌ 音频分析异常：{e}")
    finally:
        stop_sensing()

# ==========================================
# 🔄 主生成器：修复数据解析 + 稳定输出
# ==========================================
def sync_generator():
    """同步输出视频帧和音频分析结果，控制更新频率"""
    global IS_RUNNING, last_video_frame, face_cache
    IS_RUNNING = True
    
    # 启动视频和音频线程
    video_thread = threading.Thread(target=video_emotion_thread, daemon=True)
    audio_thread = threading.Thread(target=audio_analysis_thread, daemon=True)
    video_thread.start()
    audio_thread.start()
    
    # 默认结果
    default_audio = {
        "text": "", "tags": "无", "speed": "0.0字/秒", "volume": "0",
        "emotion": "🟢 正常", "bias": "无", "interaction": "✅ 正常交流",
        "risk": "🟢 低风险", "alert": "等待启动..."
    }
    default_video = np.zeros((240, 320, 3), dtype=np.uint8)
    last_video_frame = default_video
    last_audio_result = default_audio
    
    # 控制更新频率：每0.1秒更新一次
    update_interval = 0.1
    last_update_time = time.time()
    
    while IS_RUNNING:
        current_time = time.time()
        if current_time - last_update_time < update_interval:
            time.sleep(0.01)
            continue
        
        # 获取视频帧
        video_frame = last_video_frame if last_video_frame is not None else default_video
        
        # 获取音频结果
        try:
            audio_result = audio_queue.get_nowait()
            last_audio_result = audio_result
        except queue.Empty:
            audio_result = last_audio_result
        
        # 解析音频数据（容错处理）
        facial_emotion = face_cache["emotion"] if face_cache["emotion"] else "neutral"
        audio_emotion = audio_result["emotion"]
        
        # 解析语速（纯数字）
        try:
            speed_str = audio_result["speed"].replace("字/秒", "")
            speed = float(speed_str)
        except:
            speed = 0.0
        
        # 解析音量（纯数字）
        volume_str = audio_result["volume"]
        volume_digits = ''.join([c for c in volume_str if c.isdigit() or c == '.'])
        try:
            volume = int(float(volume_digits)) if volume_digits else 0
        except:
            volume = 0
        
        interaction_state = audio_result["interaction"]
        
        # 生成AI建议
        ai_suggestion = generate_ai_suggestion(facial_emotion, audio_emotion, speed, volume, interaction_state)
        
        # 输出所有数据
        yield (
            video_frame,
            audio_result["text"],
            audio_result["speed"],
            audio_result["volume"],
            audio_result["emotion"],
            audio_result["bias"],
            audio_result["interaction"],
            audio_result["risk"],
            ai_suggestion
        )
        
        last_update_time = current_time

# ==========================================
# 🛑 停止所有线程
# ==========================================
def stop_all():
    global IS_RUNNING, last_video_frame, face_cache
    IS_RUNNING = False
    stop_sensing()
    last_video_frame = None
    # 重置人脸缓存
    face_cache = {
        "x": 0, "y": 0, "w": 0, "h": 0,
        "emotion": "neutral", "confidence": 0.0,
        "color": (0, 255, 255)
    }
    # 返回空帧和默认文本
    empty_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    return (
        empty_frame, "", "0.0字/秒", "0",
        "🟢 正常", "无", "✅ 正常交流", "🟢 低风险", "系统已停止，感谢使用"
    )

# ==========================================
# 🌐 Gradio 可视化界面
# ==========================================
def create_gradio_ui():
    """创建Gradio界面"""
    with gr.Blocks(title="SOCI-AI 社交辅助系统") as demo:
        gr.Markdown("# 🎙️ SOCI-AI 社交辅助系统")
        gr.Markdown("### 视频情绪识别 + 音频社交信号分析（同步可视化）")
        
        # 视频+音频布局
        with gr.Row():
            # 视频区域
            video_out = gr.Image(
                label="情绪监测画面（颜色+概率）",
                type="numpy",
                streaming=True
            )
            
            # 音频分析区域
            with gr.Column():
                text_out = gr.Textbox(label="识别文本", lines=2)
                with gr.Row():
                    speed_out = gr.Textbox(label="语速（字/秒）")
                    volume_out = gr.Textbox(label="峰值音量")
                emotion_out = gr.Textbox(label="音频情绪")
                bias_out = gr.Textbox(label="偏见检测")
                interaction_out = gr.Textbox(label="互动状态")
                risk_out = gr.Textbox(label="风险等级")
                alert_out = gr.Textbox(label="AI建议")
        
        # 控制按钮
        with gr.Row():
            start_btn = gr.Button("▶️ 开启全方位监听")
            stop_btn = gr.Button("🛑 停止监听")
        
        # 绑定事件
        start_btn.click(
            fn=sync_generator,
            outputs=[
                video_out, text_out, speed_out, volume_out,
                emotion_out, bias_out, interaction_out, risk_out, alert_out
            ]
        )
        stop_btn.click(
            fn=stop_all,
            outputs=[
                video_out, text_out, speed_out, volume_out,
                emotion_out, bias_out, interaction_out, risk_out, alert_out
            ]
        )
    
    return demo

# ==========================================
# 🚀 程序入口
# ==========================================
if __name__ == "__main__":
    # 注册信号处理器（Ctrl+C停止）
    def signal_handler(sig, frame):
        global IS_RUNNING
        IS_RUNNING = False
        stop_sensing()
        print("\n✅ 程序已停止")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    # 启动Gradio界面（端口7861）
    try:
        demo = create_gradio_ui()
        demo.queue().launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ 程序启动失败：{e}")
        IS_RUNNING = False
        stop_sensing()
        sys.exit(1)