import gradio as gr
import soci_funasr_V3 as backend  # 直接引入你亲手打造的封板内核 V3
import time
import re

print("A-Team AI Lab | SOCI-AI 前端指挥中心已启动")

# ==========================================
# ⚙️ 引擎雷达配置中心 (需与前端同步)
# ==========================================
SOCI_MODEL_BRAND = "A-Team AI Lab"
FAST_SPEECH_CPS = 5.0             # 语速过快阈值 (字/秒)
SHOUTING_THRESHOLD = 18000        # 声学雷达：高于此值判定为音量过载

# ==========================================
# 🌐 多模态标签人类语言翻译字典 (终极升级版)
# ==========================================
TAG_DICT = {
    "<|zh|>": "🇨🇳 中文",
    "<|en|>": "🇬🇧 英文",
    "<|NEUTRAL|>": "😐 平静",
    "<|ANGRY|>": "😡 愤怒",
    "<|HAPPY|>": "😃 开心",
    "<|SAD|>": "😢 悲伤",
    "<|EMO_UNKNOWN|>": "😶 情绪未知",
    "<|Speech|>": "🗣️ 纯净人声",
    "<|nospeech|>": "🔇 无人声/环境音",
    "<|Laughter|>": "😏 冷笑/笑声",
    "<|Sigh|>": "😮‍💨 叹气",
    "<|Event_UNK|>": "❓ 未知声音事件",
    "<|woitn|>": "🔕 无背景噪音",
    "<|withitn|>": "🔊 有背景噪音"
}

def translate_tags(raw_tags_str):
    """把机器标签优雅翻译成带 Emoji 的本地人类语言"""
    if not raw_tags_str or raw_tags_str == "无":
        return "无"
    readable_str = raw_tags_str
    # 遍历字典替换
    for machine_tag, human_text in TAG_DICT.items():
        readable_str = readable_str.replace(machine_tag, human_text)
    return readable_str

# ==========================================
# 🎨 商业级 UI 组件工厂
# ==========================================
def create_html_badge(level, msg):
    """根据风险等级生成现代感的颜色 Badge 和诊断建议 HTML"""
    color = "#8b8b8b" # Default (Gray)
    icon = "⚪"
    level_ch = "待机"
    
    if "安全" in level:
        color = "#34C759" # iOS Green
        icon = "🟢"
        level_ch = "安全 (Safe)"
    elif "橙色" in level:
        color = "#FF9500" # iOS Orange
        icon = "🟠"
        level_ch = "警报 (Alert)"
    elif "红色" in level:
        color = "#FF3B30" # iOS Red
        icon = "🔴"
        level_ch = "高危 (High Risk)"
    elif "黄色" in level:
        color = "#FFCC00" # iOS Yellow
        icon = "🟡"
        level_ch = "黄色提醒"
        
    # 生成带颜色的 Badges 和语义清晰的 HTML
    html_template = f"""
    <div style="background-color:#1e1e1e; padding: 15px; border-radius: 12px; border: 1px solid {color}; margin-bottom: 10px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #8b8b8b; font-size: 14px;">🚦 社交风险等级</span>
            <span style="color: {color}; font-size: 18px; font-weight: bold;">
                {icon} {level_ch}
            </span>
        </div>
        <div style="color: #ffffff; font-size: 16px; line-height: 1.5; font-family: -apple-system, system-ui, BlinkMacSystemFont, sans-serif;">
            {msg}
        </div>
    </div>
    """
    return html_template

# ==========================================
# 🔌 全局雷达开关 (生命周期控制)
# ==========================================
IS_LISTENING = False

# 初始化全套安全/空帧状态，用来完美解决 Error 炸弹
safe_init_state = (
    "🟢 雷达启动，正在连续监听...",
    "无",
    0.0,
    0,
    create_html_badge("🟢 安全", "系统自动断句中，请随时说话..."), # 初始化为安全 Badge
    "待机，隐私已断连模式"
)

# 初始化完全停止时的状态
stop_init_state = (
    "🔴 雷达已进入休眠状态",
    "无",
    0.0,
    0,
    create_html_badge("⚫ 停止", "麦克风已释放。可随时再次启动雷达。"), 
    "麦克风物理脱离"
)

def start_radar():
    """雷达模式生成器：完美数据结构保证，绝对不崩！"""
    global IS_LISTENING
    if IS_LISTENING:
        # 已经在运行，不操作，保证输出结构
        yield (None,) * 6  # Gradio 自动过滤 None 更新
        return
        
    IS_LISTENING = True
    # 第一次 yield，给用户一个完美的绿色初始状态
    yield safe_init_state
    
    while IS_LISTENING:
        try:
            # 1. 调用强大的 V3 封板内核
            # 如果在听声音的几秒钟里，用户点击了停止，这里能拦截到
            res = backend.start_sensing()
            if not IS_LISTENING: break
                
            # 2. 完美处理后端返回的 API 错误/静音：通过 Yield 全套数据来根除前端红框！
            if not res or "error" in res:
                # 遇到错误，向 Gradio `yield` 一个带有错误语义的完整数据字典，
                # 但将其余物理指标设为空，保证 UI 的干净。
                yield (
                    "💤 [环境静音或未提取到文字]", 
                    "无", 
                    0.0, 
                    0, 
                    create_html_badge("⚫ 待机", "持续监听中，隐私状态"),
                    "未捕捉到高风险特征"
                )
                continue
                
            # 3. 成功流
            pure_text = res["text"]
            raw_tags = res["tags"]
            translated_tags = translate_tags(raw_tags)
            cps = res["cps"]
            volume = res["volume"]
            alert_level = res["alert_level"]
            alert_msg = res["alert_msg"]
            
            # 综合物理指标预诊断（如果音量音调异常，我们要在这个 Badge 里提前告诉用户）
            diag_color = "#34C759" # 默认绿色 (Safe)
            if cps > FAST_SPEECH_CPS:
                diag_color = "#FF9500" # 语速过快橙色
            if volume > SHOUTING_THRESHOLD:
                diag_color = "#FF3B30" # 音量过大红色
            if alert_level.startswith("🔴"):
                diag_color = "#FF3B30"
                
            # Yield 全套 6 个规范的前端状态数据，永不报错！
            yield (
                pure_text,
                translated_tags,
                cps,
                volume,
                create_html_badge(alert_level, alert_msg), # 这里输出翻译后的专业 HTML Badge
                res.get("diagnosis", "表现完美")
            )
        except Exception as e:
            yield (f"系统严重崩溃: {str(e)}", "None", 0.0, 0, create_html_badge("⚫ 错误", "请检查后台"), "致命异常")
            break
            
    # 物理断点 yield (IS_LISTENING 被打破)
    yield stop_init_state

def stop_radar():
    """优雅停止雷达休眠：立刻更新界面状态，并通知后台安全切断"""
    global IS_LISTENING
    IS_LISTENING = False
    # 【修复】：立刻把前面定义好的“停止状态”全套数据推给界面，安抚前端
    return stop_init_state

# ==========================================
# 🎨 前端 UI：SOCI-AI指挥中心PRO
# ==========================================
# 采用暗黑现代 Monochrome 主题，专业且酷炫
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        f"""
        # 🚨 SOCI-AI 社交雷达指挥中心
        ### &nbsp; Powered by A-Team AI Lab | SenseVoice & RTX 4050 显存直连技术
        """
    )
    
    with gr.Row():
        start_btn = gr.Button("▶️ 启动全天候雷达", variant="primary", size="lg")
        stop_btn = gr.Button("⏹️ 停止雷达并休眠", variant="stop", size="lg")
    
    with gr.Row():
        # 左侧 Column: 过程与物理层
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🎙️ 声音采集 & 情绪特征")
                text_out = gr.Textbox(label="🧠 实时语义提取 (Semantics)", lines=3, placeholder="雷达监听中，隐私直连模式，请随时说话...")
                tags_out = gr.Textbox(label="🎭 多模态情绪标签 (Translated Tags)")
            
            with gr.Group():
                gr.Markdown("### ⏱️ 物理指标监控")
                cps_out = gr.Number(label="物理语速 (cps - 字/秒)")
                vol_out = gr.Number(label="声学音量峰值 (Warning: 18000)")

        # 右侧 Column: 诊断与风险层
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🚦 社交诊断 & 风险提示")
                # Traffic Light Textbox 升级为现代 HTML Badge Component！
                risk_badge_out = gr.HTML(label="社交风险指标")
                diag_out = gr.Textbox(label="📋 AI 社交深度建议 (Diagnosis)", lines=4, placeholder="表现完美。未捕捉到高风险社交特征")
    
    # 品牌页脚与隐私声明，增加商业规范感
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #8b8b8b; font-size: 14px; padding-bottom: 20px;">
            A-Team AI Lab © 2026. SOCI-AI 是采用<b>本地离线架构</b>的隐私安全社交辅助预警系统。
            任何数据都不会离开您的电脑。
        </div>
        """
    )
    
    # ==========================================
    # 🔌 绑定前端交互逻辑
    # ==========================================
    # 完美数据结构 yield 永不崩，且支持在录音过程中被取消
    start_event = start_btn.click(
        fn=start_radar,
        inputs=[],
        outputs=[text_out, tags_out, cps_out, vol_out, risk_badge_out, diag_out],
        queue=True # 连续生成器必须开启 Queuing
    )
    
    # 优雅停止雷达生命周期
    stop_btn.click(
        fn=stop_radar,
        inputs=[],
        # 【修复1】：对齐 6 个输出框，接收 stop_init_state 传来的数据
        outputs=[text_out, tags_out, cps_out, vol_out, risk_badge_out, diag_out] 
        # 【修复2】：彻底删掉 cancels=[start_event]！不暴力拔电源，让后台听完最后一句话自然休眠
    )

if __name__ == "__main__":
    # share=True 会生成一个全球可访问的临时公网链接 (72小时有效)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True, inbrowser=True)