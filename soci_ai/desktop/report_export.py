"""Self-contained, escaped HTML review. No network, scripts or media capture."""
from html import escape
import json
from pathlib import Path

from .charts import finite_number, sample_caption, time_label


def export_payload(report, path, selected_filter):
    is_json = '(*.json)' in selected_filter if selected_filter else Path(path).suffix.lower() == '.json'
    target = Path(path)
    suffix = '.json' if is_json else '.html'
    if target.suffix.lower() in ('', '.json', '.html'):
        target = target.with_suffix(suffix)
    return str(target), json.dumps(report, ensure_ascii=False, indent=2) if is_json else render_html(report)


def render_html(report):
    def safe(value):
        return escape(str(value), quote=True)

    def number(value):
        value = finite_number(value)
        return '—' if value is None else f'{value:.1f}'

    rows = [row for row in report.get('history', [])
            if finite_number(row.get('at_ms')) is not None and float(row['at_ms']) >= 0]
    rows.sort(key=lambda row: float(row['at_ms']))
    duration = max(1, finite_number(report.get('elapsed_seconds')) or 0,
                   max((float(row['at_ms']) / 1000 for row in rows), default=0))
    paths, tips, singletons, segment = [], [], [], []
    def finish_segment():
        if len(segment) == 1:
            x, y = segment[0]
            singletons.append(f'<circle class="isolated-sample" cx="{x:.2f}" cy="{y:.2f}" '
                              'r="4" fill="url(#risk)"/>')
        segment.clear()
    for row in rows:
        value = finite_number(row.get('sbi'))
        if value is None:
            finish_segment()
            continue
        x = 60 + 920 * float(row['at_ms']) / 1000 / duration
        y = 315 - 2.5 * max(0, min(100, value))
        paths.append(f"{'L' if segment else 'M'}{x:.2f},{y:.2f}")
        segment.append((x, y))
        caption = f"{time_label(float(row['at_ms']) / 1000)} · SBI {value:.1f} · {sample_caption(row)}"
        tips.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="transparent"><title>{safe(caption)}</title></circle>')
    finish_segment()
    grid = ''.join(f'<path d="M60,{315-2.5*v}H980" stroke="#24414a" stroke-dasharray="5 5"/>'
                   f'<text x="17" y="{320-2.5*v}" fill="#90b0be">{v}</text>' for v in (0,25,60,100))
    axis = ''.join(f'<text x="{60+920*i/4}" y="345" text-anchor="middle" fill="#90b0be">'
                   f'{time_label(duration*i/4)}</text>' for i in range(5))
    curve = '<svg viewBox="0 0 1040 365" role="img" aria-label="SBI 时间曲线">' + (
        '<defs><linearGradient id="risk" x1="0" y1="315" x2="0" y2="65" gradientUnits="userSpaceOnUse">'
        '<stop offset="0" stop-color="#69e4bb"/><stop offset=".25" stop-color="#69e4bb"/>'
        '<stop offset=".48" stop-color="#edc678"/><stop offset=".60" stop-color="#ff647e"/>'
        '<stop offset="1" stop-color="#ff647e"/></linearGradient></defs>'
        '<rect x="60" y="65" width="920" height="100" fill="#ff647e" opacity=".06"/>'
        + grid + axis + '<path d="' + ' '.join(paths) + '" fill="none" stroke="url(#risk)" '
        'stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>' + ''.join(singletons) + ''.join(tips) + '</svg>')
    cards = ''.join('<section><h2>' + safe(item.get('title', '会话总结')) + '</h2><p>' +
                    safe(item.get('body', '')) + '</p></section>' for item in report.get('advice', []))
    moments = ''.join('<p><strong>' + time_label((finite_number(item.get('at_ms')) or 0)/1000) +
                      ' · ' + safe(item.get('title', '表达变化')) + '</strong><br>' +
                      safe(item.get('detail', '')) + '</p>' for item in report.get('moments', []))
    transcripts = ''.join('<p><span>' + time_label((finite_number(item.get('at_ms')) or 0)/1000) +
                          '</span>　' + safe(item.get('text', '')) + '</p>' for item in report.get('transcripts', []))
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>SOCI AI 会话报告</title>' + (
        '<style>body{background:#07141d;color:#d9ece7;font:17px/1.8 "Microsoft YaHei",sans-serif;'
        'max-width:1080px;margin:35px auto;padding:0 24px}h1{font-size:32px}h2{color:#79dfbb;font-size:23px}'
        'section{background:#0c202b;border:1px solid #24414a;border-radius:20px;padding:18px 26px;margin:20px 0}'
        '.stats{display:flex;gap:35px;flex-wrap:wrap;color:#a9ded3}svg{width:100%}span{color:#90b0be}'
        'details{margin:25px 0}p{overflow-wrap:anywhere}@media print{body{background:white;color:#172e38}'
        'section{background:white}h2{color:#187254}}</style></head><body><h1>SOCI AI · 会话报告</h1>'
        '<p class="stats">会话时长 ' + time_label(finite_number(report.get('elapsed_seconds')) or 0) +
        '　平均 SBI ' + number(report.get('average_sbi')) + '　峰值 SBI ' + number(report.get('peak_sbi')) +
        '</p><section><h2>表达变化 · SBI 时间线</h2>' + curve +
        '<p><span>绿色：平稳　黄色：留意　红色：SBI ≥ 60。悬停曲线采样点查看原因；缺失值保持断开。</span></p></section>' +
        cards + '<section><h2>关键时刻</h2>' + (moments or '<p>未记录可回看的关键变化。</p>') + '</section>' +
        '<details><summary>查看确认转写</summary>' + (transcripts or '<p>没有确认转写。</p>') + '</details><p><span>' +
        safe(report.get('limitations') or 'SBI 为原型规则评分，不代表情绪、意图或识别准确率。') +
        '</span></p></body></html>')
