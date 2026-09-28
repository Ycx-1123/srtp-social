"""Bounded, explainable language cues from the bundled original scenario library.

This is an offline deterministic demonstration, not a trained classifier. Scores
are authored rule strengths, never calibrated probabilities or intent judgments.
"""
from collections import Counter
from functools import lru_cache
import json
import re
import sqlite3
import unicodedata

from .resources import resource_root


_LIMIT = 4096
_LIMITATIONS = "原型规则提示，不是意图判断或准确率；未命中不代表绝对无冒犯。"
_NO_MATCH = "未命中当前样例及限定规则；不代表语言绝对无冒犯。"
_SENTENCES = re.compile(r"[^。！？!?；;\n]+")
_REJECT_BEFORE = re.compile(
    r"(?:不要说|不能说|别说|不应该说|不该说|不能用|不要用|拒绝|(?<![不没未])反对|不赞成|"
    r"不同意|不认同|不认为|不觉得|不是|不代表|不能认为|别再说|请勿说|谁说|不要觉得|"
    r"不是所有|并非所有|不等于)[^，,。；;！？!?]{0,60}$")
_REJECT_AFTER = re.compile(
    r"^(?:[”’\"'」』）)]|这(?:种|个|句)?(?:话|说法|观点|表述)?|这种观点|这种说法|"
    r"的说法|说法|观点|[，,\s]|我|我们|大家|对此|并|是|并不){0,20}"
    r"(?:不对|错误|不正确|不合理|不成立|不同意|不认同|反对|有偏见|是偏见|不该这样说)")
_REFERENCE = re.compile(r"(?:引用|例句|反面例子|作为反例|偏见例子|偏见示例|刻板印象示例|"
                        r"这句话为什么有偏见|这句话是否有偏见|这种说法有什么问题)")
_QUOTE = re.compile(r"[“‘\"'「『]([^”’\"'」』]{1,500})[”’\"'」』]")

# Presentation labels describe authored rule categories, not a diagnosis.
_LABELS = {
    'gender_ability': ('性别能力偏见', '能力预设'),
    'gender_roles': ('性别角色限定', '角色限制'),
    'age_bias': ('年龄偏见', '代际概括'),
    'region_accent': ('地域口音偏见', '地域归因'),
    'education_bias': ('学历偏见', '资格预设'),
    'appearance_bias': ('外貌偏见', '外貌归因'),
    'socioeconomic_bias': ('经济背景偏见', '家庭背景归因'),
    'disability_bias': ('健康残障偏见', '能力泛化'),
    'family_status': ('婚育身份偏见', '照护身份预设'),
    'identity_othering': ('身份归属排斥', '归属限制'),
    'ability_dismissal': ('能力贬低', '参与资格否定'),
    'emotional_invalidation': ('感受否定', '体验轻视'),
}
_TASK = re.compile(r'跑腿|分工|安排|负责|负责人|岗位|招聘|录用|面试|分配')
# Approximate transcriptions are review hints only. Never repair polarity or
# identity words into a risky sentence, and never score approximate matches.
_PROTECTED = frozenset('不没未无非别莫能要宜难好坏低高多少强弱男女老幼穷富病残障')


def _speech_form(text):
    return text.replace('一点', '点').replace('一些', '些')


def _one_safe_edit(left, right):
    if len(left) != len(right):
        # Only substitutions are previewed. Insertion/deletion can absorb
        # adjacent rejection words; contractions use the separate safe path.
        return False
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(left), len(right)) - prefix and
           left[len(left) - suffix - 1] == right[len(right) - suffix - 1]):
        suffix += 1
    removed = left[prefix:len(left) - suffix if suffix else len(left)]
    added = right[prefix:len(right) - suffix if suffix else len(right)]
    return len(removed) == len(added) == 1 and not any(c in _PROTECTED for c in removed + added)


def _near_spans(sentence, phrase):
    if len(phrase) < 10 or len(sentence) > 128:
        return
    starts = set()
    for anchor, offset in ((phrase[:3], 0), (phrase[-3:], len(phrase) - 3)):
        index = sentence.find(anchor)
        while index >= 0:
            starts.update(index - offset + delta for delta in (-1, 0, 1))
            index = sentence.find(anchor, index + 1)
    for start in sorted(starts):
        if start < 0:
            continue
        for length in (len(phrase),):
            end = start + length
            if end <= len(sentence) and _one_safe_edit(phrase, sentence[start:end]):
                yield start, end


def _label_fields(row, matched, kind, review=False):
    title, mechanism = _LABELS.get(row['category'], ('语言表达提醒', '需结合语境'))
    tags = [title, mechanism]
    if _TASK.search(matched):
        tags.append('任务分配')
    return {'category_label': title, 'tags': tags, 'match_kind': kind, 'review_required': review}

# Variants deliberately combine a group noun with a limited ability predicate
# and a limited task noun. A bare '女生不适合' / '你不用参加' is not a rule.
_FILLER = r"(?:啊|呀|嘛|呢)?(?:就是|本来就|天生就|天生|总是|都|就)?"
_VARIANTS = (
    ("GA001", re.compile(r"(?:女生|女孩子|女性)" + _FILLER +
                         r"(?:学不好|学不会|不擅长|不适合(?:学|做)?)"
                         r"(?:工科|工程|编程|算法|数学|机械|电路|科研|技术)(?:吧|啊|嘛)?")),
    ("GA019", re.compile(r"(?:男生|男孩子|男性)" + _FILLER +
                         r"(?:学不好|学不会|不擅长|不适合(?:学|做)?)"
                         r"(?:护理|幼教|语言|舞蹈)(?:吧|啊|嘛)?")),
    ("AB001", re.compile(r"(?:年纪大了|岁数大了|上了年纪)"
                         r"(?:就|还是|最好)?(?:别|不要)(?:再)?学(?:编程|新软件|新技术)(?:了|吧)?")),
    ("ED001", re.compile(r"(?:普通学校|非名校|专科|大专)(?:毕业|出身)?(?:的|的人)?"
                         r"(?:能有多强|做不了复杂项目|能力就是不行)(?:啊|吧)?")),
    ("EI001", re.compile(r"(?:你)?(?:别|不要)(?:这么|那么|太)敏感(?:了|吧)?")),
)


def _normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


@lru_cache(maxsize=1)
def _library():
    directory = resource_root() / "resources/language"
    database = directory / "social_bias_examples.sqlite"
    metadata, rows, source = {}, [], "unavailable"
    try:
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        try:
            connection.row_factory = sqlite3.Row
            rows = [dict(row) for row in connection.execute("SELECT * FROM examples ORDER BY id")]
            metadata = {row["key"]: json.loads(row["value"])
                        for row in connection.execute("SELECT * FROM metadata")}
            source = "sqlite"
        finally:
            connection.close()
    except (OSError, sqlite3.Error, ValueError, KeyError):
        try:
            corpus = json.loads((directory / "social_bias_examples.json").read_text(encoding="utf-8"))
            rows, metadata, source = corpus["examples"], corpus["metadata"], "json_fallback"
        except (OSError, ValueError, KeyError):
            rows, metadata = [], {}
    positives, controls = [], set()
    for row in rows:
        row = dict(row)
        row["normalized_text"] = _normalize(row["text"])
        row['speech_text'] = _speech_form(row['normalized_text'])
        if row["label"] == "bias":
            positives.append(row)
        else:
            controls.add(row["normalized_text"])
    positives.sort(key=lambda row: len(row["normalized_text"]), reverse=True)
    return tuple(positives), frozenset(controls), metadata, source


def corpus_status():
    """Return loaded counts and provenance for status labels and saved reports."""
    positives, controls, metadata, source = _library()
    return {"source": source, "bias_examples": len(positives),
            "control_examples": len(controls),
            "categories": len(Counter(row["category"] for row in positives)),
            "provenance": metadata.get("provenance", "unavailable"),
            "version": metadata.get("version", ""), "limitations": _LIMITATIONS,
            'label_catalog': {key: {'label': value[0], 'mechanism': value[1]} for key, value in _LABELS.items()}}


def _context_rejects(sentence, start, end):
    before, after = sentence[:start], sentence[end:]
    if _REJECT_BEFORE.search(before) or _REJECT_AFTER.search(after):
        return True
    if re.search(r"(?:讨论|分析|辨析|识别)", before) and re.match(
            r"[”’\"'」』]?这句话(?:为什么有偏见|是否有偏见|有什么问题)", after):
        return True
    # A quoted phrase is not automatically endorsement. Only explicit teaching,
    # reporting or rejection context suppresses it; quotes alone are ambiguous.
    for quote in _QUOTE.finditer(sentence):
        if quote.start() <= start and end <= quote.end():
            outside = sentence[:quote.start()] + sentence[quote.end():]
            if re.search(r"(?:我|我们)(?:也)?(?:觉得很对|觉得有道理|这么认为|赞同|同意这个观点)", outside):
                return False
            if _REFERENCE.search(outside) or re.search(r"(?:有人说|有人曾说|听到有人说|原话是)", outside):
                return True
    return False


def analyze_text(text):
    """Explain the strongest unsuppressed cue in at most 4096 recent characters.

    Matching is local to a sentence so rejecting one quote cannot suppress a
    separate assertion. Category rewrites are prompts for human review.
    """
    positives, controls, _, source = _library()
    text = str(text or "")
    truncated = len(text) > _LIMIT
    # Keep line boundaries: an ASR/text line can reject one statement while the
    # next line asserts another. Whitespace folding must not merge their scope.
    normalized = "\n".join(_normalize(line) for line in text[-_LIMIT:].splitlines())
    context_note = _LIMITATIONS
    if truncated:
        context_note += "本次仅检查最近4096个字符。"
    if source == "unavailable":
        context_note += "语言样例资源不可用，请检查安装目录。"
    candidates = []
    sentences = []
    by_id = {row["id"]: row for row in positives}
    for part in _SENTENCES.finditer(normalized):
        sentence = part.group()
        if sentence in controls:
            continue
        sentences.append(sentence)
        for row in positives:
            phrase = row["normalized_text"]
            start = sentence.find(phrase)
            while start >= 0:
                end = start + len(phrase)
                if not _context_rejects(sentence, start, end):
                    candidates.append((row, sentence[start:end], "样例匹配"))
                start = sentence.find(phrase, end)
        for example_id, pattern in _VARIANTS:
            row = by_id.get(example_id)
            if row is None:
                continue
            for match in pattern.finditer(sentence):
                if not _context_rejects(sentence, match.start(), match.end()):
                    candidates.append((row, match.group(), "限定表达变体"))
    # Uniform contraction handling applies to every library entry; no phrase
    # is privileged because it was used in a manual demonstration.
    if not candidates:
        for sentence in sentences:
            spoken = _speech_form(sentence)
            for row in positives:
                phrase = row['speech_text']
                start = spoken.find(phrase)
                while start >= 0:
                    end = start + len(phrase)
                    if not _context_rejects(spoken, start, end):
                        candidates.append((row, spoken[start:end], '口语缩略匹配'))
                    start = spoken.find(phrase, end)
    if not candidates:
        previews = []
        for sentence in sentences:
            for row in positives:
                for start, end in _near_spans(sentence, row['normalized_text']):
                    if not _context_rejects(sentence, start, end):
                        previews.append((row, sentence[start:end]))
        if previews:
            row, matched = max(previews, key=lambda item: len(item[1]))
            return {'score': 0.0, 'explanation': row['explanation'], 'category': row['category'],
                    'matched_text': matched, 'rewrite': row['rewrite'], 'example_id': row['id'],
                    'context_note': '与样例相近，请核对识别字幕；本提示不计入 SBI。' + _LIMITATIONS,
                    **_label_fields(row, matched, '近似转写待核对', review=True)}
        return {"score": 0.0, "explanation": _NO_MATCH, "category": "",
                "matched_text": "", "rewrite": "", "example_id": "",
                "context_note": context_note, 'category_label': '', 'tags': [],
                'match_kind': '', 'review_required': False}
    row, matched, match_kind = max(candidates, key=lambda value: (value[0]["weight"], len(value[1])))
    return {"score": float(row["weight"]), "explanation": row["explanation"],
            "category": row["category"], "matched_text": matched,
            "rewrite": row["rewrite"], "example_id": row["id"],
            **_label_fields(row, matched, match_kind),
            "context_note": f"{match_kind}；{row['context']} {_LIMITATIONS}" +
                            ("本次仅检查最近4096个字符。" if truncated else "")}


def classify_text(text):
    """Compatibility interface retained for the real-time assessment engine."""
    result = analyze_text(text)
    return result["score"], result["explanation"]
