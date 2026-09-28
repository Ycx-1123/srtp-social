# 中文社会偏见表达场景库

本库包含 **300 条偏见/轻视表达场景 + 60 条对照表达，共 360 条**。用于 SOCI-AI 桌面原型的离线语言提示、可解释复盘和演示，不包含暴力威胁或侮辱性族群称呼。

全部句子是为本项目新编写的**合成场景**，由 AI 辅助撰写；不是从社交平台采集的真实发言，不是人工标注的真实会话，不是经过实证验证的训练集。没有导入下列参考项目中的文本、用户资料或其他个人元数据。

## 文件与可追溯字段

- `social_bias_examples.json`：唯一人工可审阅的源文件，包含说明元数据、参考来源和全部样例。
- `social_bias_examples.sqlite`：从同一 JSON 生成，桌面程序默认只读加载并缓存，不访问网络。
- `../../tools/build_language_database.py`：构建工具，校验条数、唯一性、字段与权重，写入索引及源 JSON 的 SHA-256。

每条记录均有 `id`、`text`、`label`（`bias` 或 `control`）、`category`、`context`、`explanation`、`rewrite`、`weight`、`provenance`、`source_role`。所有记录明确标记 `provenance: synthetic_authored` 和 `source_role: original_sample`。解释和改写建议按类别复用，是便于实际沟通的提示，并非对每句话的专家标注。对照句包含反偏见、否定、引用和具体任务情境下的尊重表达。

| 类别 | 偏见场景 | 对照句 |
| --- | ---: | ---: |
| 性别与能力 `gender_ability` | 25 | 5 |
| 性别与角色分配 `gender_roles` | 25 | 5 |
| 年龄与代际 `age_bias` | 25 | 5 |
| 地域与口音 `region_accent` | 25 | 5 |
| 学历与学校 `education_bias` | 25 | 5 |
| 外貌与身体形象 `appearance_bias` | 25 | 5 |
| 经济与家庭背景 `socioeconomic_bias` | 25 | 5 |
| 残障与健康状况 `disability_bias` | 25 | 5 |
| 婚育与照护身份 `family_status` | 25 | 5 |
| 身份与归属排斥 `identity_othering` | 25 | 5 |
| 能力贬低与参与排除 `ability_dismissal` | 25 | 5 |
| 感受否定与轻视 `emotional_invalidation` | 25 | 5 |

后两类还涵盖不涉及群体身份的一般贬低沟通。类别用于产品提示，不声称是一套新的科学分类体系。每句是否产生伤害仍依赖语境、关系、权力差异和当事人的理解。

## 参考来源与使用边界

这些来源仅用于认识已有研究的类别与任务设计，**不作为本库 300 条句子的来源**：

1. [COLDataset（thu-coai，Apache-2.0）](https://github.com/thu-coai/COLDataset)：参考冒犯/非冒犯、反偏见测试，以及性别、地域和种族等维度。
2. [SWSR（aggiejiang，MIT）](https://github.com/aggiejiang/SWSR)：参考刻板印象、文化背景、微冒犯等类别区分。
3. [CHBias（hyintell，MIT）](https://github.com/hyintell/CHBias)：参考性别、性取向、年龄、外貌等偏见维度。

上述许可名称描述参考仓库的许可，不表示本库复制了它们的数据，也不自动决定任何后续真实社交媒体采集的使用权。若后续引入公开文本，应单独核查文本来源、许可、隐私和标注流程，并使用独立的来源标记。

本项目可主张的工作是组织了一套原创中文场景样例、对照句、离线数据库及解释性规则。不能据此主张提出新模型、创造了全新微冒犯理论、达到某个真实准确率，或完成了科研意义上的语料验证。

## 程序使用方式与已知限制

`soci_ai.desktop.language.analyze_text(text)` 返回分数、命中表达、样例 ID、类别、解释、改写建议与语境说明；`classify_text(text)` 保持 `(score, evidence)` 接口。`corpus_status()` 返回实际加载来源、样例数和来源说明。

程序优先读取 SQLite；数据库缺失或不可读取时尝试同目录 JSON，均缺失时在语境说明中报告资源不可用。没有从网络临时补充词库。热启动后缓存库内容，逐句检查规范化的原始表达，再匹配少量限定变体；仅检查输入最后 4096 个字符并在截断时说明。否定、明确引用、反面例子等上下文会抑制相应命中，但不同句子分别判断。裸露的“女生不适合”和“你不用参加”不是无条件命中规则，避免将临时嗓子不适、允许休息等情境直接判成偏见。

`weight` 是手工设定的原型规则强度，不是概率。命中不能证明发言者有恶意，未命中不能证明没有冒犯。库内 300/60 条回归检查只验证实现和库的一致性：**这些句子已在规则库中，因此不是独立测试集，不能将通过率当作真实准确率**。反讽、转述、混合立场、长距离否定、跨句上下文及语音转写错误仍可能漏报或误报。

在项目根目录运行：

```powershell
python tools/build_language_database.py
python -m unittest discover -s tests -p test_desktop_language.py -v
```

SQLite 的 `examples` 表提供主键、规范化文本唯一索引和标签/类别索引；`metadata` 表提供版本、参考列表、方法说明、验证状态及源文件摘要。桌面分发需要同时包含本目录中的 JSON、SQLite 和本说明文件。
