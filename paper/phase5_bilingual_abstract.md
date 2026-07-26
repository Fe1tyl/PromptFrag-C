# Bilingual Abstract

## English Abstract

Zero-shot vision-language classifiers normally use a fixed textual template to
construct class representations, yet semantically intended paraphrases may
respond differently to visual distribution shift. We evaluated 12 frozen
class-prompt templates and a probability ensemble with OpenAI-pretrained RN50
and ViT-B/32 CLIP on CIFAR-10-C. The fully crossed design covered 15 corruption
types, five severities, 10,000 images per condition, and three primary
outcomes: accuracy, expected calibration error (ECE), and area under the
risk-coverage curve (AURC). Prompt effects were tested after averaging severity
within each corruption, and a leave-one-corruption-out (LOCO) rule selected
prompts using a frozen stability score. The mean within-condition accuracy
range was 11.66 percentage points for RN50 and 4.75 points for ViT-B/32. All
six model-by-metric prompt effects remained significant after global Holm
correction. LOCO improved accuracy by 1.03 and 1.28 points, respectively, but
worsened ECE by 1.93 and 1.00 points. It improved ViT-B/32 AURC, while the
RN50 AURC result was inconclusive. Prompt selection can therefore increase
recognition accuracy without uniformly improving reliability. Corruption
studies should disclose prompt sensitivity and report calibration or selective
risk alongside accuracy.

**Keywords:** vision-language models, zero-shot classification, distribution
shift, confidence calibration, selective prediction, sensitivity analysis

## 中文摘要（繁體）

零樣本視覺語言分類通常以單一文字模板建立類別表徵，但語意相近的句型在影像分布偏移下未必維持一致的可靠度。本研究以 OpenAI 預訓練之 RN50 與 ViT-B/32 CLIP，在 CIFAR-10-C 上評估十二種固定類別提示及機率集成。實驗涵蓋十五種破壞、五個嚴重度與每條件一萬張影像，並同時量測準確率、預期校準誤差與風險覆蓋曲線下面積。主要推論先在各破壞內平均嚴重度，再以破壞類型作為配對區塊；另以留一破壞法測試預先固定的穩定度選擇規則。RN50 與 ViT-B/32 的條件內提示準確率平均範圍分別為 11.66 與 4.75 個百分點，六項模型與指標組合皆通過全域 Holm 校正。穩定度選擇使兩模型準確率提高 1.03 與 1.28 個百分點，卻使校準誤差惡化 1.93 與 1.00 個百分點；其僅明確改善 ViT-B/32 的選擇性風險。結果顯示，提高辨識率不等同於整體可靠度提升。未來的零樣本破壞研究應揭露提示敏感度，並將準確率與校準及選擇性風險共同報告。

**關鍵詞：** 視覺語言模型、零樣本分類、分布偏移、信心校準、選擇性預測、敏感度分析

## Abstract Quality Report

| Metric | English | Traditional Chinese |
| --- | ---: | ---: |
| Length | 176 words | 316 Chinese characters (punctuation excluded) |
| Structural components | 5/5 | 5/5 |
| Keywords | 6 | 6 |
| Citations in abstract | 0 | 0 |
| Independent composition | PASS | PASS |

The two abstracts cover the same problem, design, principal findings, and
implication, while using language-specific sentence organization rather than a
sentence-by-sentence translation.
