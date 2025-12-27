# SamGOP experiment quick guide (gaze-object improvements)

本指南列出了在本仓库中快速复现/对比的三个可落地改动：
1. **显式 gaze-instance matching / re-ranking**（主贡献）
2. **噪声鲁棒化的 VFM 伪标签利用**（质量过滤 + 软标签 + 边界降权）
3. **对比式能量约束（内高外低）**，在原有能量损失基础上升级

每个小节都给出：应修改的代码位置、实现要点、以及最省时间的消融方案。

## 1) 显式的 gaze ↔ instance 匹配 / 重打分
**修改位置**
- 训练：`maskGOP/modeling/criterion.py` 中已有的 gaze L2 与能量损失汇总逻辑（`forward` 末尾）。当前能量仅按 GT mask 做均值 [470-496]，可在此处追加“实例级能量聚合 + CrossEntropy”损失。
- 推理：`maskGOP/maskGOP.py` 的 `instance_inference` 中对 top-k mask 的筛选 [1030-1059]。在这里取出 gaze heatmap 预测（`x_gaze`）并对每个候选 `mask_pred` 计算 mask 内平均能量，作为重排分数。
- 损失权重注册：`maskGOP/maskGOP.py` 构造 `weight_dict` 处已有 gaze/energy 权重 [440-494]，在此添加新 loss 的权重键并传入 criterion。

**实现步骤要点**
1. 在 `criterion.forward` 里，根据 Hungarian 匹配结果拿到候选 mask（`outputs['pred_masks']`）与 GT instance id，计算 `energy_k = mean(H * mask_k)`；用 softmax 得 `p_k`，对 GT instance 做 CE。无需改 backbone，仅用已有张量。
2. 推理时，在 `instance_inference` 重排分数：`score = cls_score * mask_prob * energy_k`（或只用 energy_k 作为加权），返回 argmax 实例。
3. 可选：对齐训练/推理，使用相同的能量聚合逻辑以避免 distribution shift。

**最省时间的消融**
- Baseline（原始 energy_loss） vs “仅显式匹配” vs “两者并用”。
- energy 计算方式：`sum(H*mask)`、`mean(H*mask)`、或带温度的 softmax。

## 2) VFM 伪标签的噪声鲁棒化
**修改位置**
- 伪 mask 生成 & gaze item 标注：`maskGOP/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py` [430-520] 已有基于 SAM 的 mask 与 gaze item mask 生成逻辑。
- 伪标签训练：`maskGOP/modeling/criterion.py` 中的 mask/dice 损失调用链；权重由 `weight_dict` 决定 [440-494]。

**实现步骤要点**
1. **质量过滤**：在 dataset mapper 中读取 SAM 提供的 score（若已保存在 ann 字段），低于阈值的 mask 跳过或赋极小权重，可通过在样本 dict 里记录 `mask_weight` 并在 loss 里乘权。
2. **软标签/边界降权**：在 mapper 把二值 mask 转成 float，并在边界带（例如膨胀-腐蚀的差集）设置更低权重；在 criterion 里用 BCE/Dice 时乘以该权重。
3. **一致性检查**：对比 SAM mask 外接框与 GT box IoU，IoU 过低时将该实例从监督中移除或降低权重。

**最省时间的消融**
- 阈值扫描（例如 score ∈ {0.0, 0.2, 0.5}）对 mSoC / AP 的影响。
- Hard mask vs soft mask vs soft+边界降权。

## 3) 对比式能量约束（内高外低）
**修改位置**
- 现有能量损失计算：`maskGOP/modeling/criterion.py` 中的 `mask_energy_loss` 逻辑 [470-496]。
- 损失权重注入：同样在 `weight_dict` 构造处添加新项 [440-494]。

**实现步骤要点**
1. 在能量均值之外，增加“外部惩罚”：`L = 1 - mean(H*mask_gt) + λ * mean(H*(1-mask_gt))`。
2. 也可使用“最难负样本”实例 mask（IoU 低但距离近）做对比：`L = -log( exp(E_pos) / (exp(E_pos) + exp(E_neg)) )`。
3. λ 可与“显式匹配”共享或单独调节，方便写成 ablation。

**最省时间的消融**
- 单纯外部惩罚 vs 正负对比；λ ∈ {0.1, 0.5, 1.0}。
- 观察密集小物体场景的定性可视化（热力图泄露到邻近物体的抑制效果）。

## 训练/验证建议
- 主指标：mSoC / mSoC@{50,75,95}、heatmap AUC/L2/Angle、以及实例分割 AP。
- 复现实验时保持 backbone / query 数不变，仅改输出端逻辑即可凸显贡献。
- 若引入新 loss，建议在 `configs` 中复制现有 config 新增超参字段，方便脚本化 grid search。
