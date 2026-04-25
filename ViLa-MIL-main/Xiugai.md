<!--
 * @Author: ljh 1294245800@qq.com
 * @Date: 2026-04-24 10:12:15
 * @LastEditors: ljh 1294245800@qq.com
 * @LastEditTime: 2026-04-24 16:12:38
 * @FilePath: /ViLMIL/ViLa-MIL-main/Xiugai.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
需求文档：基于特权信息学习 (LUPI) 改进 ViLa-MIL 的二分类网络
项目基线状态：
我已经拥有了一套基于 PyTorch 跑通的 ViLa-MIL 代码，并且底层特征提取器已经成功替换为 BiomedCLIP。目前模型拥有 Prototype-guided Patch Decoder，能够处理图像块特征。
当前改造目标：
将现有的模型推理范式重构为 特权信息学习 (LUPI) 框架，用于全切片图像（WSI）的二分类任务（腺癌 vs. 非腺癌，数据量约 968 例，类别比例约 2:1）。
核心诉求是：训练时使用患者专属的病理文本作为特权信息指导图像特征学习；测试时完全切断文本分支，仅使用图像进行单模态分类。
请帮我编写核心网络结构的修改代码以及配套的 Loss 和训练逻辑。
1. 离线数据输入约定
模型在线训练部分不包含重型特征提取器，直接接收离线提取好的 BiomedCLIP 特征：
image_patch_features: 形状为 $N \times D$ （$N$ 为图块数，$D=512$）。
text_features: 形状为 $1 \times D$ （这是用病理石蜡报告清洗后提取的特权文本特征，仅在训练时提供）。
label: 二分类标签。
2. 核心网络架构修改 (LUPI 结构)
请在我的 ViLa-MIL 基线之上，实现以下 LUPI_ViLaMIL 整体模型类：
复用图像聚合器：调用已有的 Prototype-guided Patch Decoder，输入 $N \times D$ 的图像块特征，输出聚合后的 Slide 级别图像特征 $V_{img}$（形状 $1 \times D$）。
新增跨模态投影层 (Projection Heads)：
包含两个独立的非线性 MLP（例如 Linear -> ReLU -> Linear）。
分别用于将 $V_{img}$ 和输入的特权文本特征 $V_{text}$ 映射到同一个对齐空间，输出 $Z_{img}$ 和 $Z_{text}$。
新增纯图像分类头 (Classification Head)：
一个简单的线性分类器（MLP）。
注意输入：它直接接收未经投影的 $V_{img}$（而不是 $Z_{img}$），输出大小为 2 的 Logits。
3. 双损失函数设计 (Dual-Loss)
在 forward 函数中（处于 training 模式时），需计算并返回两个 Loss：
特征对齐损失 (Alignment Loss)：使用 InfoNCE Loss 计算同 Batch 内 $Z_{img}$ 与 $Z_{text}$ 的对比损失（拉近正样本对，推开负样本对）。
分类损失 (Classification Loss)：为了应对 2:1 的数据不平衡，请使用 Focal Loss 或带类别权重的 CrossEntropyLoss 计算分类头 Logits 与 Label 之间的损失。
总损失返回：Total_Loss = Classification_Loss + lambda * Alignment_Loss。
4. 前向传播逻辑控制 (Train vs. Eval)
这是最关键的改进，必须在代码中明确区分：
当 self.training == True 时：接收 (image_features, text_features)，计算并返回分类 Logits 以及 Alignment Loss（或直接返回 Total Loss）。
当 self.training == False (或 Eval 模式) 时：模型绝不接收或处理任何文本特征。数据流仅为：image_features $\rightarrow$ Patch Decoder $\rightarrow$ $V_{img}$ $\rightarrow$ Classification Head $\rightarrow$ 返回分类 Logits。
5. 输出要求
请直接提供：
LUPI_ViLaMIL 的网络结构代码（含投影层和分类头构建，以及条件区分的 forward 函数）。
自定义的联合 Loss 计算模块代码。
演示如何调用该模型进行 train_step 和 val_step 的核心循环伪代码/代码片段。
