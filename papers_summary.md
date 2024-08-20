# Papers Summary for 2024-08-20

## Advancing Voice Cloning for Nepali_ Leveraging Transfer Learning in a Low-Resource Language

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10128v1)

### Abstract (English)

Voice cloning is a prominent feature in personalized speech interfaces. A
neural vocal cloning system can mimic someone's voice using just a few audio
samples. Both speaker encoding and speaker adaptation are topics of research in
the field of voice cloning. Speaker adaptation relies on fine-tuning a
multi-speaker generative model, which involves training a separate model to
infer a new speaker embedding used for speaker encoding. Both methods can
achieve excellent performance, even with a small number of cloning audios, in
terms of the speech's naturalness and similarity to the original speaker.
Speaker encoding approaches are more appropriate for low-resource deployment
since they require significantly less memory and have a faster cloning time
than speaker adaption, which can offer slightly greater naturalness and
similarity. The main goal is to create a vocal cloning system that produces
audio output with a Nepali accent or that sounds like Nepali. For the further
advancement of TTS, the idea of transfer learning was effectively used to
address several issues that were encountered in the development of this system,
including the poor audio quality and the lack of available data.

### 摘要 (中文)

语音克隆是个性化语音接口中一个突出的特征。神经声纹克隆系统只需几段音频样本就能模仿某人的声音。在语音克隆领域，演讲编码和演讲适应都是研究热点。演讲适应依赖于微调一个多语种生成模型，该模型涉及训练专门模型来推断用于演讲编码的新演讲嵌入。这两种方法都能取得很好的表现，在使用少量克隆音频的情况下，即使是在自然性和原说话者相似度方面，两者都可以达到良好的性能。演讲编码方法更适用于低资源部署，因为它们需要较少的记忆空间，并且比演讲适应的时间更快，这可以提供稍高的自然性和相似性。主要目标是创建一个能够产生尼泊尔口音或听起来像尼泊尔的声音的声纹克隆系统。为了进一步发展TTS，转移学习的有效思想被用来解决开发过程中遇到的一些问题，包括音频质量差和可用数据不足的问题。

---

## Out-of-distribution generalization via composition_ a lens through induction heads in Transformers

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09503v1)

### Abstract (English)

Large language models (LLMs) such as GPT-4 sometimes appear to be creative,
solving novel tasks often with a few demonstrations in the prompt. These tasks
require the models to generalize on distributions different from those from
training data -- which is known as out-of-distribution (OOD) generalization.
Despite the tremendous success of LLMs, how they approach OOD generalization
remains an open and underexplored question. We examine OOD generalization in
settings where instances are generated according to hidden rules, including
in-context learning with symbolic reasoning. Models are required to infer the
hidden rules behind input prompts without any fine-tuning.
  We empirically examined the training dynamics of Transformers on a synthetic
example and conducted extensive experiments on a variety of pretrained LLMs,
focusing on a type of components known as induction heads. We found that OOD
generalization and composition are tied together -- models can learn rules by
composing two self-attention layers, thereby achieving OOD generalization.
Furthermore, a shared latent subspace in the embedding (or feature) space acts
as a bridge for composition by aligning early layers and later layers, which we
refer to as the common bridge representation hypothesis.

### 摘要 (中文)

大型语言模型（LLM）如GPT-4有时似乎具有创造力，通过在提示中演示几个示例任务，这些任务需要模型在训练数据的分布之外进行泛化。尽管大型语言模型的成功巨大，但它们如何处理OOD泛化的问题仍然是一个开放且未探索的问题。我们研究了在隐藏规则下生成实例的情况，包括在上下文学习中使用符号推理的情况下。模型必须从输入提示中推断出隐藏规则，而无需任何微调。
我们在合成例子上进行了培训动力学的实验，并对各种预训练的语言模型进行了广泛的经验研究，重点是称为诱导头的一种组件类型。我们发现OOD泛化和组合紧密地联系在一起——模型可以通过组合两个自注意力层来实现OOD泛化。此外，在嵌入空间（或特征空间）中的共享低维子空间也起到了桥梁的作用，这被称为共同桥接表示假设。

---

## Perturb-and-Compare Approach for Detecting Out-of-Distribution Samples in Constrained Access Environments

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10107v1)

### Abstract (English)

Accessing machine learning models through remote APIs has been gaining
prevalence following the recent trend of scaling up model parameters for
increased performance. Even though these models exhibit remarkable ability,
detecting out-of-distribution (OOD) samples remains a crucial safety concern
for end users as these samples may induce unreliable outputs from the model. In
this work, we propose an OOD detection framework, MixDiff, that is applicable
even when the model's parameters or its activations are not accessible to the
end user. To bypass the access restriction, MixDiff applies an identical
input-level perturbation to a given target sample and a similar in-distribution
(ID) sample, then compares the relative difference in the model outputs of
these two samples. MixDiff is model-agnostic and compatible with existing
output-based OOD detection methods. We provide theoretical analysis to
illustrate MixDiff's effectiveness in discerning OOD samples that induce
overconfident outputs from the model and empirically demonstrate that MixDiff
consistently enhances the OOD detection performance on various datasets in
vision and text domains.

### 摘要 (中文)

通过远程API访问机器学习模型已经越来越普遍，随着对模型参数进行大规模增加以提高性能的趋势。虽然这些模型表现出非凡的能力，但对于终端用户来说，检测离群样本（Out-of-Distribution，OOD）仍然是一个至关重要的安全问题，因为这些样本可能会从模型中产生不可靠的输出。在本工作中，我们提出了一种名为MixDiff的OOD检测框架，该框架适用于即使无法访问模型参数或激活的端点用户也能应用的情况。为了绕过访问限制，MixDiff将给定的目标样本和相似的ID样本施加相同的输入级扰动，并比较这两个样本模型输出之间的相对差异。MixDiff是模版无关的，兼容现有基于输出的OOD检测方法。我们提供了理论分析来说明MixDiff如何有效地区分诱导模型输出过于自信的OOD样本，并实证地展示了MixDiff在图像和文本领域的各种数据集上都能显著增强OOD检测性能。

---

## Detecting the Undetectable_ Combining Kolmogorov-Arnold Networks and MLP for AI-Generated Image Detection

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09371v1)

### Abstract (English)

As artificial intelligence progresses, the task of distinguishing between
real and AI-generated images is increasingly complicated by sophisticated
generative models. This paper presents a novel detection framework adept at
robustly identifying images produced by cutting-edge generative AI models, such
as DALL-E 3, MidJourney, and Stable Diffusion 3. We introduce a comprehensive
dataset, tailored to include images from these advanced generators, which
serves as the foundation for extensive evaluation. we propose a classification
system that integrates semantic image embeddings with a traditional Multilayer
Perceptron (MLP). This baseline system is designed to effectively differentiate
between real and AI-generated images under various challenging conditions.
Enhancing this approach, we introduce a hybrid architecture that combines
Kolmogorov-Arnold Networks (KAN) with the MLP. This hybrid model leverages the
adaptive, high-resolution feature transformation capabilities of KAN, enabling
our system to capture and analyze complex patterns in AI-generated images that
are typically overlooked by conventional models. In out-of-distribution
testing, our proposed model consistently outperformed the standard MLP across
three out of distribution test datasets, demonstrating superior performance and
robustness in classifying real images from AI-generated images with impressive
F1 scores.

### 摘要 (中文)

随着人工智能的发展，对抗真实图像和AI生成图像的识别任务变得越来越复杂。本文提出了一种新型检测框架，能够有效地识别来自先进生成式AI模型（如DALL-E 3、MidJourney和Stable Diffusion 3）产生的图像。我们提供了一个定制化的数据集，包括这些高级生成器生成的图像，用于进行广泛的评估。我们提出了一个分类系统，该系统结合了语义图象嵌入和传统多层感知机（MLP）。这个基线系统在各种挑战条件下都能有效区分真实和AI生成的图像。

为了增强这一方法，我们引入了一个融合了Kolmogorov-Arnold网络（KAN）与MLP的混合架构。这种混合模型利用了KAN的自适应高分辨率特征变换能力，使得我们的系统可以捕捉到AI生成图像中通常被常规模型忽视的复杂模式。在离域测试中，我们提出的模型在三个离域测试数据集上均表现出色，优于标准的MLP，在真实图像与AI生成图像之间进行了出色的身份识别，实现了令人印象深刻的F1分数。

---

## HYDEN_ Hyperbolic Density Representations for Medical Images and Reports

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09715v1)

### Abstract (English)

In light of the inherent entailment relations between images and text,
hyperbolic point vector embeddings, leveraging the hierarchical modeling
advantages of hyperbolic space, have been utilized for visual semantic
representation learning. However, point vector embedding approaches fail to
address the issue of semantic uncertainty, where an image may have multiple
interpretations, and text may refer to different images, a phenomenon
particularly prevalent in the medical domain. Therefor, we propose
\textbf{HYDEN}, a novel hyperbolic density embedding based image-text
representation learning approach tailored for specific medical domain data.
This method integrates text-aware local features alongside global features from
images, mapping image-text features to density features in hyperbolic space via
using hyperbolic pseudo-Gaussian distributions. An encapsulation loss function
is employed to model the partial order relations between image-text density
distributions. Experimental results demonstrate the interpretability of our
approach and its superior performance compared to the baseline methods across
various zero-shot tasks and different datasets.

### 摘要 (中文)

鉴于图像和文本之间内在的蕴含关系，我们利用超几何空间中的层次建模优势来学习视觉语义表示。然而，点矢量嵌入方法无法解决模糊不确定性问题，即图像可能有多个解释，而文本可能指向不同的图像，这是一种在医疗领域特别常见的现象。因此，我们提出一种基于超几何密度嵌入的新方法——HYDEN，它专门针对特定的医学数据集进行了定制化的图像-文本表示学习方法。这种方法结合了来自图像和文本的本地特征，并通过使用超几何伪高斯分布映射图像-文本特征到超几何空间中，从而在超几何伪高斯分布上使用封装损失函数模型部分序关系。实验结果表明，我们的方法具有良好的可解释性，并且在各种零任务任务以及不同数据集上的性能优于基准方法。

---

## 3D-Aware Instance Segmentation and Tracking in Egocentric Videos

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09860v1)

### Abstract (English)

Egocentric videos present unique challenges for 3D scene understanding due to
rapid camera motion, frequent object occlusions, and limited object visibility.
This paper introduces a novel approach to instance segmentation and tracking in
first-person video that leverages 3D awareness to overcome these obstacles. Our
method integrates scene geometry, 3D object centroid tracking, and instance
segmentation to create a robust framework for analyzing dynamic egocentric
scenes. By incorporating spatial and temporal cues, we achieve superior
performance compared to state-of-the-art 2D approaches. Extensive evaluations
on the challenging EPIC Fields dataset demonstrate significant improvements
across a range of tracking and segmentation consistency metrics. Specifically,
our method outperforms the next best performing approach by $7$ points in
Association Accuracy (AssA) and $4.5$ points in IDF1 score, while reducing the
number of ID switches by $73\%$ to $80\%$ across various object categories.
Leveraging our tracked instance segmentations, we showcase downstream
applications in 3D object reconstruction and amodal video object segmentation
in these egocentric settings.

### 摘要 (中文)

自我的视角视频在快速的相机运动、频繁的对象遮挡以及对象可见性有限的情况下，面临着独特的挑战。本论文提出了一种新的实例分割和跟踪方法，在第一人称视频中利用三维意识来克服这些障碍。我们的方法结合了场景几何学、三维物体中心追踪以及实例分割，创建了一个分析动态自我视角场景的强大框架。通过集成空间和时间线索，我们实现了比最先进的二维方法优越的表现。在具有挑战性的EPIC Fields数据集上进行广泛评估时，显著提高了各种跟踪和分割一致性指标的性能。具体来说，我们的方法在关联精度（AssA）和IDF1得分方面超过了最佳方法7分，在ID切换次数方面减少了73%-80%，在不同类别的对象分类中，平均减少了73%-80%。

利用跟踪到的实例分割结果，我们可以展示在这些自我视角环境中对三维对象重建和无模态视频对象分割的下游应用。

---

## New spectral imaging biomarkers for sepsis and mortality in intensive care

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09873v1)

### Abstract (English)

With sepsis remaining a leading cause of mortality, early identification of
septic patients and those at high risk of death is a challenge of high
socioeconomic importance. The driving hypothesis of this study was that
hyperspectral imaging (HSI) could provide novel biomarkers for sepsis diagnosis
and treatment management due to its potential to monitor microcirculatory
alterations. We conducted a comprehensive study involving HSI data of the palm
and fingers from more than 480 patients on the day of their intensive care unit
(ICU) admission. The findings demonstrate that HSI measurements can predict
sepsis with an area under the receiver operating characteristic curve (AUROC)
of 0.80 (95 % confidence interval (CI) [0.76; 0.84]) and mortality with an
AUROC of 0.72 (95 % CI [0.65; 0.79]). The predictive performance improves
substantially when additional clinical data is incorporated, leading to an
AUROC of up to 0.94 (95 % CI [0.92; 0.96]) for sepsis and 0.84 (95 % CI [0.78;
0.89]) for mortality. We conclude that HSI presents novel imaging biomarkers
for the rapid, non-invasive prediction of sepsis and mortality, suggesting its
potential as an important modality for guiding diagnosis and treatment.

### 摘要 (中文)

由于脓毒症仍然是致死性最高的原因之一，早期识别脓毒症患者和高死亡风险人群是一个具有高度社会经济重要性的挑战。这项研究的驱动力是HSI有可能通过监测微循环变化来提供新的脓毒症诊断和治疗管理的生物标志物。我们进行了一个涉及ICU入院当天患者的手掌和手指HSI数据的全面研究。结果表明，HSI测量可以预测脓毒症的AUC为0.80（95%置信区间[0.76；0.84]），并预测死亡的AUC为0.72（95%置信区间[0.65；0.79]）。当额外临床数据被纳入时，性能改善显著，最终AUC分别达到0.94（95%置信区间[0.92；0.96]）和0.84（95%置信区间[0.78；0.89]）对于脓毒症和死亡。我们得出结论，HSI提供了快速、非侵入性预测脓毒症和死亡的新影像学生物标志物，这表明它有潜力作为引导诊断和治疗的重要模式。

---

## Weakly Supervised Pretraining and Multi-Annotator Supervised Finetuning for Facial Wrinkle Detection

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09952v1)

### Abstract (English)

1. Research question: With the growing interest in skin diseases and skin
aesthetics, the ability to predict facial wrinkles is becoming increasingly
important. This study aims to evaluate whether a computational model,
convolutional neural networks (CNN), can be trained for automated facial
wrinkle segmentation. 2. Findings: Our study presents an effective technique
for integrating data from multiple annotators and illustrates that transfer
learning can enhance performance, resulting in dependable segmentation of
facial wrinkles. 3. Meaning: This approach automates intricate and
time-consuming tasks of wrinkle analysis with a deep learning framework. It
could be used to facilitate skin treatments and diagnostics.

### 摘要 (中文)

研究问题：随着对皮肤疾病和面部美学的兴趣日益增长，预测面部皱纹的能力变得越来越重要。本研究旨在评估卷积神经网络（CNN）是否可以用于自动面部皱纹分割。

发现：我们的研究提出了一种有效的融合多标记员数据的技术，并展示了迁移学习如何增强性能，从而实现了面部皱纹可靠分割。

意义：这种方法利用深度学习框架自动化了复杂的、耗时的任务，如面部皱纹分析。它可能被用来促进皮肤治疗和诊断。

---

## Facial Wrinkle Segmentation for Cosmetic Dermatology_ Pretraining with Texture Map-Based Weak Supervision

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10060v1)

### Abstract (English)

Facial wrinkle detection plays a crucial role in cosmetic dermatology.
Precise manual segmentation of facial wrinkles is challenging and
time-consuming, with inherent subjectivity leading to inconsistent results
among graders. To address this issue, we propose two solutions. First, we build
and release the first public facial wrinkle dataset, `FFHQ-Wrinkle', an
extension of the NVIDIA FFHQ dataset. This dataset includes 1,000 images with
human labels and 50,000 images with automatically generated weak labels. This
dataset can foster the research community to develop advanced wrinkle detection
algorithms. Second, we introduce a training strategy for U-Net-like
encoder-decoder models to detect wrinkles across the face automatically. Our
method employs a two-stage training strategy: texture map pretraining and
finetuning on human-labeled data. Initially, we pretrain models on a large
dataset with weak labels (N=50k) or masked texture maps generated through
computer vision techniques, without human intervention. Subsequently, we
finetune the models using human-labeled data (N=1k), which consists of manually
labeled wrinkle masks. During finetuning, the network inputs a combination of
RGB and masked texture maps, comprising four channels. We effectively combine
labels from multiple annotators to minimize subjectivity in manual labeling.
Our strategies demonstrate improved segmentation performance in facial wrinkle
segmentation both quantitatively and visually compared to existing pretraining
methods.

### 摘要 (中文)

面部皱纹检测在美容皮肤学中起着至关重要的作用。
精确的人工分割面部皱纹是一个具有挑战性和耗时的任务，其中主观性导致评分者之间结果不一致。为了应对这一问题，我们提出了两种解决方案。首先，我们建立并发布了一张公共的面部皱纹数据集`FFHQ-Wrinkle'，它是NVIDIA FFHQ数据集的扩展。该数据集包括1000张带有人类标签和50000张自动生成弱标签的图像。这个数据集可以促进研究社区开发先进的皱纹检测算法。其次，我们介绍了对U-Net类似编码器解码模型进行自动面部皱纹检测的方法训练策略。我们的方法采用两阶段训练策略：纹理地图预训练和使用人工标记的数据进行微调。首先，在一个包含弱标签（n=50K）或由计算机视觉技术生成的遮罩纹理图组成的大型数据集中预训练模型。随后，我们在使用人工标记的数据（n=1K），即手动标记的皱纹掩模上微调模型，这些数据由手动标记的皱纹掩膜组成。在微调过程中，网络输入一组包含四个通道的混合RGB和遮罩纹理图。我们有效地结合了多个标注者的标签来最小化手动标注中的主观性。我们的策略在定量和可视化方面都显示了比现有预训练方法更好的面部皱纹分割性能。

---

## Enhancing Modal Fusion by Alignment and Label Matching for Multimodal Emotion Recognition

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09438v1)

### Abstract (English)

To address the limitation in multimodal emotion recognition (MER) performance
arising from inter-modal information fusion, we propose a novel MER framework
based on multitask learning where fusion occurs after alignment, called
Foal-Net. The framework is designed to enhance the effectiveness of modality
fusion and includes two auxiliary tasks: audio-video emotion alignment (AVEL)
and cross-modal emotion label matching (MEM). First, AVEL achieves alignment of
emotional information in audio-video representations through contrastive
learning. Then, a modal fusion network integrates the aligned features.
Meanwhile, MEM assesses whether the emotions of the current sample pair are the
same, providing assistance for modal information fusion and guiding the model
to focus more on emotional information. The experimental results conducted on
IEMOCAP corpus show that Foal-Net outperforms the state-of-the-art methods and
emotion alignment is necessary before modal fusion.

### 摘要 (中文)

为了应对多模态情感识别（MER）性能受限于信息融合的局限性，我们提出了一个基于多任务学习的新MER框架，其中在对齐后进行融合，称为Foal-Net。该框架旨在增强模态融合的有效性，并包括两个辅助任务：音频视频情绪对齐（AVEL）和跨模态情绪标签匹配（MEM）。首先，通过对抗学习实现音频视频表示中情绪信息的对齐。然后，融合网络整合对齐特征。同时，MEM评估当前样本对是否相同的情绪，为模态信息融合提供帮助并引导模型更多地关注情感信息。在IEMOCAP语料库上进行的实验结果显示，Foal-Net优于最先进的方法，而在模态融合之前进行情绪对齐是必要的。

---

## Meta-Learning Empowered Meta-Face_ Personalized Speaking Style Adaptation for Audio-Driven 3D Talking Face Animation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09357v1)

### Abstract (English)

Audio-driven 3D face animation is increasingly vital in live streaming and
augmented reality applications. While remarkable progress has been observed,
most existing approaches are designed for specific individuals with predefined
speaking styles, thus neglecting the adaptability to varied speaking styles. To
address this limitation, this paper introduces MetaFace, a novel methodology
meticulously crafted for speaking style adaptation. Grounded in the novel
concept of meta-learning, MetaFace is composed of several key components: the
Robust Meta Initialization Stage (RMIS) for fundamental speaking style
adaptation, the Dynamic Relation Mining Neural Process (DRMN) for forging
connections between observed and unobserved speaking styles, and the Low-rank
Matrix Memory Reduction Approach to enhance the efficiency of model
optimization as well as learning style details. Leveraging these novel designs,
MetaFace not only significantly outperforms robust existing baselines but also
establishes a new state-of-the-art, as substantiated by our experimental
results.

### 摘要 (中文)

音频驱动的三维面部动画在直播和增强现实应用中日益重要。虽然取得了显著的进步，但现有的大多数方法都是针对特定个人设计的，具有预先设定的声音风格，因此忽视了对各种声音风格的适应性。为了应对这一局限性，本文提出了MetaFace，这是一种精心设计的方法，用于语音风格的自适应。它基于新颖的概念元学习，由几个关键组件组成：基础语音风格适应的稳健元初始化阶段（RMIS）、观察到的与未观察到的语言风格之间的动态关系挖掘神经过程（DRMN）以及低秩矩阵记忆减少方法来提高模型优化效率并增强学习风格细节。借助这些新颖的设计，MetaFace不仅大大超过了现有基准中的强健基线，而且建立了新的最高水平，这通过我们的实验结果得到了证实。

---

## SZU-AFS Antispoofing System for the ASVspoof 5 Challenge

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09933v1)

### Abstract (English)

This paper presents the SZU-AFS anti-spoofing system, designed for Track 1 of
the ASVspoof 5 Challenge under open conditions. The system is built with four
stages: selecting a baseline model, exploring effective data augmentation (DA)
methods for fine-tuning, applying a co-enhancement strategy based on gradient
norm aware minimization (GAM) for secondary fine-tuning, and fusing logits
scores from the two best-performing fine-tuned models. The system utilizes the
Wav2Vec2 front-end feature extractor and the AASIST back-end classifier as the
baseline model. During model fine-tuning, three distinct DA policies have been
investigated: single-DA, random-DA, and cascade-DA. Moreover, the employed
GAM-based co-enhancement strategy, designed to fine-tune the augmented model at
both data and optimizer levels, helps the Adam optimizer find flatter minima,
thereby boosting model generalization. Overall, the final fusion system
achieves a minDCF of 0.115 and an EER of 4.04% on the evaluation set.

### 摘要 (中文)

这篇论文提出了一种名为SZU-AFS的反欺骗系统，专为在开放条件下参加ASVspoof 5挑战赛下的Track 1。该系统由四个阶段组成：选择基线模型、探索有效的数据增强（DA）方法进行微调、基于梯度范数感知最小化（GAM）的联合强化策略对二级微调进行优化，并融合来自两个性能最佳微调模型的logits分数。系统利用Wav2Vec2前端特征提取器和AASIST后端分类器作为基准模型。在模型微调过程中，已经调查了三种不同的DA政策：单DA、随机DA和级联DA。此外，采用的设计用于在数据和优化器级别上微调增强模型的GAM - 基于策略帮助Adam优化器找到更平坦的极小值，从而提高模型的一般化能力。总的来说，最终的融合系统在评估集上的minDCF达到了0.115，EER达到了4.04%。

---

## Convert and Speak_ Zero-shot Accent Conversion with Minimum Supervision

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10096v1)

### Abstract (English)

Low resource of parallel data is the key challenge of accent conversion(AC)
problem in which both the pronunciation units and prosody pattern need to be
converted. We propose a two-stage generative framework "convert-and-speak" in
which the conversion is only operated on the semantic token level and the
speech is synthesized conditioned on the converted semantic token with a speech
generative model in target accent domain. The decoupling design enables the
"speaking" module to use massive amount of target accent speech and relieves
the parallel data required for the "conversion" module. Conversion with the
bridge of semantic token also relieves the requirement for the data with text
transcriptions and unlocks the usage of language pre-training technology to
further efficiently reduce the need of parallel accent speech data. To reduce
the complexity and latency of "speaking", a single-stage AR generative model is
designed to achieve good quality as well as lower computation cost. Experiments
on Indian-English to general American-English conversion show that the proposed
framework achieves state-of-the-art performance in accent similarity, speech
quality, and speaker maintenance with only 15 minutes of weakly parallel data
which is not constrained to the same speaker. Extensive experimentation with
diverse accent types suggests that this framework possesses a high degree of
adaptability, making it readily scalable to accommodate other accents with
low-resource data. Audio samples are available at
https://www.microsoft.com/en-us/research/project/convert-and-speak-zero-shot-accent-conversion-with-minimumsupervision/.

### 摘要 (中文)

低资源并行数据是语调转换（AC）问题的关键挑战，该问题要求同时转换发音单元和节奏模式。我们提出了一种生成性框架“转换-说话”，其中只在词级上进行转换，并且在目标方言域中使用语音生成模型合成转换后的词。解耦设计允许“说话”模块使用大量目标方言的语音，从而减轻了“转换”模块所需的平行数据需求。通过桥接词级，也减轻了对文本转录数据的需求，从而进一步高效地减少平行方言语音数据的需要。为了降低“说话”的复杂性和延迟，“单阶段AR生成模型”被设计出来以获得良好的质量以及较低的计算成本。在印度英语到通用美国英语的转换实验中，提出的框架在只有15分钟弱平行数据的情况下，在音调相似度、声音质量和发言人维护方面实现了最先进的性能，这仅限于不相同的发言者。对于多样化的口音类型进行了广泛的经验验证，表明这个框架具有高度的适应性，使得它可以很容易地扩展到容纳其他资源有限的口音。音频样本可在以下链接中访问：
https://www.microsoft.com/en-us/research/project/convert-and-speak-zero-shot-accent-conversion-with-minimumsupervision/

---

## Hear Your Face_ Face-based voice conversion with F0 estimation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09802v1)

### Abstract (English)

This paper delves into the emerging field of face-based voice conversion,
leveraging the unique relationship between an individual's facial features and
their vocal characteristics. We present a novel face-based voice conversion
framework that particularly utilizes the average fundamental frequency of the
target speaker, derived solely from their facial images. Through extensive
analysis, our framework demonstrates superior speech generation quality and the
ability to align facial features with voice characteristics, including tracking
of the target speaker's fundamental frequency.

### 摘要 (中文)

这篇论文深入探讨了基于面部的语音转换这一新兴领域，它利用了个人面部特征与嗓音特点之间独特的联系。我们提出了一种新的基于面部的语音转换框架，特别利用目标演讲者仅从他们的面部图像中提取的平均基频。通过广泛的研究，我们的框架展示了出色的语音生成质量和与面部特征和嗓音特点相匹配的能力，包括跟踪目标演讲者的基频。

---

## Unsupervised Composable Representations for Audio

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09792v1)

### Abstract (English)

Current generative models are able to generate high-quality artefacts but
have been shown to struggle with compositional reasoning, which can be defined
as the ability to generate complex structures from simpler elements. In this
paper, we focus on the problem of compositional representation learning for
music data, specifically targeting the fully-unsupervised setting. We propose a
simple and extensible framework that leverages an explicit compositional
inductive bias, defined by a flexible auto-encoding objective that can leverage
any of the current state-of-art generative models. We demonstrate that our
framework, used with diffusion models, naturally addresses the task of
unsupervised audio source separation, showing that our model is able to perform
high-quality separation. Our findings reveal that our proposal achieves
comparable or superior performance with respect to other blind source
separation methods and, furthermore, it even surpasses current state-of-art
supervised baselines on signal-to-interference ratio metrics. Additionally, by
learning an a-posteriori masking diffusion model in the space of composable
representations, we achieve a system capable of seamlessly performing
unsupervised source separation, unconditional generation, and variation
generation. Finally, as our proposal works in the latent space of pre-trained
neural audio codecs, it also provides a lower computational cost with respect
to other neural baselines.

### 摘要 (中文)

当前的生成模型能够生成高质量的图像，但它们在构建组合推理方面表现不佳，这可以定义为从简单元素中生成复杂结构的能力。在此论文中，我们专注于音乐数据中的组合表示学习问题，具体关注全监督设置。我们提出了一种简单的且可扩展的框架，该框架利用一种明确的组合归纳偏差，由一个灵活的自编码目标定义，该目标可以利用任何当前最先进的生成模型之一。我们演示了使用扩散模型时，我们的框架自然地解决了音频源分离任务，显示我们的模型能够进行高质量的分离。我们的发现揭示了我们的提议与其他盲源分离方法相比，在信噪比指标上具有相当或优于性能，并且它甚至超越了当前最先进的监督基线。此外，通过在可组合表示空间中学习后验掩蔽扩散模型，我们实现了能够在无监督源分离、无条件生成和变异生成中无缝执行的任务系统。最后，由于我们的提议工作在预训练神经音频码流的空间中，因此它也提供了与其他神经基准相比的成本更低的计算成本。

---

## A Markov Random Field Multi-Modal Variational AutoEncoder

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09576v1)

### Abstract (English)

Recent advancements in multimodal Variational AutoEncoders (VAEs) have
highlighted their potential for modeling complex data from multiple modalities.
However, many existing approaches use relatively straightforward aggregating
schemes that may not fully capture the complex dynamics present between
different modalities. This work introduces a novel multimodal VAE that
incorporates a Markov Random Field (MRF) into both the prior and posterior
distributions. This integration aims to capture complex intermodal interactions
more effectively. Unlike previous models, our approach is specifically designed
to model and leverage the intricacies of these relationships, enabling a more
faithful representation of multimodal data. Our experiments demonstrate that
our model performs competitively on the standard PolyMNIST dataset and shows
superior performance in managing complex intermodal dependencies in a specially
designed synthetic dataset, intended to test intricate relationships.

### 摘要 (中文)

最近，多模态变分自编码器（VAEs）的近期进步强调了它们在从多个模态中模拟复杂数据的能力。然而，许多现有方法使用相对简单的聚合方案，这些方案可能无法完全捕捉不同模态之间复杂动态的存在。本工作引入了一个新的多模态VAE，其中MRF被整合到先验和后验分布中。这一集成旨在更有效地捕获不同模态之间的复杂交互关系。与以往模型相比，我们的方法专门设计用于建模和利用这些关系的细节，从而能够更准确地表示多模态数据。我们的实验表明，在标准PolyMNIST数据集上，我们的模型表现出良好的性能，并且在设计特殊合成数据集时显示了优于其他模式的性能，该数据集旨在测试复杂的相互关系。

---

## Convolutional Conditional Neural Processes

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09583v1)

### Abstract (English)

Neural processes are a family of models which use neural networks to directly
parametrise a map from data sets to predictions. Directly parametrising this
map enables the use of expressive neural networks in small-data problems where
neural networks would traditionally overfit. Neural processes can produce
well-calibrated uncertainties, effectively deal with missing data, and are
simple to train. These properties make this family of models appealing for a
breadth of applications areas, such as healthcare or environmental sciences.
  This thesis advances neural processes in three ways.
  First, we propose convolutional neural processes (ConvNPs). ConvNPs improve
data efficiency of neural processes by building in a symmetry called
translation equivariance. ConvNPs rely on convolutional neural networks rather
than multi-layer perceptrons.
  Second, we propose Gaussian neural processes (GNPs). GNPs directly
parametrise dependencies in the predictions of a neural process. Current
approaches to modelling dependencies in the predictions depend on a latent
variable, which consequently requires approximate inference, undermining the
simplicity of the approach.
  Third, we propose autoregressive conditional neural processes (AR CNPs). AR
CNPs train a neural process without any modifications to the model or training
procedure and, at test time, roll out the model in an autoregressive fashion.
AR CNPs equip the neural process framework with a new knob where modelling
complexity and computational expense at training time can be traded for
computational expense at test time.
  In addition to methodological advancements, this thesis also proposes a
software abstraction that enables a compositional approach to implementing
neural processes. This approach allows the user to rapidly explore the space of
neural process models by putting together elementary building blocks in
different ways.

### 摘要 (中文)

神经过程是一种使用神经网络直接参数化数据集到预测的映射模型的家庭。直接参数化这个映射使得在小数据问题中使用表达性神经网络成为可能，这些神经网络通常会过拟合。神经过程可以产生良好的置信度，有效处理缺失数据，并且训练简单。这种模型家族对各种应用领域有吸引力，例如医疗保健或环境科学。

本论文通过三种方式推进了神经过程。首先，我们提出了卷积神经过程（ConvNPs）。ConvNPs通过引入一个称为等变性的对称来提高神经过程的数据效率。ConvNPs依赖于卷积神经网络而不是多层感知器。其次，我们提出了高斯神经过程（GNPs）。GNPs直接参数化神经过程中的预测依赖关系。当前用于建模预测依赖关系的方法依赖于隐变量，因此需要近似推断，这削弱了这种方法的简洁性。第三，我们提出了自动相关条件神经过程（AR CNPs）。AR CNPs在没有修改模型或训练程序的情况下，在测试时间以自回归的方式展开模型。AR CNPs使神经过程框架获得了新的调节器，即在训练时间上的复杂性和计算开销可以在测试时间上进行交换。

除了方法学进步外，本论文还提出了软件抽象，该抽象允许用户通过组合不同的元素构建不同的神经过程模型来进行组件式实施。这种方法允许用户快速探索神经过程模型的空间，通过将不同类型的单元组合在一起来实现这一点。

---

## Contextual Bandits for Unbounded Context Distributions

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09655v1)

### Abstract (English)

Nonparametric contextual bandit is an important model of sequential decision
making problems. Under $\alpha$-Tsybakov margin condition, existing research
has established a regret bound of
$\tilde{O}\left(T^{1-\frac{\alpha+1}{d+2}}\right)$ for bounded supports.
However, the optimal regret with unbounded contexts has not been analyzed. The
challenge of solving contextual bandit problems with unbounded support is to
achieve both exploration-exploitation tradeoff and bias-variance tradeoff
simultaneously. In this paper, we solve the nonparametric contextual bandit
problem with unbounded contexts. We propose two nearest neighbor methods
combined with UCB exploration. The first method uses a fixed $k$. Our analysis
shows that this method achieves minimax optimal regret under a weak margin
condition and relatively light-tailed context distributions. The second method
uses adaptive $k$. By a proper data-driven selection of $k$, this method
achieves an expected regret of
$\tilde{O}\left(T^{1-\frac{(\alpha+1)\beta}{\alpha+(d+2)\beta}}+T^{1-\beta}\right)$,
in which $\beta$ is a parameter describing the tail strength. This bound
matches the minimax lower bound up to logarithm factors, indicating that the
second method is approximately optimal.

### 摘要 (中文)

非参数上下文贪心策略是序列决策问题的一个重要模型。在$\alpha$-Tsybakov边界条件下，已有的研究已经建立了支持集为有限的支持的贪心策略的最大后悔值为$\tilde{O}(T^{1-\frac{\alpha+1}{d+2}})$的上界。然而，无界支持的最优后悔没有被分析过。解决有无界支持的上下文贪心策略的挑战在于同时实现探索与利用的权衡以及正态分布和偏度之间的权衡。本论文解决了有无界的上下文贪心策略的问题。我们提出了两种近邻方法结合UCB探索。第一种方法使用固定$k$。我们的分析表明，在弱边界的条件下，这种方法实现了在弱边界的最小化后悔，并且相对尾部分布较轻。第二种方法使用自适应$k$。通过适当的数据驱动选择$k$，这个方法期望的最大后悔为$\tilde{O}(T^{1-\frac{(\alpha+1)\beta}{\alpha+(d+2)\beta}}+T^{1-\beta})$，其中$\beta$描述了尾部强度。这个下界匹配了最小化下的对数因子，表明第二方法大约是最优的。

---

## Towards Few-Shot Learning in the Open World_ A Review and Beyond

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09722v1)

### Abstract (English)

Human intelligence is characterized by our ability to absorb and apply
knowledge from the world around us, especially in rapidly acquiring new
concepts from minimal examples, underpinned by prior knowledge. Few-shot
learning (FSL) aims to mimic this capacity by enabling significant
generalizations and transferability. However, traditional FSL frameworks often
rely on assumptions of clean, complete, and static data, conditions that are
seldom met in real-world environments. Such assumptions falter in the
inherently uncertain, incomplete, and dynamic contexts of the open world. This
paper presents a comprehensive review of recent advancements designed to adapt
FSL for use in open-world settings. We categorize existing methods into three
distinct types of open-world few-shot learning: those involving varying
instances, varying classes, and varying distributions. Each category is
discussed in terms of its specific challenges and methods, as well as its
strengths and weaknesses. We standardize experimental settings and metric
benchmarks across scenarios, and provide a comparative analysis of the
performance of various methods. In conclusion, we outline potential future
research directions for this evolving field. It is our hope that this review
will catalyze further development of effective solutions to these complex
challenges, thereby advancing the field of artificial intelligence.

### 摘要 (中文)

人类智能的特征是我们在周围环境中吸收和应用知识的能力，尤其是从少量示例中迅速获得新的概念。快速学习（FSL）旨在通过使我们能够显著泛化并转移能力来模仿这一能力，从而使我们可以从世界周围的知识中吸收和应用知识。然而，传统的FSL框架通常依赖于干净、完整和静态数据的假设条件，这些条件在现实世界的环境中很少被满足。这种假设条件在开放世界环境中的不确定性、不完备性和动态性方面失败了。本文提出了一个综合的回顾，以适应开放式环境的方法设计用于使用开放式环境。我们将现有的方法分为三种不同的类型：涉及不同实例、不同类别的开放式少样本学习；以及涉及不同分布的开放式少样本学习。每种类别都根据其特定挑战和方法进行了讨论，并且还讨论了它们的优势和劣势。我们标准化了实验设置和指标基准场景，并提供了各种方法性能的比较分析。总的来说，我们概述了该不断演变的领域潜在的未来研究方向。我们认为，这项审查将催化进一步开发有效解决方案的潜力，从而推动人工智能领域的进步。

---

## ALTBI_ Constructing Improved Outlier Detection Models via Optimization of Inlier-Memorization Effect

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09791v1)

### Abstract (English)

Outlier detection (OD) is the task of identifying unusual observations (or
outliers) from a given or upcoming data by learning unique patterns of normal
observations (or inliers). Recently, a study introduced a powerful unsupervised
OD (UOD) solver based on a new observation of deep generative models, called
inlier-memorization (IM) effect, which suggests that generative models memorize
inliers before outliers in early learning stages. In this study, we aim to
develop a theoretically principled method to address UOD tasks by maximally
utilizing the IM effect. We begin by observing that the IM effect is observed
more clearly when the given training data contain fewer outliers. This finding
indicates a potential for enhancing the IM effect in UOD regimes if we can
effectively exclude outliers from mini-batches when designing the loss
function. To this end, we introduce two main techniques: 1) increasing the
mini-batch size as the model training proceeds and 2) using an adaptive
threshold to calculate the truncated loss function. We theoretically show that
these two techniques effectively filter out outliers from the truncated loss
function, allowing us to utilize the IM effect to the fullest. Coupled with an
additional ensemble strategy, we propose our method and term it Adaptive Loss
Truncation with Batch Increment (ALTBI). We provide extensive experimental
results to demonstrate that ALTBI achieves state-of-the-art performance in
identifying outliers compared to other recent methods, even with significantly
lower computation costs. Additionally, we show that our method yields robust
performances when combined with privacy-preserving algorithms.

### 摘要 (中文)

异常检测（OD）是根据给定或即将要的数据学习异常观察（或异常点）的独特模式的任务。最近，一项研究引入了一种强大的无监督OD（UOD）求解器，基于深度生成模型的新观察，称为内存效应（IM），指出在早期学习阶段，生成模型会记住正样本（或正常观测）之前的所有负样本（或异常观测）。在此研究中，我们的目标是通过充分利用IM效果来开发一个理论上原则的方法来解决UOD任务。我们首先观察到，在给定的训练数据中包含较少的异常值时，IM效应更清晰。这一发现表明，如果我们能够有效地从损失函数的设计中排除异常值，那么可以有效提高IM效应。为此，我们引入了两个主要技术：1）随着模型训练的进行，增加小批量大小；2）使用自适应阈值计算截断损失函数。理论证明表明，这两种方法有效地过滤出了截断损失函数中的异常值，允许我们将IM效应发挥到极致。此外，结合额外的并行策略，我们提出了我们的方法，并将其命名为Adaptive Loss Truncation with Batch Increment（ALTBI）。我们提供了广泛的实验结果来展示，与近期其他方法相比，ALTBI在识别异常点方面取得了最先进的性能，即使在计算成本显著较低的情况下也是如此。此外，我们还展示了在与其他隐私保护算法相结合时，我们的方法具有良好的鲁棒性。

---

## Area under the ROC Curve has the Most Consistent Evaluation for Binary Classification

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10193v1)

### Abstract (English)

Evaluation Metrics is an important question for model evaluation and model
selection in binary classification tasks. This study investigates how
consistent metrics are at evaluating different models under different data
scenarios. Analyzing over 150 data scenarios and 18 model evaluation metrics
using statistical simulation, I find that for binary classification tasks,
evaluation metrics that are less influenced by prevalence offer more consistent
ranking of a set of different models. In particular, Area Under the ROC Curve
(AUC) has smallest variance in ranking of different models. Matthew's
correlation coefficient as a more strict measure of model performance has the
second smallest variance. These patterns holds across a rich set of data
scenarios and five commonly used machine learning models as well as a naive
random guess model. The results have significant implications for model
evaluation and model selection in binary classification tasks.

### 摘要 (中文)

在二分类任务中，模型评估和选择的问题是一个重要的问题。本研究探讨了不同数据场景下不同的指标如何在评价不同模型时保持一致性。通过统计模拟分析超过150个数据场景和18种模型评估指标，我发现对于二分类任务，影响预后较少的评估指标能够更有效地对一组不同模型进行排名。例如，AUC排序模型的表现差异最小。Matthew相关系数作为更加严格的模型性能衡量标准，其排序偏差也最小。这些模式在整个丰富的数据场景以及五种常见的机器学习模型（包括随机猜测模型）上都存在。研究结果具有重大的意义，特别是在二分类任务中的模型评估和选择中。

---

## Panorama Tomosynthesis from Head CBCT with Simulated Projection Geometry

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09358v1)

### Abstract (English)

Cone Beam Computed Tomography (CBCT) and Panoramic X-rays are the most
commonly used imaging modalities in dental health care. CBCT can produce
three-dimensional views of a patient's head, providing clinicians with better
diagnostic capability, whereas Panoramic X-ray can capture the entire
maxillofacial region in a single image. If the CBCT is already available, it
can be beneficial to synthesize a Panoramic X-ray, thereby avoiding an
immediate additional scan and extra radiation exposure. Existing methods focus
on delineating an approximate dental arch and creating orthogonal projections
along this arch. However, no golden standard is available for such dental arch
extractions, and this choice can affect the quality of synthesized X-rays. To
avoid such issues, we propose a novel method for synthesizing Panoramic X-rays
from diverse head CBCTs, employing a simulated projection geometry and dynamic
rotation centers. Our method effectively synthesized panoramic views from CBCT,
even for patients with missing or nonexistent teeth and in the presence of
severe metal implants. Our results demonstrate that this method can generate
high-quality panoramic images irrespective of the CBCT scanner geometry.

### 摘要 (中文)

锥形束计算机断层摄影（CBCT）和全景X光片是牙科健康护理中最常用的成像技术。CBCT可以生成患者头部的三维图像，提供临床医生更好的诊断能力，而全景X光片可以在一张照片中捕捉整个颅面部区域。如果已经存在CBCT，通过合成全景X光片可以避免立即进行额外扫描和额外辐射暴露。现有的方法集中在描绘大致的牙齿排列并创建沿此排列的投影。然而，并没有适用于此类牙齿排布的黄金标准，这一选择会直接影响合成X光片的质量。为了避免这些问题，我们提出了一种从多种头部CBCT合成全景X光片的新方法，采用模拟投影几何和动态旋转中心。我们的方法有效地从CBCT合成全景图，即使对于缺失或不存在的牙齿以及存在严重金属植入体的患者也是如此。我们的结果表明，这种方法能够生成不受CBCT扫描器几何影响的高质量全景图像。

---

## Obtaining Optimal Spiking Neural Network in Sequence Learning via CRNN-SNN Conversion

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09403v1)

### Abstract (English)

Spiking neural networks (SNNs) are becoming a promising alternative to
conventional artificial neural networks (ANNs) due to their rich neural
dynamics and the implementation of energy-efficient neuromorphic chips.
However, the non-differential binary communication mechanism makes SNN hard to
converge to an ANN-level accuracy. When SNN encounters sequence learning, the
situation becomes worse due to the difficulties in modeling long-range
dependencies. To overcome these difficulties, researchers developed variants of
LIF neurons and different surrogate gradients but still failed to obtain good
results when the sequence became longer (e.g., $>$500). Unlike them, we obtain
an optimal SNN in sequence learning by directly mapping parameters from a
quantized CRNN. We design two sub-pipelines to support the end-to-end
conversion of different structures in neural networks, which is called
CNN-Morph (CNN $\rightarrow$ QCNN $\rightarrow$ BIFSNN) and RNN-Morph (RNN
$\rightarrow$ QRNN $\rightarrow$ RBIFSNN). Using conversion pipelines and the
s-analog encoding method, the conversion error of our framework is zero.
Furthermore, we give the theoretical and experimental demonstration of the
lossless CRNN-SNN conversion. Our results show the effectiveness of our method
over short and long timescales tasks compared with the state-of-the-art
learning- and conversion-based methods. We reach the highest accuracy of 99.16%
(0.46 $\uparrow$) on S-MNIST, 94.95% (3.95 $\uparrow$) on PS-MNIST (sequence
length of 784) respectively, and the lowest loss of 0.057 (0.013 $\downarrow$)
within 8 time-steps in collision avoidance dataset.

### 摘要 (中文)

跳变神经网络（SPINNs）因其丰富的神经动力学和实现低能耗的神经元微片而成为传统人工神经网络（ANNs）的一个有前景的选择。然而，非等差二进制通信机制使得SPINNs很难收敛到ANN级别的精度。当序列学习时，情况变得更糟，因为长距离依赖性建模的困难。为了克服这些困难，研究人员开发了LIF神经元的变体以及不同的预估梯度但仍然无法在序列变得更长的情况下获得良好的结果（例如，>500）。相反，我们通过直接从量化CRNN映射参数来优化序列学习中的SPINNs，在序列学习中获得了最佳的SPINNs。我们设计两个子管道支持端到端结构的不同转换，并命名为CNN-Morph（CNN $\rightarrow$ QCNN $\rightarrow$ BIFSNN）和RNN-Morph（RNN $\rightarrow$ QRNN $\rightarrow$ RBIFSNN），使用转换管道和s-模拟编码方法，我们的框架的转换误差为零。此外，我们还给出了损失不损失的CRNN-SPINN转换理论和实验演示。我们的结果显示，与基于学习和转换的方法相比，我们的方法在短时间和长期任务上表现出有效性。分别在S-MNIST上达到最高准确率为99.16%（0.46$\uparrow$），PS-MNIST上的准确率分别为94.95%（3.95$\uparrow$），并且在碰撞避让数据集中的损失为0.057（0.013$\downarrow$）内达到了最低值。

注：本文未提供原文链接或参考文献。

---

## A Robust Algorithm for Contactless Fingerprint Enhancement and Matching

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09426v1)

### Abstract (English)

Compared to contact fingerprint images, contactless fingerprint images
exhibit four distinct characteristics: (1) they contain less noise; (2) they
have fewer discontinuities in ridge patterns; (3) the ridge-valley pattern is
less distinct; and (4) they pose an interoperability problem, as they lack the
elastic deformation caused by pressing the finger against the capture device.
These properties present significant challenges for the enhancement of
contactless fingerprint images. In this study, we propose a novel contactless
fingerprint identification solution that enhances the accuracy of minutiae
detection through improved frequency estimation and a new region-quality-based
minutia extraction algorithm. In addition, we introduce an efficient and highly
accurate minutiae-based encoding and matching algorithm. We validate the
effectiveness of our approach through extensive experimental testing. Our
method achieves a minimum Equal Error Rate (EER) of 2.84\% on the PolyU
contactless fingerprint dataset, demonstrating its superior performance
compared to existing state-of-the-art techniques. The proposed fingerprint
identification method exhibits notable precision and resilience, proving to be
an effective and feasible solution for contactless fingerprint-based
identification systems.

### 摘要 (中文)

与接触式指纹图像相比，非接触式指纹图像具有四个显著特性：（1）噪声较少；（2）在沟槽模式中出现的断续性较少；（3）沟槽和山谷图案更加模糊；（4）存在互操作性问题，因为没有因压指装置而产生的弹性变形。这些属性对提高非接触式指纹图像的增强提出了重大挑战。本研究提出了一种新型的非接触式指纹识别解决方案，通过改进频率估计和基于区域质量的新微小特征提取算法来提高微小特征检测的准确性。此外，我们引入了一个高效的高准确度基于微小特征编码匹配算法。我们通过大量的实验验证了该方法的有效性。我们的方法在PolyU非接触式指纹数据集上的最小错误率（EER）为2.84%，显示了其优于现有最先进的技术的性能。提出的指纹识别方法表现出显著的精确性和韧性，证明是用于非接触式指纹识别系统的一个有效可行的解决方案。

---

## MedMAP_ Promoting Incomplete Multi-modal Brain Tumor Segmentation with Alignment

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09465v1)

### Abstract (English)

Brain tumor segmentation is often based on multiple magnetic resonance
imaging (MRI). However, in clinical practice, certain modalities of MRI may be
missing, which presents a more difficult scenario. To cope with this challenge,
Knowledge Distillation, Domain Adaption, and Shared Latent Space have emerged
as commonly promising strategies. However, recent efforts typically overlook
the modality gaps and thus fail to learn important invariant feature
representations across different modalities. Such drawback consequently leads
to limited performance for missing modality models. To ameliorate these
problems, pre-trained models are used in natural visual segmentation tasks to
minimize the gaps. However, promising pre-trained models are often unavailable
in medical image segmentation tasks. Along this line, in this paper, we propose
a novel paradigm that aligns latent features of involved modalities to a
well-defined distribution anchor as the substitution of the pre-trained model}.
As a major contribution, we prove that our novel training paradigm ensures a
tight evidence lower bound, thus theoretically certifying its effectiveness.
Extensive experiments on different backbones validate that the proposed
paradigm can enable invariant feature representations and produce models with
narrowed modality gaps. Models with our alignment paradigm show their superior
performance on both BraTS2018 and BraTS2020 datasets.

### 摘要 (中文)

脑肿瘤分割通常基于多模态磁共振成像（MRI）。然而，在临床实践中，某些模态的MRI可能缺失，这提出了一种更为困难的情况。为了应对这一挑战，知识蒸馏、领域适应和共享隐空间作为常见有希望的方法脱颖而出。然而，最近的努力往往忽视了模态差距，因此无法学习不同模态的重要不变特征表示。这种缺点最终导致缺失模态模型的表现受限。为了缓解这些问题，预训练模型被用于自然视觉分割任务来最小化差距。然而，有价值的预训练模型在医学图像分割任务中往往是不可用的。沿着这条线，本文提出了一个新颖的范式，将参与模态的相关性转换到定义良好的分布锚点作为预训练模型的替代品。作为主要贡献，我们证明了我们的新型训练范式确保了一个严格的证据下界，从而理论上认证其有效性。对不同后端验证表明，提出的范式可以促进不变特征表示，并产生具有狭窄模态差距的模型。使用我们的对齐范式的模型在BraTS2018和BraTS2020数据集上表现出优于两者的表现。

---

## MambaLoc_ Efficient Camera Localisation via State Space Model

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09680v1)

### Abstract (English)

Location information is pivotal for the automation and intelligence of
terminal devices and edge-cloud IoT systems, such as autonomous vehicles and
augmented reality. However, achieving reliable positioning across diverse IoT
applications remains challenging due to significant training costs and the
necessity of densely collected data. To tackle these issues, we have
innovatively applied the selective state space (SSM) model to visual
localization, introducing a new model named MambaLoc. The proposed model
demonstrates exceptional training efficiency by capitalizing on the SSM model's
strengths in efficient feature extraction, rapid computation, and memory
optimization, and it further ensures robustness in sparse data environments due
to its parameter sparsity. Additionally, we propose the Global Information
Selector (GIS), which leverages selective SSM to implicitly achieve the
efficient global feature extraction capabilities of Non-local Neural Networks.
This design leverages the computational efficiency of the SSM model alongside
the Non-local Neural Networks' capacity to capture long-range dependencies with
minimal layers. Consequently, the GIS enables effective global information
capture while significantly accelerating convergence. Our extensive
experimental validation using public indoor and outdoor datasets first
demonstrates our model's effectiveness, followed by evidence of its versatility
with various existing localization models.

### 摘要 (中文)

终端设备和边缘云物联网系统，如自动驾驶车辆和增强现实，对于自动化和智能至关重要。然而，由于大量的培训成本和密集收集数据的必要性，实现跨多种IoT应用的有效定位仍然是一个挑战。为此，我们创新地应用了选择状态空间（SSM）模型到视觉本地化中，引入了一个新的模型名为MambaLoc。该提出的模型通过利用SSM模型在高效特征提取、快速计算和内存优化方面的优势，展示了异常高效的训练效率，并且在稀疏数据环境中增强了鲁棒性，因为参数稠密。此外，我们提出了一种全局信息选择器（GIS），它利用选择SSM来隐式实现非局部神经网络的高效全局特征提取能力。这种设计利用SSM模型的计算效率以及非局部神经网络捕捉长距离依赖的能力所需的最小层数之间的结合。因此，GIS能够在有效全球信息捕获的同时显著加速收敛。我们在公共室内和室外数据集上进行的广泛实验验证首先展示了我们的模型的有效性，然后提供了各种现有定位模型的多样性的证据。

---

## Mutually-Aware Feature Learning for Few-Shot Object Counting

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09734v1)

### Abstract (English)

Few-shot object counting has garnered significant attention for its
practicality as it aims to count target objects in a query image based on given
exemplars without the need for additional training. However, there is a
shortcoming in the prevailing extract-and-match approach: query and exemplar
features lack interaction during feature extraction since they are extracted
unaware of each other and later correlated based on similarity. This can lead
to insufficient target awareness of the extracted features, resulting in target
confusion in precisely identifying the actual target when multiple class
objects coexist. To address this limitation, we propose a novel framework,
Mutually-Aware FEAture learning(MAFEA), which encodes query and exemplar
features mutually aware of each other from the outset. By encouraging
interaction between query and exemplar features throughout the entire pipeline,
we can obtain target-aware features that are robust to a multi-category
scenario. Furthermore, we introduce a background token to effectively associate
the target region of query with exemplars and decouple its background region
from them. Our extensive experiments demonstrate that our model reaches a new
state-of-the-art performance on the two challenging benchmarks, FSCD-LVIS and
FSC-147, with a remarkably reduced degree of the target confusion problem.

### 摘要 (中文)

少数样本目标计数因其实用性而受到广泛关注，因为它旨在基于给定示例图像中目标对象的数量，在查询图像中根据给出的示例对象不需额外训练来计数目标对象。然而，当前提取和匹配方法的一个局限性在于，由于在特征提取过程中没有意识到彼此的存在，因此在后续相似度关联时缺乏交互。这可能导致从提取的特征中无法获得足够的目标意识，从而导致在存在多个类别对象的情况下精确识别实际目标时目标混淆的问题。为解决这一限制，我们提出了一种新颖的方法，即互相关注特征学习（MAFEA），该方法在一开始就编码出相互之间感知的查询和示例特征。通过在整个管道中鼓励查询和示例特征之间的互动，我们可以获得具有抗多类别场景能力的目标意识特征。此外，我们还引入了一个背景令牌，有效地将查询中的目标区域与示例联系起来，并将其背景区域与其分离开来。我们的大量实验表明，我们的模型在两个挑战性的基准数据集上达到了新的状态- 243和FSC-147上的新性能水平，同时显著减少了目标混淆问题的程度。

---

## Enhanced Cascade Prostate Cancer Classifier in mp-MRI Utilizing Recall Feedback Adaptive Loss and Prior Knowledge-Based Feature Extraction

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09746v1)

### Abstract (English)

Prostate cancer is the second most common cancer in males worldwide, and
mpMRI is commonly used for diagnosis. However, interpreting mpMRI is
challenging and requires expertise from radiologists. This highlights the
urgent need for automated grading in mpMRI. Existing studies lack integration
of clinical prior information and suffer from uneven training sample
distribution due to prevalence. Therefore, we propose a solution that
incorporates prior knowledge, addresses the issue of uneven medical sample
distribution, and maintains high interpretability in mpMRI. Firstly, we
introduce Prior Knowledge-Based Feature Extraction, which mathematically models
the PI-RADS criteria for prostate cancer as diagnostic information into model
training. Secondly, we propose Adaptive Recall Feedback Loss to address the
extremely imbalanced data problem. This method adjusts the training dynamically
based on accuracy and recall in the validation set, resulting in high accuracy
and recall simultaneously in the testing set.Thirdly, we design an Enhanced
Cascade Prostate Cancer Classifier that classifies prostate cancer into
different levels in an interpretable way, which refines the classification
results and helps with clinical intervention. Our method is validated through
experiments on the PI-CAI dataset and outperforms other methods with a more
balanced result in both accuracy and recall rate.

### 摘要 (中文)

前列腺癌是全球男性中最常见的癌症之一，而MPMRI通常用于诊断。然而，解读MPMRI具有挑战性，并需要放射科医生的专业知识。这凸显了在MPMRI中自动评级的迫切需求。现有的研究缺乏临床先验信息的整合，并且由于流行率不均等原因遭受不平衡训练样本分布。因此，我们提出了一种解决方案，该方案融合了先验知识，解决了数据不平衡的问题，并保持了在MPMRI中的可解释性。首先，我们引入基于先验知识的特征提取，将PI-RADS标准模型转化为模型训练中的诊断信息数学上表示出来。其次，我们提出了适应召回反馈损失来解决极度失衡的数据问题。这种方法根据验证集的准确性和召回率动态调整训练，同时在测试集中的准确性与召回率同时得到提高。第三，我们设计了一个增强的多级前列腺癌分类器，这是一种以可解释的方式对前列腺癌进行分类的方法，这可以进一步改进分类结果，并有助于临床干预。我们的方法通过在PI-CAI数据集上的实验进行了验证，并且在准确率和召回率方面都超过了其他方法。

---

## Segment-Anything Models Achieve Zero-shot Robustness in Autonomous Driving

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09839v1)

### Abstract (English)

Semantic segmentation is a significant perception task in autonomous driving.
It suffers from the risks of adversarial examples. In the past few years, deep
learning has gradually transitioned from convolutional neural network (CNN)
models with a relatively small number of parameters to foundation models with a
huge number of parameters. The segment-anything model (SAM) is a generalized
image segmentation framework that is capable of handling various types of
images and is able to recognize and segment arbitrary objects in an image
without the need to train on a specific object. It is a unified model that can
handle diverse downstream tasks, including semantic segmentation, object
detection, and tracking. In the task of semantic segmentation for autonomous
driving, it is significant to study the zero-shot adversarial robustness of
SAM. Therefore, we deliver a systematic empirical study on the robustness of
SAM without additional training. Based on the experimental results, the
zero-shot adversarial robustness of the SAM under the black-box corruptions and
white-box adversarial attacks is acceptable, even without the need for
additional training. The finding of this study is insightful in that the
gigantic model parameters and huge amounts of training data lead to the
phenomenon of emergence, which builds a guarantee of adversarial robustness.
SAM is a vision foundation model that can be regarded as an early prototype of
an artificial general intelligence (AGI) pipeline. In such a pipeline, a
unified model can handle diverse tasks. Therefore, this research not only
inspects the impact of vision foundation models on safe autonomous driving but
also provides a perspective on developing trustworthy AGI. The code is
available at: https://github.com/momo1986/robust_sam_iv.

### 摘要 (中文)

语义分割是自动驾驶中一个重要的感知任务。它面临着对抗性示例的风险。近年来，深度学习逐渐从具有相对较少参数的卷积神经网络（CNN）模型过渡到参数数量巨大的基础模型。段式任何模型（SAM）是一种通用图像分割框架，能够处理各种类型的照片，并能够在不需在特定对象上进行训练的情况下识别和分割图像中的任意物体。它是可以处理多样化的下游任务的统一模型，包括语义分割、目标检测和跟踪等。对于自主驾驶中的语义分割而言，研究零样本对抗鲁棒性的SAM至关重要。因此，我们进行了基于实验结果的系统经验研究，以评估在黑盒污染和白盒对抗攻击下SAM的鲁棒性，即使不需要额外的训练。这项研究发现是有启发性的，庞大的模型参数和大量的训练数据导致了出现现象，这建立了一种对抗鲁棒性的保证。SAM是一种视图基础模型，可以被视为一种早期的AI管道的原型。在这种管道中，统一模型可以处理多种任务。因此，这项研究不仅关注视觉基础模型对安全自主驾驶的影响，还提供了一个发展可信赖的AI的视角。代码可在https://github.com/momo1986/robust_sam_iv中找到。

---

## Caption-Driven Explorations_ Aligning Image and Text Embeddings through Human-Inspired Foveated Vision

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09948v1)

### Abstract (English)

Understanding human attention is crucial for vision science and AI. While
many models exist for free-viewing, less is known about task-driven image
exploration. To address this, we introduce CapMIT1003, a dataset with captions
and click-contingent image explorations, to study human attention during the
captioning task. We also present NevaClip, a zero-shot method for predicting
visual scanpaths by combining CLIP models with NeVA algorithms. NevaClip
generates fixations to align the representations of foveated visual stimuli and
captions. The simulated scanpaths outperform existing human attention models in
plausibility for captioning and free-viewing tasks. This research enhances the
understanding of human attention and advances scanpath prediction models.

### 摘要 (中文)

了解人类注意力对于视觉科学和人工智能至关重要。虽然许多用于自由观看的模型存在，但关于任务驱动的图像探索知之甚少。为了应对这一挑战，我们引入了CapMIT1003，一个包含注释和点击相关的图像探索数据集，以研究在提词任务中的人类注意力。我们还提出了NevaClip，一种零样本方法，通过结合CLIP模型和NeVA算法来预测视觉扫描路径。NevaClip使用聚焦视网膜视觉刺激和提词的表示进行配对。模拟的扫描路径在提词和自由观看任务上都优于现有的人类注意力模型的可信度。这项研究表明了人类注意力的理解，并推进了扫路预测模型的发展。

---

## FFAA_ Multimodal Large Language Model based Explainable Open-World Face Forgery Analysis Assistant

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10072v1)

### Abstract (English)

The rapid advancement of deepfake technologies has sparked widespread public
concern, particularly as face forgery poses a serious threat to public
information security. However, the unknown and diverse forgery techniques,
varied facial features and complex environmental factors pose significant
challenges for face forgery analysis. Existing datasets lack descriptions of
these aspects, making it difficult for models to distinguish between real and
forged faces using only visual information amid various confounding factors. In
addition, existing methods do not yield user-friendly and explainable results,
complicating the understanding of the model's decision-making process. To
address these challenges, we introduce a novel Open-World Face Forgery Analysis
VQA (OW-FFA-VQA) task and the corresponding benchmark. To tackle this task, we
first establish a dataset featuring a diverse collection of real and forged
face images with essential descriptions and reliable forgery reasoning. Base on
this dataset, we introduce FFAA: Face Forgery Analysis Assistant, consisting of
a fine-tuned Multimodal Large Language Model (MLLM) and Multi-answer
Intelligent Decision System (MIDS). By integrating hypothetical prompts with
MIDS, the impact of fuzzy classification boundaries is effectively mitigated,
enhancing the model's robustness. Extensive experiments demonstrate that our
method not only provides user-friendly explainable results but also
significantly boosts accuracy and robustness compared to previous methods.

### 摘要 (中文)

随着深度伪造技术的迅速发展，引发了公众广泛的担忧，特别是面部伪造对公共信息安全构成了严重威胁。然而，未知和多样的伪造技巧、各种复杂的环境因素以及面部特征多样性等挑战性问题给面部伪造分析带来了巨大的困难。现有的数据集缺乏这些方面的描述，使得在多种混淆因素的影响下，仅凭视觉信息便无法通过模型区分真实与伪造人脸变得非常困难。此外，现有方法难以提供用户友好的且可解释的结果，这又进一步阻碍了理解模型决策过程的理解。为了应对这些问题，我们引入了一个新的开放世界面部伪造分析视觉问答（Open-World Face Forgery Analysis VQA, OW-FFA-VQA）任务及其对应的基准。为解决这一任务，首先建立了包含多样化的现实和伪造人脸图像的丰富数据集，并提供了关键描述和可靠伪造推理的数据。基于此数据集，我们推出了Face Forgery Analysis Assistant（FFAA），由一个预训练的多模态大型语言模型（MLLLM）和一个多答案智能决策系统（MIDS）组成。通过结合假设性的提示与MIDS，有效降低了模糊分类边界的影响，增强了模型的鲁棒性。大量的实验表明，我们的方法不仅提供了用户友好且易于解释的结果，而且显著提高了准确性和鲁棒性，比之前的方法有明显的提升。

---

## Factorized-Dreamer_ Training A High-Quality Video Generator with Limited and Low-Quality Data

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10119v1)

### Abstract (English)

Text-to-video (T2V) generation has gained significant attention due to its
wide applications to video generation, editing, enhancement and translation,
\etc. However, high-quality (HQ) video synthesis is extremely challenging
because of the diverse and complex motions existed in real world. Most existing
works struggle to address this problem by collecting large-scale HQ videos,
which are inaccessible to the community. In this work, we show that publicly
available limited and low-quality (LQ) data are sufficient to train a HQ video
generator without recaptioning or finetuning. We factorize the whole T2V
generation process into two steps: generating an image conditioned on a highly
descriptive caption, and synthesizing the video conditioned on the generated
image and a concise caption of motion details. Specifically, we present
\emph{Factorized-Dreamer}, a factorized spatiotemporal framework with several
critical designs for T2V generation, including an adapter to combine text and
image embeddings, a pixel-aware cross attention module to capture pixel-level
image information, a T5 text encoder to better understand motion description,
and a PredictNet to supervise optical flows. We further present a noise
schedule, which plays a key role in ensuring the quality and stability of video
generation. Our model lowers the requirements in detailed captions and HQ
videos, and can be directly trained on limited LQ datasets with noisy and brief
captions such as WebVid-10M, largely alleviating the cost to collect
large-scale HQ video-text pairs. Extensive experiments in a variety of T2V and
image-to-video generation tasks demonstrate the effectiveness of our proposed
Factorized-Dreamer. Our source codes are available at
\url{https://github.com/yangxy/Factorized-Dreamer/}.

### 摘要 (中文)

文本到视频(T2V)生成因其在视频生成、编辑、增强和翻译等广泛应用而备受关注。然而，高质量（HQ）视频合成极其挑战性，因为真实世界中存在多种复杂的运动。大多数现有工作通过收集大量高品质视频来解决这个问题，这些视频对社区来说是不可访问的。在这项工作中，我们展示了公开可用的有限低质量（LQ）数据足以训练一个高质量视频生成器而不需注释或微调。我们将整个T2V生成过程分解为两个步骤：基于高度描述性的标题生成图像，以及根据生成的图像和运动细节简短的摘要合成视频。具体而言，我们提出了一种名为Factorized-Dreamer的分形框架，它具有几个关键设计用于T2V生成，包括将文本和图像嵌入组合成适应器，以捕获像素级图像信息；一种跨像素注意力模块，以捕捉图像中的像素级别信息；一种T5文本编码器，更好地理解动作描述；以及一种监督光流预测的预言网络。此外，我们还提出了一个噪声调度方案，这在确保视频生成的质量和稳定性方面起着关键作用。我们的模型降低了详细标题和高质量视频的需求，并可以直接在有限的LQ数据集上进行训练，如WebVid-10M中的噪音和简短注释，大大缓解了收集大规模高清视频对文本-视频对的要求。在各种T2V和图像至视频生成任务的广泛实验中，证明了我们提出的Factorized-Dreamer的有效性。我们的源代码可在此处找到：https://github.com/yangxy/Factorized-Dreamer/。

---

## Fairness Under Cover_ Evaluating the Impact of Occlusions on Demographic Bias in Facial Recognition

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10175v1)

### Abstract (English)

This study investigates the effects of occlusions on the fairness of face
recognition systems, particularly focusing on demographic biases. Using the
Racial Faces in the Wild (RFW) dataset and synthetically added realistic
occlusions, we evaluate their effect on the performance of face recognition
models trained on the BUPT-Balanced and BUPT-GlobalFace datasets. We note
increases in the dispersion of FMR, FNMR, and accuracy alongside decreases in
fairness according to Equilized Odds, Demographic Parity, STD of Accuracy, and
Fairness Discrepancy Rate. Additionally, we utilize a pixel attribution method
to understand the importance of occlusions in model predictions, proposing a
new metric, Face Occlusion Impact Ratio (FOIR), that quantifies the extent to
which occlusions affect model performance across different demographic groups.
Our results indicate that occlusions exacerbate existing demographic biases,
with models placing higher importance on occlusions in an unequal fashion,
particularly affecting African individuals more severely.

### 摘要 (中文)

这项研究调查了遮挡对面部识别系统公平性的影响，尤其是关注人口统计学偏见。使用Racial Faces in the Wild（RFW）数据集和合成的现实主义遮挡，我们评估了训练于BUPT-Balanced和BUPT-GlobalFace数据集上的模型在面部识别中的性能。我们注意到FMR、FNMR以及准确率水平随等效机会、人口平等度、标准误差、不公平差距比率的增加而下降。此外，我们利用像素着色方法来理解遮挡在模型预测中的重要性，并提出了一个新的指标Face Occlusion Impact Ratio（FOIR），该指标量化了不同人口群体中遮挡对模型性能的影响程度。我们的结果表明，遮挡加剧了现有的人口统计学偏见，特别是在不均等的方式下，模型特别着重于遮挡，尤其是在非洲人身上，影响更加严重。

---

## NeuRodin_ A Two-stage Framework for High-Fidelity Neural Surface Reconstruction

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10178v1)

### Abstract (English)

Signed Distance Function (SDF)-based volume rendering has demonstrated
significant capabilities in surface reconstruction. Although promising,
SDF-based methods often fail to capture detailed geometric structures,
resulting in visible defects. By comparing SDF-based volume rendering to
density-based volume rendering, we identify two main factors within the
SDF-based approach that degrade surface quality: SDF-to-density representation
and geometric regularization. These factors introduce challenges that hinder
the optimization of the SDF field. To address these issues, we introduce
NeuRodin, a novel two-stage neural surface reconstruction framework that not
only achieves high-fidelity surface reconstruction but also retains the
flexible optimization characteristics of density-based methods. NeuRodin
incorporates innovative strategies that facilitate transformation of arbitrary
topologies and reduce artifacts associated with density bias. Extensive
evaluations on the Tanks and Temples and ScanNet++ datasets demonstrate the
superiority of NeuRodin, showing strong reconstruction capabilities for both
indoor and outdoor environments using solely posed RGB captures. Project
website: https://open3dvlab.github.io/NeuRodin/

### 摘要 (中文)

基于签名距离函数（SDF）的体积渲染在表面重建方面显示出显著的能力。虽然有希望，但基于SDF的方法经常无法捕获详细的几何结构，导致可见缺陷。通过比较基于密度的体积渲染与基于SDF的方法，我们识别了SDF基于方法中两个主要因素，这些因素降低了表面质量：SDF到密度代表和几何正规化。这些因素引入了挑战，阻碍了SDF场的优化。为了应对这些问题，我们引入了一个新的两阶段神经表面重构框架，即NeuRodin，该框架不仅实现了高保真表面重建，还保留了基于密度方法灵活优化特性的灵活性。NeuRodin结合了创新策略，使任意拓扑得以转换，并减少密度偏见相关的瑕疵。对Tanks和Temples以及ScanNet++数据集进行的广泛评估表明，NeuRodin显示出了优于NeuRodin的优势，仅使用姿势RGB捕捉即可用于室内和室外环境的高质量重建。项目网站：https://open3dvlab.github.io/NeuRodin/

---

## Imbalance-Aware Culvert-Sewer Defect Segmentation Using an Enhanced Feature Pyramid Network

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10181v1)

### Abstract (English)

Imbalanced datasets are a significant challenge in real-world scenarios. They
lead to models that underperform on underrepresented classes, which is a
critical issue in infrastructure inspection. This paper introduces the Enhanced
Feature Pyramid Network (E-FPN), a deep learning model for the semantic
segmentation of culverts and sewer pipes within imbalanced datasets. The E-FPN
incorporates architectural innovations like sparsely connected blocks and
depth-wise separable convolutions to improve feature extraction and handle
object variations. To address dataset imbalance, the model employs strategies
like class decomposition and data augmentation. Experimental results on the
culvert-sewer defects dataset and a benchmark aerial semantic segmentation
drone dataset show that the E-FPN outperforms state-of-the-art methods,
achieving an average Intersection over Union (IoU) improvement of 13.8% and
27.2%, respectively. Additionally, class decomposition and data augmentation
together boost the model's performance by approximately 6.9% IoU. The proposed
E-FPN presents a promising solution for enhancing object segmentation in
challenging, multi-class real-world datasets, with potential applications
extending beyond culvert-sewer defect detection.

### 摘要 (中文)

在实际应用场景中，不平衡数据集是一个重大挑战。它们导致模型在未被代表的类上表现不佳，这是基础设施检查中的一个关键问题。本文介绍了增强特征金字塔网络（E-FPN），这是一种用于处理具有不平衡数据集内的涵洞和下水道管道语义分割的深度学习模型。E-FPN采用了如稀疏连接块和深度可分离卷积等架构创新来改进特征提取并处理对象变异。为了应对数据不平衡，该模型采用策略如类别分解和数据增强。在涵洞-下水道缺陷检测数据集及其基准无人机语义分割图像数据集上的实验结果显示，E-FPN优于最先进的方法，实现了平均交并比（IoU）提升分别为13.8%和27.2%。此外，通过结合类别分解和数据增强，模型的性能提高了大约6.9%的IoU。提出的E-FPN对于提高挑战性、多类现实世界数据集中的物体分割提供了有希望的解决方案，其潜在应用范围可能超出涵洞-下水道缺陷检测。

---

## Learning Fair Invariant Representations under Covariate and Correlation Shifts Simultaneously

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09312v1)

### Abstract (English)

Achieving the generalization of an invariant classifier from training domains
to shifted test domains while simultaneously considering model fairness is a
substantial and complex challenge in machine learning. Existing methods address
the problem of fairness-aware domain generalization, focusing on either
covariate shift or correlation shift, but rarely consider both at the same
time. In this paper, we introduce a novel approach that focuses on learning a
fairness-aware domain-invariant predictor within a framework addressing both
covariate and correlation shifts simultaneously, ensuring its generalization to
unknown test domains inaccessible during training. In our approach, data are
first disentangled into content and style factors in latent spaces.
Furthermore, fairness-aware domain-invariant content representations can be
learned by mitigating sensitive information and retaining as much other
information as possible. Extensive empirical studies on benchmark datasets
demonstrate that our approach surpasses state-of-the-art methods with respect
to model accuracy as well as both group and individual fairness.

### 摘要 (中文)

从训练域到偏移测试域实现不变分类器的一般化是一个巨大的复杂挑战在机器学习中。现有的方法专注于公平性意识的领域一般化，侧重于两者之一：变量移动或相关性移动，但很少同时考虑两者。本论文提出了一种新的方法，该方法集中在框架内，在学习一个公平性意识的领域不变预测器的同时，考虑到变量和相关性移动，并确保其对未知测试域的通用化不受训练期间的数据不可访问的影响。我们的方法首先将数据解离成内容和风格因素在隐空间中。此外，通过消除敏感信息并尽可能保留其他信息来学习公平性的内容表示可以被学习。对于基准数据集进行的广泛实验证明了我们方法在模型准确性以及群体和个体公正方面超越了最先进的方法。

---

## E-CGL_ An Efficient Continual Graph Learner

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09350v1)

### Abstract (English)

Continual learning has emerged as a crucial paradigm for learning from
sequential data while preserving previous knowledge. In the realm of continual
graph learning, where graphs continuously evolve based on streaming graph data,
continual graph learning presents unique challenges that require adaptive and
efficient graph learning methods in addition to the problem of catastrophic
forgetting. The first challenge arises from the interdependencies between
different graph data, where previous graphs can influence new data
distributions. The second challenge lies in the efficiency concern when dealing
with large graphs. To addresses these two problems, we produce an Efficient
Continual Graph Learner (E-CGL) in this paper. We tackle the interdependencies
issue by demonstrating the effectiveness of replay strategies and introducing a
combined sampling strategy that considers both node importance and diversity.
To overcome the limitation of efficiency, E-CGL leverages a simple yet
effective MLP model that shares weights with a GCN during training, achieving
acceleration by circumventing the computationally expensive message passing
process. Our method comprehensively surpasses nine baselines on four graph
continual learning datasets under two settings, meanwhile E-CGL largely reduces
the catastrophic forgetting problem down to an average of -1.1%. Additionally,
E-CGL achieves an average of 15.83x training time acceleration and 4.89x
inference time acceleration across the four datasets. These results indicate
that E-CGL not only effectively manages the correlation between different graph
data during continual training but also enhances the efficiency of continual
learning on large graphs. The code is publicly available at
https://github.com/aubreygjh/E-CGL.

### 摘要 (中文)

持续学习已成为从连续数据中学习的关键范式，同时保留先前知识。在持续图学习的领域，随着流式图数据的不断演变，持续图学习提出了独特的挑战，这些挑战需要适应和高效的图学习方法来解决，而不仅仅是灾难性遗忘的问题。首先，问题在于不同图之间的依赖关系，其中以前的图可以影响新的分布。第二个问题是处理大型图时效率的关注。为了应对这两个问题，我们在本文中生产了一个高效持续图学习者（E-CGL）。我们通过展示回放策略的有效性，并引入一个综合采样策略，该策略考虑了节点的重要性以及多样性来解决依赖关系问题。为了解决效率限制，E-CGL利用一种简单但有效的MLP模型，在训练过程中与GCN共享权重，从而实现加速，绕过计算昂贵的消息传递过程。我们的方法全面超越了四个图持续学习数据集中的九个基准线，在两个设置下，E-CGL在四张数据集中降低了平均灾难性遗忘问题至-1.1%。此外，E-CGL实现了四个数据集的平均培训时间加速和推理时间加速分别为15.83倍和4.89倍。这些结果表明，E-CGL不仅有效地管理了不同图数据之间的相关性，而且增强了大图上持续学习的效率。代码已公开发布在 https://github.com/aubreygjh/E-CGL 上。

---

## GraphSPNs_ Sum-Product Networks Benefit From Canonical Orderings

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09451v1)

### Abstract (English)

Deep generative models have recently made a remarkable progress in capturing
complex probability distributions over graphs. However, they are intractable
and thus unable to answer even the most basic probabilistic inference queries
without resorting to approximations. Therefore, we propose graph sum-product
networks (GraphSPNs), a tractable deep generative model which provides exact
and efficient inference over (arbitrary parts of) graphs. We investigate
different principles to make SPNs permutation invariant. We demonstrate that
GraphSPNs are able to (conditionally) generate novel and chemically valid
molecular graphs, being competitive to, and sometimes even better than,
existing intractable models. We find out that (Graph)SPNs benefit from ensuring
the permutation invariance via canonical ordering.

### 摘要 (中文)

最近，深度生成模型在模拟图的概率分布方面取得了显著的进步。然而，它们是不可计算的，因此无法通过近似方法回答即使是最基本的概率推理查询。因此，我们提出了一个可计算的深度生成模型——图加权和乘积网络（GraphSPNs），它能够精确有效地在任意部分上对（任意部分）进行推理。我们研究了不同的原则来使SPNs保持正交不变性。我们发现，GraphSPNs能够在条件下生成新颖且化学有效的分子图，甚至比现有的不可计算模型还要好。我们发现，确保正交性的可以利用常数排序受益于GraphSPNs。

---

## Leveraging Invariant Principle for Heterophilic Graph Structure Distribution Shifts

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09490v1)

### Abstract (English)

Heterophilic Graph Neural Networks (HGNNs) have shown promising results for
semi-supervised learning tasks on graphs. Notably, most real-world heterophilic
graphs are composed of a mixture of nodes with different neighbor patterns,
exhibiting local node-level homophilic and heterophilic structures. However,
existing works are only devoted to designing better HGNN backbones or
architectures for node classification tasks on heterophilic and homophilic
graph benchmarks simultaneously, and their analyses of HGNN performance with
respect to nodes are only based on the determined data distribution without
exploring the effect caused by this structural difference between training and
testing nodes. How to learn invariant node representations on heterophilic
graphs to handle this structure difference or distribution shifts remains
unexplored. In this paper, we first discuss the limitations of previous
graph-based invariant learning methods from the perspective of data
augmentation. Then, we propose \textbf{HEI}, a framework capable of generating
invariant node representations through incorporating heterophily information to
infer latent environments without augmentation, which are then used for
invariant prediction, under heterophilic graph structure distribution shifts.
We theoretically show that our proposed method can achieve guaranteed
performance under heterophilic graph structure distribution shifts. Extensive
experiments on various benchmarks and backbones can also demonstrate the
effectiveness of our method compared with existing state-of-the-art baselines.

### 摘要 (中文)

异质性图神经网络（HGNN）在半监督学习任务上表现出色，特别是在由具有不同邻接模式的节点组成的混合图中。值得注意的是，大多数现实世界的异质性图是由具有不同邻接模式的不同节点组成的，显示出局部节点级别的同质性和异质性结构。然而，现有的工作仅专注于设计更优秀的HGNN骨架或架构来同时处理异质性和同质性图谱基准上的节点分类任务，其对节点性能的分析基于确定的数据分布，而没有探索训练和测试节点之间这种结构差异的影响。如何在异质性图中学习不变性的节点表示以处理这一结构差异或数据漂移仍是一个未被探索的问题。本文首先从数据增强的角度讨论了以往基于图的不变学习方法存在的局限性。然后，我们提出了\textbf{HEI}框架，该框架通过融合异质性信息推断隐含环境，从而生成不变的节点表示，这些表示用于变异性图结构变化预测，在异质性图结构分布的变化下。理论上证明，我们的提出的方法可以在异质性图结构分布的变化下实现保证的表现。广泛的实验结果表明，与现有最先进的基线相比，我们在各种基准和骨架上比较有效。

---

## Addressing Heterogeneity in Federated Learning_ Challenges and Solutions for a Shared Production Environment

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09556v1)

### Abstract (English)

Federated learning (FL) has emerged as a promising approach to training
machine learning models across decentralized data sources while preserving data
privacy, particularly in manufacturing and shared production environments.
However, the presence of data heterogeneity variations in data distribution,
quality, and volume across different or clients and production sites, poses
significant challenges to the effectiveness and efficiency of FL. This paper
provides a comprehensive overview of heterogeneity in FL within the context of
manufacturing, detailing the types and sources of heterogeneity, including
non-independent and identically distributed (non-IID) data, unbalanced data,
variable data quality, and statistical heterogeneity. We discuss the impact of
these types of heterogeneity on model training and review current methodologies
for mitigating their adverse effects. These methodologies include personalized
and customized models, robust aggregation techniques, and client selection
techniques. By synthesizing existing research and proposing new strategies,
this paper aims to provide insight for effectively managing data heterogeneity
in FL, enhancing model robustness, and ensuring fair and efficient training
across diverse environments. Future research directions are also identified,
highlighting the need for adaptive and scalable solutions to further improve
the FL paradigm in the context of Industry 4.0.

### 摘要 (中文)

分布式学习（FL）作为一种在分散的数据源上训练机器学习模型的有效方法，已经在制造业和共享生产环境中脱颖而出。然而，在数据分布、质量和数量等不同或客户端和生产站点之间的异质性变化中，存在大量的挑战，这极大地影响了FL的效能和效率。

本论文提供了关于制造环境中FL异质性的全面概述，详细介绍了不同类型和来源的异质性，包括非独立同分布（非IID）数据、失衡数据、变量质量不均等，并讨论了这些类型异质性对模型训练的影响。我们还回顾了当前针对消除其不利影响的方法，如个性化和定制化的模型、稳健聚合技术以及客户选择策略。

通过综合现有研究并提出新的策略，本文旨在提供在FL中管理数据异质性的见解，增强模型的鲁棒性，并确保公平且高效的跨环境培训。未来的研究方向也得到了识别，强调了需要进一步改进以适应工业4.0背景下FL范式的需求。

---

## AdaResNet_ Enhancing Residual Networks with Dynamic Weight Adjustment for Improved Feature Integration

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09958v1)

### Abstract (English)

In very deep neural networks, gradients can become extremely small during
backpropagation, making it challenging to train the early layers. ResNet
(Residual Network) addresses this issue by enabling gradients to flow directly
through the network via skip connections, facilitating the training of much
deeper networks. However, in these skip connections, the input ipd is directly
added to the transformed data tfd, treating ipd and tfd equally, without
adapting to different scenarios. In this paper, we propose AdaResNet
(Auto-Adapting Residual Network), which automatically adjusts the ratio between
ipd and tfd based on the training data. We introduce a variable,
weight}_{tfd}^{ipd, to represent this ratio. This variable is dynamically
adjusted during backpropagation, allowing it to adapt to the training data
rather than remaining fixed. Experimental results demonstrate that AdaResNet
achieves a maximum accuracy improvement of over 50\% compared to traditional
ResNet.

### 摘要 (中文)

在非常深的神经网络中，梯度在反向传播时可能会变得非常小，这使得早期层的训练变得困难。ResNet（残差网络）通过使梯度直接通过网络中的跳接来流过解决这一问题，有助于训练更深的网络。然而，在这些跳接中，输入ipd直接添加到经过变换的数据tfd上，将其视为相同，而没有根据不同的场景进行调整。在这篇论文中，我们提出AdaResNet（自动适应性残差网络），它可以根据训练数据自动调整ipd和tfd之间的比例。我们将这个比值表示为变量weight}_{tfd}^{ipd,。这个变量在反向传播过程中动态调整，允许其根据训练数据而不是保持固定地适应。实验结果表明，AdaResNet与传统ResNet相比可以实现超过50\%的最大准确性提升。

---

## PLUTUS_ A Well Pre-trained Large Unified Transformer can Unveil Financial Time Series Regularities

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10111v1)

### Abstract (English)

Financial time series modeling is crucial for understanding and predicting
market behaviors but faces challenges such as non-linearity, non-stationarity,
and high noise levels. Traditional models struggle to capture complex patterns
due to these issues, compounded by limitations in computational resources and
model capacity. Inspired by the success of large language models in NLP, we
introduce \textbf{PLUTUS}, a \textbf{P}re-trained \textbf{L}arge
\textbf{U}nified \textbf{T}ransformer-based model that \textbf{U}nveils
regularities in financial time \textbf{S}eries. PLUTUS uses an invertible
embedding module with contrastive learning and autoencoder techniques to create
an approximate one-to-one mapping between raw data and patch embeddings.
TimeFormer, an attention based architecture, forms the core of PLUTUS,
effectively modeling high-noise time series. We incorporate a novel attention
mechanisms to capture features across both variable and temporal dimensions.
PLUTUS is pre-trained on an unprecedented dataset of 100 billion observations,
designed to thrive in noisy financial environments. To our knowledge, PLUTUS is
the first open-source, large-scale, pre-trained financial time series model
with over one billion parameters. It achieves state-of-the-art performance in
various tasks, demonstrating strong transferability and establishing a robust
foundational model for finance. Our research provides technical guidance for
pre-training financial time series data, setting a new standard in the field.

### 摘要 (中文)

金融时间序列建模对于理解并预测市场行为至关重要，但面临着非线性、非平稳性和高噪声等挑战。传统的模型由于这些问题而无法捕捉复杂模式，同时受限于计算资源和模型容量的限制。我们受到自然语言处理（NLP）中大型语言模型成功的启发，引入了名为PLUTUS的预训练统一变换器基模型。PLUTUS使用逆向嵌入模块与对抗学习和自编码技术相结合，创建了一个对原始数据和片段嵌入的有效近似一对一映射。TimeFormer是一种注意力架构，是PLUTUS的核心，有效地模拟高噪音时间序列。我们引入了一种新颖的注意力机制来捕获变量和时间维度上的特征之间的跨越。PLUTUS在前所未有的100亿观测数据集上进行了预训练，旨在适应嘈杂的金融市场环境。到目前为止，PLUTUS是我们所知的第一个开源大规模预训练的金融时间序列模型，参数超过一亿。它在各种任务中表现出色，展示了强大的迁移能力和建立金融领域的稳健基础模型的能力。我们的研究为预训练金融时间序列数据提供了技术支持，并为该领域设定了一个新的标准。

---

## SMILE_ Zero-Shot Sparse Mixture of Low-Rank Experts Construction From Pre-Trained Foundation Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10174v1)

### Abstract (English)

Deep model training on extensive datasets is increasingly becoming
cost-prohibitive, prompting the widespread adoption of deep model fusion
techniques to leverage knowledge from pre-existing models. From simple weight
averaging to more sophisticated methods like AdaMerging, model fusion
effectively improves model performance and accelerates the development of new
models. However, potential interference between parameters of individual models
and the lack of interpretability in the fusion progress remain significant
challenges. Existing methods often try to resolve the parameter interference
issue by evaluating attributes of parameters, such as their magnitude or sign,
or by parameter pruning. In this study, we begin by examining the fine-tuning
of linear layers through the lens of subspace analysis and explicitly define
parameter interference as an optimization problem to shed light on this
subject. Subsequently, we introduce an innovative approach to model fusion
called zero-shot Sparse MIxture of Low-rank Experts (SMILE) construction, which
allows for the upscaling of source models into an MoE model without extra data
or further training. Our approach relies on the observation that fine-tuning
mostly keeps the important parts from the pre-training, but it uses less
significant or unused areas to adapt to new tasks. Also, the issue of parameter
interference, which is intrinsically intractable in the original parameter
space, can be managed by expanding the dimensions. We conduct extensive
experiments across diverse scenarios, such as image classification and text
generalization tasks, using full fine-tuning and LoRA fine-tuning, and we apply
our method to large language models (CLIP models, Flan-T5 models, and
Mistral-7B models), highlighting the adaptability and scalability of SMILE.
Code is available at https://github.com/tanganke/fusion_bench

### 摘要 (中文)

深度模型在大量数据上的训练变得越来越成本高昂，因此越来越多地采用了融合技术来利用现有模型的知识。从简单的权重平均到更复杂的AdaMerging等方法，模型融合有效地提高了模型性能，并加速了新模型的开发。然而，单个模型参数之间的干扰以及融合过程中解释性的缺乏仍然是重要的挑战。现有的方法通常通过评估参数属性（例如其大小或符号）或参数剪枝来解决这个问题。在这项研究中，我们首先以子空间分析的角度审视线性层微调，明确参数干扰是一个优化问题，以此来探讨这一主题。随后，我们引入了一个创新的方法来实现深度模型的零样本稀疏混合低秩专家（SMILE）构建，该方法允许通过不额外的数据或进一步训练将源模型升级为MoE模型。我们的方法依赖于观察到微调主要保留了预训练中的重要部分，但使用了对任务更为次要或未使用的区域进行适应。此外，在原始参数空间中存在内在不可处理的参数干扰问题可以通过扩展维度来管理。我们在不同的场景，如图像分类和文本一般化任务中进行了广泛的实验，使用全微调和LoRA微调，并将其应用于大型语言模型（CLIP模型、Flan-T5模型和Mistral-7B模型），强调了SMILE的可适应性和可伸缩性。代码可在https://github.com/tanganke/fusion_bench上找到。

---

## Transformers to SSMs_ Distilling Quadratic Knowledge to Subquadratic Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10189v1)

### Abstract (English)

Transformer architectures have become a dominant paradigm for domains like
language modeling but suffer in many inference settings due to their
quadratic-time self-attention. Recently proposed subquadratic architectures,
such as Mamba, have shown promise, but have been pretrained with substantially
less computational resources than the strongest Transformer models. In this
work, we present a method that is able to distill a pretrained Transformer
architecture into alternative architectures such as state space models (SSMs).
The key idea to our approach is that we can view both Transformers and SSMs as
applying different forms of mixing matrices over the token sequences. We can
thus progressively distill the Transformer architecture by matching different
degrees of granularity in the SSM: first matching the mixing matrices
themselves, then the hidden units at each block, and finally the end-to-end
predictions. Our method, called MOHAWK, is able to distill a Mamba-2 variant
based on the Phi-1.5 architecture (Phi-Mamba) using only 3B tokens and a hybrid
version (Hybrid Phi-Mamba) using 5B tokens. Despite using less than 1% of the
training data typically used to train models from scratch, Phi-Mamba boasts
substantially stronger performance compared to all past open-source
non-Transformer models. MOHAWK allows models like SSMs to leverage
computational resources invested in training Transformer-based architectures,
highlighting a new avenue for building such models.

### 摘要 (中文)

Transformer架构在语言建模等领域取得了主导地位，但其自注意力的时间复杂度较高，导致很多推理设置下表现不佳。最近提出的亚平方时间架构，如Mamba，显示出了希望，但训练资源却比最强的Transformer模型少得多。本工作提出了一种能够从预训练的Transformer架构中提炼出其他架构（如状态空间模型）的方法。我们方法的关键思想是，我们可以把Transformer和状态空间模型（SSMs）看作是在不同的混合矩阵上对词序列进行不同形式的混入。因此，我们可以通过匹配SSM中的不同粒度来逐步提取Transformer架构。我们的方法被称为MOHAWK，它仅使用了30亿个token和混合版本（Hybrid Phi-Mamba），就能根据Phi-1.5架构（Phi-Mamba）提炼出Mamba-2变体，并且比所有开源非Transformer模型都表现出更强性能。这表明SSMs可以利用训练Transformer架构时投入的计算资源，开辟了一条新的构建此类模型的道路。

---

## Attention Is Not What You Need_ Revisiting Multi-Instance Learning for Whole Slide Image Classification

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09449v1)

### Abstract (English)

Although attention-based multi-instance learning algorithms have achieved
impressive performances on slide-level whole slide image (WSI) classification
tasks, they are prone to mistakenly focus on irrelevant patterns such as
staining conditions and tissue morphology, leading to incorrect patch-level
predictions and unreliable interpretability. Moreover, these attention-based
MIL algorithms tend to focus on salient instances and struggle to recognize
hard-to-classify instances. In this paper, we first demonstrate that
attention-based WSI classification methods do not adhere to the standard MIL
assumptions. From the standard MIL assumptions, we propose a surprisingly
simple yet effective instance-based MIL method for WSI classification
(FocusMIL) based on max-pooling and forward amortized variational inference. We
argue that synergizing the standard MIL assumption with variational inference
encourages the model to focus on tumour morphology instead of spurious
correlations. Our experimental evaluations show that FocusMIL significantly
outperforms the baselines in patch-level classification tasks on the Camelyon16
and TCGA-NSCLC benchmarks. Visualization results show that our method also
achieves better classification boundaries for identifying hard instances and
mitigates the effect of spurious correlations between bags and labels.

### 摘要 (中文)

虽然基于注意力的多实例学习算法在全切片图像（WSI）分类任务上取得了令人印象深刻的性能，但它们容易误将无关模式（如染色条件和组织形态学）聚焦，导致错误的区域级预测和不可靠解释。此外，基于注意力的MIL算法倾向于关注突出的实例，并且很难识别难以分类的实例。本论文首先展示了基于注意力的WSI分类方法并不遵循标准MIL假设。从标准MIL假设出发，我们提出了一个看似简单而有效的实例级MIL方法（FocusMIL），基于最大池化和前向变分信息推断。我们认为结合标准MIL假设与变分信息鼓励模型专注于肿瘤形态而不是假相关的联系。我们的实验评估显示，在Camelyon16和TCGA-NSCLC基准上的区域级分类任务中，FocusMIL明显优于基线。可视化结果表明，我们的方法也能够更好地区分硬实例，并减轻标签和袋子之间的虚假相关性的影响。

---

## Advances in Multiple Instance Learning for Whole Slide Image Analysis_ Techniques_ Challenges_ and Future Directions

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09476v1)

### Abstract (English)

Whole slide images (WSIs) are gigapixel-scale digital images of H\&E-stained
tissue samples widely used in pathology. The substantial size and complexity of
WSIs pose unique analytical challenges. Multiple Instance Learning (MIL) has
emerged as a powerful approach for addressing these challenges, particularly in
cancer classification and detection. This survey provides a comprehensive
overview of the challenges and methodologies associated with applying MIL to
WSI analysis, including attention mechanisms, pseudo-labeling, transformers,
pooling functions, and graph neural networks. Additionally, it explores the
potential of MIL in discovering cancer cell morphology, constructing
interpretable machine learning models, and quantifying cancer grading. By
summarizing the current challenges, methodologies, and potential applications
of MIL in WSI analysis, this survey aims to inform researchers about the state
of the field and inspire future research directions.

### 摘要 (中文)

全切片图像（WSI）是广泛用于病理学的高分辨率数字组织样本。WSI的大量尺寸和复杂性提出了独特的分析挑战。多实例学习（MIL）已成为解决这些问题的强大方法，尤其是在癌症分类和检测中尤为突出。这项调查提供了在应用MIL到WSI分析中的综合概述，包括注意力机制、伪标签化、Transformer、池化函数和图神经网络等。此外，它还探索了MIL在发现癌症细胞形态、构建可解释的机器学习模型以及量化癌症分级方面的潜力。通过总结当前WSI分析中的挑战、方法和潜在应用，本调查旨在告知研究人员该领域的现状，并激发未来的研究方向。

---

## Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09929v1)

### Abstract (English)

Inspired by the idea of Positive-incentive Noise (Pi-Noise or $\pi$-Noise)
that aims at learning the reliable noise beneficial to tasks, we scientifically
investigate the connection between contrastive learning and $\pi$-noise in this
paper. By converting the contrastive loss to an auxiliary Gaussian distribution
to quantitatively measure the difficulty of the specific contrastive model
under the information theory framework, we properly define the task entropy,
the core concept of $\pi$-noise, of contrastive learning. It is further proved
that the predefined data augmentation in the standard contrastive learning
paradigm can be regarded as a kind of point estimation of $\pi$-noise. Inspired
by the theoretical study, a framework that develops a $\pi$-noise generator to
learn the beneficial noise (instead of estimation) as data augmentations for
contrast is proposed. The designed framework can be applied to diverse types of
data and is also completely compatible with the existing contrastive models.
From the visualization, we surprisingly find that the proposed method
successfully learns effective augmentations.

### 摘要 (中文)

被积极激励噪声（Pi-Noise或$\pi$-Noise）的想法启发，该研究旨在学习对任务有益的可靠噪声。在这一论文中，我们科学地探讨了对比学习与$\pi$-噪音之间的联系。通过将其对抗损失转换为辅助高斯分布来定量测量特定对比模型下信息理论框架下的具体对比模型难度，我们正确定义了对比学习的核心概念$\pi$-噪音的任务熵，这是$\pi$-噪音的概念。进一步证明，在标准对比学习范式中预先定义的数据增强可以被视为一种点估计$\pi$-噪音。基于理论研究，提出了一种发展$\pi$-噪音生成器以学习有益噪声（而不是估计）作为数据增强对比的方法。设计的框架可以在各种类型的数据上应用，并且也完全兼容现有对比模型。从可视化中，令人惊讶的是，提出的这种方法成功地学习到了有效的增强。

---

## Exploiting Fine-Grained Prototype Distribution for Boosting Unsupervised Class Incremental Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10046v1)

### Abstract (English)

The dynamic nature of open-world scenarios has attracted more attention to
class incremental learning (CIL). However, existing CIL methods typically
presume the availability of complete ground-truth labels throughout the
training process, an assumption rarely met in practical applications.
Consequently, this paper explores a more challenging problem of unsupervised
class incremental learning (UCIL). The essence of addressing this problem lies
in effectively capturing comprehensive feature representations and discovering
unknown novel classes. To achieve this, we first model the knowledge of class
distribution by exploiting fine-grained prototypes. Subsequently, a granularity
alignment technique is introduced to enhance the unsupervised class discovery.
Additionally, we proposed a strategy to minimize overlap between novel and
existing classes, thereby preserving historical knowledge and mitigating the
phenomenon of catastrophic forgetting. Extensive experiments on the five
datasets demonstrate that our approach significantly outperforms current
state-of-the-art methods, indicating the effectiveness of the proposed method.

### 摘要 (中文)

开放世界场景的动态性吸引了对类增量学习（CIL）的关注。然而，现有的CIL方法通常假设在整个训练过程中都有完整的标注正确标签，而在实践中这种假设很少遇到。因此，本研究探索了一个更具有挑战性的未监督类增量学习（UCIL）问题。解决这一问题的关键在于有效地捕获全面特征表示和发现未知新类别。为此，我们首先利用细粒度原型来利用知识分类。随后，引入了粒度匹配技术以增强无监督类发现。此外，我们提出了一种策略来最小化新旧类别的重叠，从而保留历史知识并缓解遗忘现象。在五个数据集上进行的广泛实验表明，我们的方法显著优于当前最先进的方法，这表明所提出的算法的有效性。

---

## Criticality Leveraged Adversarial Training _CLAT_ for Boosted Performance via Parameter Efficiency

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10204v1)

### Abstract (English)

Adversarial training enhances neural network robustness but suffers from a
tendency to overfit and increased generalization errors on clean data. This
work introduces CLAT, an innovative approach that mitigates adversarial
overfitting by introducing parameter efficiency into the adversarial training
process, improving both clean accuracy and adversarial robustness. Instead of
tuning the entire model, CLAT identifies and fine-tunes robustness-critical
layers - those predominantly learning non-robust features - while freezing the
remaining model to enhance robustness. It employs dynamic critical layer
selection to adapt to changes in layer criticality throughout the fine-tuning
process. Empirically, CLAT can be applied on top of existing adversarial
training methods, significantly reduces the number of trainable parameters by
approximately 95%, and achieves more than a 2% improvement in adversarial
robustness compared to baseline methods.

### 摘要 (中文)

对抗性训练增强了神经网络的鲁棒性，但存在过度拟合和在干净数据上增加泛化错误的趋势。本文提出了一种创新的方法CLAT，通过引入参数效率到对抗性训练过程中来缓解对抗性过拟合，同时提高了清洁准确率和对抗性鲁棒性。与现有对抗性训练方法相比，CLAT不调整个模型，而是识别并调整关键层——主要学习非鲁棒特征的那些——同时冻结剩余模型以增强鲁棒性。它采用动态关键层选择来根据精细调整过程中的层重要性的变化进行适应。实证研究表明，CLAT可以在现有的对抗性训练方法之上应用，显著减少了大约95%可训练参数的数量，并且比基准方法取得了超过2%的对抗性鲁棒性改进。

---

## Exploratory Optimal Stopping_ A Singular Control Formulation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09335v1)

### Abstract (English)

This paper explores continuous-time and state-space optimal stopping problems
from a reinforcement learning perspective. We begin by formulating the stopping
problem using randomized stopping times, where the decision maker's control is
represented by the probability of stopping within a given time--specifically, a
bounded, non-decreasing, c\`adl\`ag control process. To encourage exploration
and facilitate learning, we introduce a regularized version of the problem by
penalizing it with the cumulative residual entropy of the randomized stopping
time. The regularized problem takes the form of an (n+1)-dimensional degenerate
singular stochastic control with finite-fuel. We address this through the
dynamic programming principle, which enables us to identify the unique optimal
exploratory strategy. For the specific case of a real option problem, we derive
a semi-explicit solution to the regularized problem, allowing us to assess the
impact of entropy regularization and analyze the vanishing entropy limit.
Finally, we propose a reinforcement learning algorithm based on policy
iteration. We show both policy improvement and policy convergence results for
our proposed algorithm.

### 摘要 (中文)

这篇论文从强化学习的角度探讨了连续时间和状态空间最优停机问题。我们首先通过随机停止时间的形式，用概率来表示决策者的控制，其中控制过程代表在给定的时间内停止的几率，具体来说是一个非负、递增且c\`adl\`ag的概率控制过程。为了鼓励探索并促进学习，我们将问题进行规范化处理，通过惩罚随机停止时间累积残余熵来限制其行为。规范化后的问题是（n+1）维等价于有限燃料的降阶奇异动量控制。我们通过动态规划原则来进行分析，从而确定唯一有效的探索策略。对于特定的真实期权问题，我们导出了对规范化问题的半解析解，允许我们评估熵规正的影响，并分析消失熵极限。最后，我们基于政策迭代提出了一种基于政策优化的学习算法。我们展示了我们的提议算法的政策改进和政策收敛结果。

---

## Mutual Information Multinomial Estimation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09377v1)

### Abstract (English)

Estimating mutual information (MI) is a fundamental yet challenging task in
data science and machine learning. This work proposes a new estimator for
mutual information. Our main discovery is that a preliminary estimate of the
data distribution can dramatically help estimate. This preliminary estimate
serves as a bridge between the joint and the marginal distribution, and by
comparing with this bridge distribution we can easily obtain the true
difference between the joint distributions and the marginal distributions.
Experiments on diverse tasks including non-Gaussian synthetic problems with
known ground-truth and real-world applications demonstrate the advantages of
our method.

### 摘要 (中文)

估计互信息（MI）是数据科学和机器学习中一个基础且具有挑战性的任务。本工作提出了一种新的互信息估计器。我们的主要发现是，对数据分布的初步估计可以极大地帮助估计。这种初步估计充当了联合分布和边际分布之间的桥梁，并通过比较这个桥接分布，我们可以很容易地获得联合分布与边际分布之间的真实差异。在包括已知真值的非高斯合成问题在内的多种任务上进行实验，展示了我们方法的优点。

---

## Deep Limit Model-free Prediction in Regression

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09532v1)

### Abstract (English)

In this paper, we provide a novel Model-free approach based on Deep Neural
Network (DNN) to accomplish point prediction and prediction interval under a
general regression setting. Usually, people rely on parametric or
non-parametric models to bridge dependent and independent variables (Y and X).
However, this classical method relies heavily on the correct model
specification. Even for the non-parametric approach, some additive form is
often assumed. A newly proposed Model-free prediction principle sheds light on
a prediction procedure without any model assumption. Previous work regarding
this principle has shown better performance than other standard alternatives.
Recently, DNN, one of the machine learning methods, has received increasing
attention due to its great performance in practice. Guided by the Model-free
prediction idea, we attempt to apply a fully connected forward DNN to map X and
some appropriate reference random variable Z to Y. The targeted DNN is trained
by minimizing a specially designed loss function so that the randomness of Y
conditional on X is outsourced to Z through the trained DNN. Our method is more
stable and accurate compared to other DNN-based counterparts, especially for
optimal point predictions. With a specific prediction procedure, our prediction
interval can capture the estimation variability so that it can render a better
coverage rate for finite sample cases. The superior performance of our method
is verified by simulation and empirical studies.

### 摘要 (中文)

在本文中，我们提供了一种基于深度神经网络（DNN）的新方法，用于完成一般回归设置下的点预测和预测区间。通常，人们依赖参数模型或非参数模型来连接相关和独立变量（Y和X）。然而，这种方法依赖于正确的模型假设。即使对于非参数方法，也经常假设有加性形式。提出的一种新的无模型预测原理揭示了一个无需任何模型假设的预测程序。有关这一原理的先前工作显示出其性能优于其他标准替代方案。最近，作为机器学习方法之一的深度神经网络（DNN）因其在实践中取得的巨大成功而受到越来越多的关注。受无模型预测想法的指导，我们尝试使用全连接前向DNN将X和一些适当的参考随机变量Z映射到Y。目标DNN通过最小化特设损失函数进行训练，以将随机变量Y的条件依赖于X外包给Z，从而通过训练的DNN。与基于DNN的其他方法相比，我们的方法更稳定、准确，尤其是在最优点预测方面尤其如此。通过特定的预测程序，我们的预测区间可以捕获估计变异性，从而使有限样本案例的覆盖率更好。我们的方法优越性的性能经由仿真和实证研究得到验证。

---

## Sample-Optimal Large-Scale Optimal Subset Selection

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09537v1)

### Abstract (English)

Ranking and selection (R&S) conventionally aims to select the unique best
alternative with the largest mean performance from a finite set of
alternatives. However, for better supporting decision making, it may be more
informative to deliver a small menu of alternatives whose mean performances are
among the top $m$. Such problem, called optimal subset selection (OSS), is
generally more challenging to address than the conventional R&S. This challenge
becomes even more significant when the number of alternatives is considerably
large. Thus, the focus of this paper is on addressing the large-scale OSS
problem. To achieve this goal, we design a top-$m$ greedy selection mechanism
that keeps sampling the current top $m$ alternatives with top $m$ running
sample means and propose the explore-first top-$m$ greedy (EFG-$m$) procedure.
Through an extended boundary-crossing framework, we prove that the EFG-$m$
procedure is both sample optimal and consistent in terms of the probability of
good selection, confirming its effectiveness in solving large-scale OSS
problem. Surprisingly, we also demonstrate that the EFG-$m$ procedure enables
to achieve an indifference-based ranking within the selected subset of
alternatives at no extra cost. This is highly beneficial as it delivers deeper
insights to decision-makers, enabling more informed decision-makings. Lastly,
numerical experiments validate our results and demonstrate the efficiency of
our procedures.

### 摘要 (中文)

传统的排名和选择（R&R）通常旨在从有限的备选方案中选择具有最大平均性能的唯一最佳替代品。然而，为了更好地支持决策制定，可能更有效地提供一个包含少数顶m个平均性能在top m之上的备选方案菜单，这样的问题被称为最优子集选择（OSS），一般比常规的R&R更为挑战性。当备选方案数量相当大时，这种挑战变得尤其重要。因此，本论文关注解决大规模的OSS问题。为了实现这一目标，我们设计了一个基于top m的贪婪选择机制，该机制不断抽取当前top m个替代品，并使用top m运行样本均值进行采样，并提出探索-首先的top m贪婪（EFG-m）程序。通过扩展边界穿越框架，我们证明了EFG-m程序在概率上良好选择的概率方面是样本最优且一致的，这证实了它在解决大规模的OSS问题中的有效性。令人惊讶的是，我们还演示了EFG-m程序能够以不额外成本在一个备选方案的选定子集中实现无偏排名。这对决策者来说是一种巨大的益处，因为它提供了对决策者的深入洞察，使他们能做出更加知情的决策。最后，数值实验验证了我们的结果并展示了我们的方法的有效性。

---

## MoDeGPT_ Modular Decomposition for Large Language Model Compression

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09632v1)

### Abstract (English)

Large Language Models (LLMs) have reshaped the landscape of artificial
intelligence by demonstrating exceptional performance across various tasks.
However, substantial computational requirements make their deployment
challenging on devices with limited resources. Recently, compression methods
using low-rank matrix techniques have shown promise, yet these often lead to
degraded accuracy or introduce significant overhead in parameters and inference
latency. This paper introduces \textbf{Mo}dular \textbf{De}composition
(MoDeGPT), a novel structured compression framework that does not need recovery
fine-tuning while resolving the above drawbacks. MoDeGPT partitions the
Transformer block into modules comprised of matrix pairs and reduces the hidden
dimensions via reconstructing the module-level outputs. MoDeGPT is developed
based on a theoretical framework that utilizes three well-established matrix
decomposition algorithms -- Nystr\"om approximation, CR decomposition, and SVD
-- and applies them to our redefined transformer modules. Our comprehensive
experiments show MoDeGPT, without backward propagation, matches or surpasses
previous structured compression methods that rely on gradient information, and
saves 98% of compute costs on compressing a 13B model. On \textsc{Llama}-2/3
and OPT models, MoDeGPT maintains 90-95% zero-shot performance with 25-30%
compression rates. Moreover, the compression can be done on a single GPU within
a few hours and increases the inference throughput by up to 46%.

### 摘要 (中文)

大型语言模型（LLM）通过在各种任务上表现出色，彻底改变了人工智能的面貌。然而，其部署所需的大量计算要求使得有限资源设备上的部署变得具有挑战性。最近，使用低秩矩阵技术进行压缩的方法显示出潜力，但这些方法往往会导致准确性下降或参数和推理延迟引入重大开销。本文提出了一种名为MoDEGPT的新结构压缩框架，该框架不需要恢复微调，同时解决了上述缺点。MoDEGPT基于一种利用三个已确立的矩阵分解算法——Nyström近似、CR分解和SVD——并将其应用到我们重新定义的变换模块中。我们的全面实验表明，在不进行反向传播的情况下，MoDEGPT与依赖梯度信息的传统结构压缩方法相比，可以匹配甚至超过它们，并且在压缩13B模型时节省了高达98%的计算成本。对于Llama-2/3和OPT模型，MoDEGPT保持了零-shot性能，压缩率达到了25-30%，并且可以在几小时内完成单GPU的压缩工作，并提高了推理吞吐量高达46%。

---

## Regularization for Adversarial Robust Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09672v1)

### Abstract (English)

Despite the growing prevalence of artificial neural networks in real-world
applications, their vulnerability to adversarial attacks remains to be a
significant concern, which motivates us to investigate the robustness of
machine learning models. While various heuristics aim to optimize the
distributionally robust risk using the $\infty$-Wasserstein metric, such a
notion of robustness frequently encounters computation intractability. To
tackle the computational challenge, we develop a novel approach to adversarial
training that integrates $\phi$-divergence regularization into the
distributionally robust risk function. This regularization brings a notable
improvement in computation compared with the original formulation. We develop
stochastic gradient methods with biased oracles to solve this problem
efficiently, achieving the near-optimal sample complexity. Moreover, we
establish its regularization effects and demonstrate it is asymptotic
equivalence to a regularized empirical risk minimization (ERM) framework, by
considering various scaling regimes of the regularization parameter $\eta$ and
robustness level $\rho$. These regimes yield gradient norm regularization,
variance regularization, or a smoothed gradient norm regularization that
interpolates between these extremes. We numerically validate our proposed
method in supervised learning, reinforcement learning, and contextual learning
and showcase its state-of-the-art performance against various adversarial
attacks.

### 摘要 (中文)

尽管在实际世界应用中人工神经网络的出现越来越普遍，但其对抗性攻击的脆弱性仍然是一个重要的关注点，这促使我们研究机器学习模型的鲁棒性。虽然各种启发式方法试图通过$\infty$-Wasserstein距离使用分布稳健风险优化优化分布稳健风险，但这种度量通常遇到计算不完整性的问题。为了应对这一计算挑战，我们开发了一种新的对抗训练策略，该策略将$\phi$差异规范化引入到分布稳健风险函数中。这种规范带来了比原始表述明显的改进。我们发展了基于有偏或acles的随机梯度方法来高效地解决这个问题，实现了近最优样本复杂度。此外，我们证明了它的正则化效果，并展示了它与正规化的经验风险最小化（ERM）框架的渐近等价性，考虑了不同缩放规则下的正则参数$\eta$和鲁棒水平$\rho$。这些规制范围产生梯度范数正则化、方差正则化，或者是一个光滑的梯度范数正则化，后者介于这两者之间。我们在监督学习、强化学习和语境学习中进行了数值验证，并展示了其在对抗性攻击方面的先进性能。

---

## Parseval Convolution Operators and Neural Networks

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09981v1)

### Abstract (English)

We first establish a kernel theorem that characterizes all linear
shift-invariant (LSI) operators acting on discrete multicomponent signals. This
result naturally leads to the identification of the Parseval convolution
operators as the class of energy-preserving filterbanks. We then present a
constructive approach for the design/specification of such filterbanks via the
chaining of elementary Parseval modules, each of which being parameterized by
an orthogonal matrix or a 1-tight frame. Our analysis is complemented with
explicit formulas for the Lipschitz constant of all the components of a
convolutional neural network (CNN), which gives us a handle on their stability.
Finally, we demonstrate the usage of those tools with the design of a CNN-based
algorithm for the iterative reconstruction of biomedical images. Our algorithm
falls within the plug-and-play framework for the resolution of inverse
problems. It yields better-quality results than the sparsity-based methods used
in compressed sensing, while offering essentially the same convergence and
robustness guarantees.

### 摘要 (中文)

我们首先建立了定理，它描述了所有线性移不变（LTI）操作作用于离散多成分信号的全部。这一结果自然地导致了对能量保持滤波器进行识别。然后，我们通过链式方法设计和指定这样的滤波器。我们的分析被附加了一个卷积神经网络（CNN）中所有组件的Lipschitz常数的明确公式，这给了我们一个稳定性的把握。最后，我们将这些工具用于设计基于CNN的图像逆重构算法。我们的算法属于解决反问题的插件和通用框架的一部分。它比压缩感知中的稠密编码方法产生更好的质量的结果，同时提供几乎相同的安全性和收敛性保证。

---

## Deformation-aware GAN for Medical Image Synthesis with Substantially Misaligned Pairs

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09432v1)

### Abstract (English)

Medical image synthesis generates additional imaging modalities that are
costly, invasive or harmful to acquire, which helps to facilitate the clinical
workflow. When training pairs are substantially misaligned (e.g., lung MRI-CT
pairs with respiratory motion), accurate image synthesis remains a critical
challenge. Recent works explored the directional registration module to adjust
misalignment in generative adversarial networks (GANs); however, substantial
misalignment will lead to 1) suboptimal data mapping caused by correspondence
ambiguity, and 2) degraded image fidelity caused by morphology influence on
discriminators. To address the challenges, we propose a novel Deformation-aware
GAN (DA-GAN) to dynamically correct the misalignment during the image synthesis
based on multi-objective inverse consistency. Specifically, in the generative
process, three levels of inverse consistency cohesively optimise symmetric
registration and image generation for improved correspondence. In the
adversarial process, to further improve image fidelity under misalignment, we
design deformation-aware discriminators to disentangle the mismatched spatial
morphology from the judgement of image fidelity. Experimental results show that
DA-GAN achieved superior performance on a public dataset with simulated
misalignments and a real-world lung MRI-CT dataset with respiratory motion
misalignment. The results indicate the potential for a wide range of medical
image synthesis tasks such as radiotherapy planning.

### 摘要 (中文)

医学图像合成可以生成额外的影像模态，这些模态昂贵、侵入性或有害于获取，这有助于促进临床流程。当训练对齐严重偏离（例如，肺MRI-CT对齐中呼吸运动）时，准确的图像合成仍然是一个关键挑战。最近的工作探索了方向注册模块来调整生成对抗网络（GAN）中的不一致性；然而，严重的偏差会导致1）由于对应模糊导致的数据映射不佳和2）由于形态影响导致的图像质量下降。为了应对这一挑战，我们提出了一种新型的变形感知GAN（DA-GAN），在基于多目标逆一致性的情况下动态修正合成过程中的偏差。具体来说，在生成过程中，通过三水平的逆一致性的优化实现对称注册和图像生成以改善匹配性。在对抗性过程中，为了解决因偏差而产生的图像质量下降，我们设计了变形感知判别器，以从图像质量的判断中解离出错配的空间形态。实验结果表明，DA-GAN在模拟不对齐和实际世界中的肺MRI-CT对齐数据上都取得了优于公共数据集的优势性能。结果显示，这种方法具有广泛医疗图像合成任务的可能性，如放疗计划。

---

## ExpoMamba_ Exploiting Frequency SSM Blocks for Efficient and Effective Image Enhancement

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09650v1)

### Abstract (English)

Low-light image enhancement remains a challenging task in computer vision,
with existing state-of-the-art models often limited by hardware constraints and
computational inefficiencies, particularly in handling high-resolution images.
Recent foundation models, such as transformers and diffusion models, despite
their efficacy in various domains, are limited in use on edge devices due to
their computational complexity and slow inference times. We introduce
ExpoMamba, a novel architecture that integrates components of the frequency
state space within a modified U-Net, offering a blend of efficiency and
effectiveness. This model is specifically optimized to address mixed exposure
challenges, a common issue in low-light image enhancement, while ensuring
computational efficiency. Our experiments demonstrate that ExpoMamba enhances
low-light images up to 2-3x faster than traditional models with an inference
time of 36.6 ms and achieves a PSNR improvement of approximately 15-20% over
competing models, making it highly suitable for real-time image processing
applications.

### 摘要 (中文)

低光图像增强在计算机视觉中是一个具有挑战性的任务，现有的最先进的模型往往受限于硬件约束和计算效率低下，尤其是在处理高分辨率图像方面。最近的基础模型，如Transformer和扩散模型，在各种领域表现出色，但由于其计算复杂性和慢的推断时间，它们在边缘设备上无法使用。我们引入了ExpoMamba，一种新型架构，它集成到修改后的U-Net中的频率状态空间组件，提供了一种效率和效果的结合体。这个模型特别优化以解决混合曝光问题，这是低光图像增强中常见的问题之一，同时确保计算效率。我们的实验表明，ExpoMamba可以比传统模型快2-3倍地增强低光图像，推理时间为36.6毫秒，并且与竞争模型相比实现了约15-20%的PSNR改进，使其非常适合实时图像处理应用。

---

## Photorealistic Object Insertion with Diffusion-Guided Inverse Rendering

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09702v1)

### Abstract (English)

The correct insertion of virtual objects in images of real-world scenes
requires a deep understanding of the scene's lighting, geometry and materials,
as well as the image formation process. While recent large-scale diffusion
models have shown strong generative and inpainting capabilities, we find that
current models do not sufficiently "understand" the scene shown in a single
picture to generate consistent lighting effects (shadows, bright reflections,
etc.) while preserving the identity and details of the composited object. We
propose using a personalized large diffusion model as guidance to a physically
based inverse rendering process. Our method recovers scene lighting and
tone-mapping parameters, allowing the photorealistic composition of arbitrary
virtual objects in single frames or videos of indoor or outdoor scenes. Our
physically based pipeline further enables automatic materials and tone-mapping
refinement.

### 摘要 (中文)

虚拟对象在真实世界场景中的正确插入，需要对场景的照明、几何形状和材料有深入的理解，以及图像形成过程。虽然最近大规模扩散模型显示出强大的生成能力和修复能力，但我们发现当前模型不足以从单幅图像中理解场景，从而在保持复合物体身份和细节的同时产生一致的光照效果（阴影、明亮反射等）。我们提出使用个性化的大规模扩散模型作为逆渲染流程的指导。我们的方法恢复了场景的照明和色调映射参数，允许在室内或室外单帧或视频中任意虚拟对象的光效自动生成，进一步提高了自动材料和色调映射的可变性。

---

## Pedestrian Attribute Recognition_ A New Benchmark Dataset and A Large Language Model Augmented Framework

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09720v1)

### Abstract (English)

Pedestrian Attribute Recognition (PAR) is one of the indispensable tasks in
human-centered research. However, existing datasets neglect different domains
(e.g., environments, times, populations, and data sources), only conducting
simple random splits, and the performance of these datasets has already
approached saturation. In the past five years, no large-scale dataset has been
opened to the public. To address this issue, this paper proposes a new
large-scale, cross-domain pedestrian attribute recognition dataset to fill the
data gap, termed MSP60K. It consists of 60,122 images and 57 attribute
annotations across eight scenarios. Synthetic degradation is also conducted to
further narrow the gap between the dataset and real-world challenging
scenarios. To establish a more rigorous benchmark, we evaluate 17
representative PAR models under both random and cross-domain split protocols on
our dataset. Additionally, we propose an innovative Large Language Model (LLM)
augmented PAR framework, named LLM-PAR. This framework processes pedestrian
images through a Vision Transformer (ViT) backbone to extract features and
introduces a multi-embedding query Transformer to learn partial-aware features
for attribute classification. Significantly, we enhance this framework with LLM
for ensemble learning and visual feature augmentation. Comprehensive
experiments across multiple PAR benchmark datasets have thoroughly validated
the efficacy of our proposed framework. The dataset and source code
accompanying this paper will be made publicly available at
\url{https://github.com/Event-AHU/OpenPAR}.

### 摘要 (中文)

行人特征识别（Par）是人类中心研究不可或缺的任务之一。然而，现有的数据集忽略了不同的领域（例如环境、时间、人口和数据源），仅进行简单的随机分割，并且这些数据集的表现已经接近饱和状态。在过去五年中，没有大型公开数据集。为了应对这一问题，本论文提出了一种新的大规模跨域行人特征识别数据集，名为MSP60K。它包括八个场景的60,122张图像和57个属性标注。此外，我们还提出了一个创新的大型语言模型（Lam）增强的行人特征识别框架，称为Lam-par。这个框架通过视觉Transformer（Vit）骨架提取特征，并引入一个多嵌入查询Transformer来学习部分感知特征用于属性分类。显著的是，我们在此框架上增强了Lam以集成学习和视觉特征增强。通过在多个行人特征基准数据集上的全面验证，我们的提议框架的有效性得到了证明。与本文相关的数据集及其源代码将由以下网址https://github.com/Event-AHU/OpenPAR公共可用：

---

## R2GenCSR_ Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09743v1)

### Abstract (English)

Inspired by the tremendous success of Large Language Models (LLMs), existing
X-ray medical report generation methods attempt to leverage large models to
achieve better performance. They usually adopt a Transformer to extract the
visual features of a given X-ray image, and then, feed them into the LLM for
text generation. How to extract more effective information for the LLMs to help
them improve final results is an urgent problem that needs to be solved.
Additionally, the use of visual Transformer models also brings high
computational complexity. To address these issues, this paper proposes a novel
context-guided efficient X-ray medical report generation framework.
Specifically, we introduce the Mamba as the vision backbone with linear
complexity, and the performance obtained is comparable to that of the strong
Transformer model. More importantly, we perform context retrieval from the
training set for samples within each mini-batch during the training phase,
utilizing both positively and negatively related samples to enhance feature
representation and discriminative learning. Subsequently, we feed the vision
tokens, context information, and prompt statements to invoke the LLM for
generating high-quality medical reports. Extensive experiments on three X-ray
report generation datasets (i.e., IU-Xray, MIMIC-CXR, CheXpert Plus) fully
validated the effectiveness of our proposed model. The source code of this work
will be released on \url{https://github.com/Event-AHU/Medical_Image_Analysis}.

### 摘要 (中文)

由大型语言模型（LLM）的巨大成功启发，现有的医疗报告生成方法试图利用大模型来实现更好的性能。它们通常采用Transformer提取给定X光图像的视觉特征，并将其输入到LLM中进行文本生成。如何从LLMs那里获取更有效的信息以帮助他们改善最终结果是一个迫切需要解决的问题。

此外，使用视觉Transformer模型也带来了很高的计算复杂性。为了解决这些问题，本文提出了一种新颖的上下文引导高效X光医学报告生成框架。
具体而言，我们引入了线性复杂度的Mamba作为视图骨架，训练结果与强Transformer模型相当。更重要的是，在训练阶段，我们在每个mini-batch内的样本之间执行上下文检索，同时利用正负相关样本来增强特征表示和区分学习。随后，我们将视觉令牌、上下文信息以及提示语句输入到LLM中，以调用其生成高质量的医学报告。在三个X光报告生成数据集（即IU-Xray、MIMIC-CXR、CheXpert Plus）上进行了广泛的实验，验证了我们提出的模型的有效性。该工作的源代码将在https://github.com/Event-AHU/Medical_Image_Analysis上发布。

---

## Event Stream based Human Action Recognition_ A High-Definition Benchmark Dataset and Algorithms

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09764v1)

### Abstract (English)

Human Action Recognition (HAR) stands as a pivotal research domain in both
computer vision and artificial intelligence, with RGB cameras dominating as the
preferred tool for investigation and innovation in this field. However, in
real-world applications, RGB cameras encounter numerous challenges, including
light conditions, fast motion, and privacy concerns. Consequently, bio-inspired
event cameras have garnered increasing attention due to their advantages of low
energy consumption, high dynamic range, etc. Nevertheless, most existing
event-based HAR datasets are low resolution ($346 \times 260$). In this paper,
we propose a large-scale, high-definition ($1280 \times 800$) human action
recognition dataset based on the CeleX-V event camera, termed CeleX-HAR. It
encompasses 150 commonly occurring action categories, comprising a total of
124,625 video sequences. Various factors such as multi-view, illumination,
action speed, and occlusion are considered when recording these data. To build
a more comprehensive benchmark dataset, we report over 20 mainstream HAR models
for future works to compare. In addition, we also propose a novel Mamba vision
backbone network for event stream based HAR, termed EVMamba, which equips the
spatial plane multi-directional scanning and novel voxel temporal scanning
mechanism. By encoding and mining the spatio-temporal information of event
streams, our EVMamba has achieved favorable results across multiple datasets.
Both the dataset and source code will be released on
\url{https://github.com/Event-AHU/CeleX-HAR}

### 摘要 (中文)

人类动作识别（HAR）是计算机视觉和人工智能领域中至关重要的一项研究主题，RGB摄像头作为这一领域的首选工具，在现实世界的应用中遇到了诸多挑战，包括光照条件、快速运动以及隐私问题。因此，生物启发式事件相机因其低能耗、高动态范围等优势而受到了越来越多的关注。然而，现有的基于事件的HAR数据集分辨率较低（346 x 260），在本论文中，我们提出了一个基于CeleX-V事件相机的大规模高分辨率（1280 x 800）人类动作识别数据集，名为CeleX-HAR。它涵盖了150种常见的动作类别，总计有124,625个视频序列。我们在录制这些数据时考虑了诸如多视图、照明、动作速度以及遮挡等多种因素。为了构建更全面的基准数据集，我们报告了20种主流的人类动作识别模型供未来工作比较。此外，我们还提出了一种新的Mamba视角背骨网络用于基于事件流的人体动作识别，命名为EVMamba，该网络配备了空间方向的多维度扫描和新颖的时间步长扫描机制。通过编码并挖掘事件流的空间-时间信息，我们的EVMamba在多个数据集上取得了良好的结果。同时，该数据集及其源代码将在https://github.com/Event-AHU/CeleX-HAR上发布。

---

## Preoperative Rotator Cuff Tear Prediction from Shoulder Radiographs using a Convolutional Block Attention Module-Integrated Neural Network

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09894v1)

### Abstract (English)

Research question: We test whether a plane shoulder radiograph can be used
together with deep learning methods to identify patients with rotator cuff
tears as opposed to using an MRI in standard of care. Findings: By integrating
convolutional block attention modules into a deep neural network, our model
demonstrates high accuracy in detecting patients with rotator cuff tears,
achieving an average AUC of 0.889 and an accuracy of 0.831. Meaning: This study
validates the efficacy of our deep learning model to accurately detect rotation
cuff tears from radiographs, offering a viable pre-assessment or alternative to
more expensive imaging techniques such as MRI.

### 摘要 (中文)

研究问题：我们测试是否可以使用深学习方法与飞机肩部X光片一起识别患者是否有肩袖损伤，而不是在常规护理中使用MRI。发现：通过将卷积块注意力模块整合到深度神经网络中，我们的模型在检测肩袖损伤方面表现出极高的准确性，平均AUC为0.889，准确率为0.831。意义：这项研究表明，我们的深度学习模型的有效性已验证，在X光片上正确检测肩袖损伤的能力，并提供了可行的预先评估或替代更昂贵的成像技术（如MRI）的选择。

---

## LCE_ A Framework for Explainability of DNNs for Ultrasound Image Based on Concept Discovery

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09899v1)

### Abstract (English)

Explaining the decisions of Deep Neural Networks (DNNs) for medical images
has become increasingly important. Existing attribution methods have difficulty
explaining the meaning of pixels while existing concept-based methods are
limited by additional annotations or specific model structures that are
difficult to apply to ultrasound images. In this paper, we propose the Lesion
Concept Explainer (LCE) framework, which combines attribution methods with
concept-based methods. We introduce the Segment Anything Model (SAM),
fine-tuned on a large number of medical images, for concept discovery to enable
a meaningful explanation of ultrasound image DNNs. The proposed framework is
evaluated in terms of both faithfulness and understandability. We point out
deficiencies in the popular faithfulness evaluation metrics and propose a new
evaluation metric. Our evaluation of public and private breast ultrasound
datasets (BUSI and FG-US-B) shows that LCE performs well compared to
commonly-used explainability methods. Finally, we also validate that LCE can
consistently provide reliable explanations for more meaningful fine-grained
diagnostic tasks in breast ultrasound.

### 摘要 (中文)

解释深神经网络（DNN）对医学图像的决策变得越来越重要。现有的归因方法难以解释像素的意义，而现有的基于概念的方法则受到额外标注或特定模型结构的限制，这些结构难以应用于超声图像。本论文提出了一种融合归因方法和基于概念的方法的Lesion Concept Explainer（LCE）框架。我们引入了Segment Anything Model（SAM），它是在大量医疗图像上微调的，用于发现概念以提供有意义的超声图像DNN解释。提出的框架在信度和可理解性方面进行了评估。我们在公共私有乳腺超声数据集（BUSI和FG-US-B）中评估了LCE，并显示其性能优于常用解释性方法。最后，我们也验证了LCE可以为更具有意义的细粒度诊断任务提供可靠的解释。

---

## SpaRP_ Fast 3D Object Reconstruction and Pose Estimation from Sparse Views

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10195v1)

### Abstract (English)

Open-world 3D generation has recently attracted considerable attention. While
many single-image-to-3D methods have yielded visually appealing outcomes, they
often lack sufficient controllability and tend to produce hallucinated regions
that may not align with users' expectations. In this paper, we explore an
important scenario in which the input consists of one or a few unposed 2D
images of a single object, with little or no overlap. We propose a novel
method, SpaRP, to reconstruct a 3D textured mesh and estimate the relative
camera poses for these sparse-view images. SpaRP distills knowledge from 2D
diffusion models and finetunes them to implicitly deduce the 3D spatial
relationships between the sparse views. The diffusion model is trained to
jointly predict surrogate representations for camera poses and multi-view
images of the object under known poses, integrating all information from the
input sparse views. These predictions are then leveraged to accomplish 3D
reconstruction and pose estimation, and the reconstructed 3D model can be used
to further refine the camera poses of input views. Through extensive
experiments on three datasets, we demonstrate that our method not only
significantly outperforms baseline methods in terms of 3D reconstruction
quality and pose prediction accuracy but also exhibits strong efficiency. It
requires only about 20 seconds to produce a textured mesh and camera poses for
the input views. Project page: https://chaoxu.xyz/sparp.

### 摘要 (中文)

开放世界三维生成技术最近引起了广泛的关注。虽然许多单图像到三维方法产生了视觉上吸引人的结果，但它们往往缺乏足够的可控性和容易产生用户期望不一致的幻觉区域。在这篇论文中，我们探索了一个重要的场景，在这个场景中输入是一些未对齐的单一对象的二维图像中的一个或几个图像，几乎没有重叠。我们提出了一种新的方法，SpaRP，用于重建三维纹理网格和估计这些稀疏视角图像的相对相机位置。SpaRP从二维扩散模型中汲取知识，并对其进行微调以隐式推断出稀疏视图之间三维空间关系。扩散模型被训练来同时预测相机位置的近似表示和已知姿势下的多视角物体图像的联合信息，整合了输入稀疏视图的所有信息。这些预测随后被用来完成三维重建和相机位置估计，并且重构后的三维模型可以用于进一步调整输入视图的相机位置。通过在三个数据集上的广泛实验，我们展示了我们的方法不仅在三维重建质量和相机位置预测精度方面显著超过了基准方法，而且表现出强大的效率。只需大约20秒就可以生成三维模型和输入视图的相机位置。项目页面：https://chaoxu.xyz/sparp。

---

## Federated Graph Learning with Structure Proxy Alignment

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09393v1)

### Abstract (English)

Federated Graph Learning (FGL) aims to learn graph learning models over graph
data distributed in multiple data owners, which has been applied in various
applications such as social recommendation and financial fraud detection.
Inherited from generic Federated Learning (FL), FGL similarly has the data
heterogeneity issue where the label distribution may vary significantly for
distributed graph data across clients. For instance, a client can have the
majority of nodes from a class, while another client may have only a few nodes
from the same class. This issue results in divergent local objectives and
impairs FGL convergence for node-level tasks, especially for node
classification. Moreover, FGL also encounters a unique challenge for the node
classification task: the nodes from a minority class in a client are more
likely to have biased neighboring information, which prevents FGL from learning
expressive node embeddings with Graph Neural Networks (GNNs). To grapple with
the challenge, we propose FedSpray, a novel FGL framework that learns local
class-wise structure proxies in the latent space and aligns them to obtain
global structure proxies in the server. Our goal is to obtain the aligned
structure proxies that can serve as reliable, unbiased neighboring information
for node classification. To achieve this, FedSpray trains a global
feature-structure encoder and generates unbiased soft targets with structure
proxies to regularize local training of GNN models in a personalized way. We
conduct extensive experiments over four datasets, and experiment results
validate the superiority of FedSpray compared with other baselines. Our code is
available at https://github.com/xbfu/FedSpray.

### 摘要 (中文)

联邦图学习（Federated Graph Learning，简称FGL）旨在学习分布在网络多个数据所有者处的分布式图数据上的图学习模型。它与通用联邦学习（FedLearn）有类似之处，同样面临数据异质性问题，即对于分布在客户端的分布式图数据而言，标签分布可能有很大差异。例如，一个客户可能会拥有该类中的大多数节点，而另一个客户则只有少数相同类别的节点。这种问题导致了局部目标的分歧，并影响了基于节点的任务的学习收敛，特别是在节点分类任务中尤其如此。此外，FGL还面临着针对节点分类任务的独特挑战：在某个客户的少数类别中，节点更有可能具有偏向性的邻域信息，这使得FGL无法通过图神经网络（GNN）学习表达丰富的节点嵌入。为解决这一挑战，我们提出了一种新的联邦学习框架FedSpray，该框架在隐空间中学习本地类别的结构代理，并使它们相互关联以获得服务器端全局结构代理。我们的目标是获得可以作为可靠、无偏倚邻域信息用于节点分类的关联结构代理。为了实现这一目标，FedSpray训练了一个全局特征结构编码器，并生成了不受结构代理影响的软目标来对GNN模型进行个性化地正则化。我们在四个数据集上进行了广泛的研究，实验结果验证了FedSpray相对于其他基准的优越性。我们的代码可以在https://github.com/xbfu/FedSpray中找到。

---

## Parallel Sampling via Counting

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09442v1)

### Abstract (English)

We show how to use parallelization to speed up sampling from an arbitrary
distribution $\mu$ on a product space $[q]^n$, given oracle access to counting
queries: $\mathbb{P}_{X\sim \mu}[X_S=\sigma_S]$ for any $S\subseteq [n]$ and
$\sigma_S \in [q]^S$. Our algorithm takes $O({n^{2/3}\cdot
\operatorname{polylog}(n,q)})$ parallel time, to the best of our knowledge, the
first sublinear in $n$ runtime for arbitrary distributions. Our results have
implications for sampling in autoregressive models. Our algorithm directly
works with an equivalent oracle that answers conditional marginal queries
$\mathbb{P}_{X\sim \mu}[X_i=\sigma_i\;\vert\; X_S=\sigma_S]$, whose role is
played by a trained neural network in autoregressive models. This suggests a
roughly $n^{1/3}$-factor speedup is possible for sampling in any-order
autoregressive models. We complement our positive result by showing a lower
bound of $\widetilde{\Omega}(n^{1/3})$ for the runtime of any parallel sampling
algorithm making at most $\operatorname{poly}(n)$ queries to the counting
oracle, even for $q=2$.

### 摘要 (中文)

我们展示了如何利用并行化来加快对任意分布 $\mu$ 在产品空间 $[q]^n$ 上的采样，给出计数查询的Oracle访问。我们的算法以 $O(n^{2/3} \cdot \operatorname{polylog}(n,q))$ 并发时间，这是目前为止已知的最小的时间复杂度，对于任意分布。我们的结果有重要的意义，它在自回归模型中影响了采样的速度。我们的算法直接与等价的Oracle工作，该Oracle回答条件边缘查询 $\mathbb{P}_{X\sim \mu}[X_i=\sigma_i\mid X_S=\sigma_S]$，其角色由自动回复的神经网络扮演。这表明任何阶自回归模型中的大约 $n^{1/3}$ 因子加速是可能的。我们通过展示一个下界 $\widetilde{\Omega}(n^{1/3})$ 来补充正向结果，即使对于 $q = 2$，对于最多使用 $\operatorname{poly}(n)$ 计数查询的任何并行采样算法也给出了下界。

---

## In-Memory Learning Automata Architecture using Y-Flash Cell

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09456v1)

### Abstract (English)

The modern implementation of machine learning architectures faces significant
challenges due to frequent data transfer between memory and processing units.
In-memory computing, primarily through memristor-based analog computing, offers
a promising solution to overcome this von Neumann bottleneck. In this
technology, data processing and storage are located inside the memory. Here, we
introduce a novel approach that utilizes floating-gate Y-Flash memristive
devices manufactured with a standard 180 nm CMOS process. These devices offer
attractive features, including analog tunability and moderate device-to-device
variation; such characteristics are essential for reliable decision-making in
ML applications. This paper uses a new machine learning algorithm, the Tsetlin
Machine (TM), for in-memory processing architecture. The TM's learning element,
Automaton, is mapped into a single Y-Flash cell, where the Automaton's range is
transferred into the Y-Flash's conductance scope. Through comprehensive
simulations, the proposed hardware implementation of the learning automata,
particularly for Tsetlin machines, has demonstrated enhanced scalability and
on-edge learning capabilities.

### 摘要 (中文)

现代机器学习架构的实现面临着数据传输频繁导致的内存与处理器之间的瓶颈。基于可编程电阻的模拟计算技术在其中提供了一种具有挑战性的解决方案，通过这种技术，数据处理和存储位于内存内部。在这里，我们介绍了使用标准180纳米CMOS工艺制造的浮点门Y-Flash毫伏记数器设备的一种新型方法。这些设备提供了吸引人的特性，包括模拟可调性和较小的设备到设备变化；这样的特征对于可靠地进行机器学习应用中的决策至关重要。本文使用新的机器学习算法Tsetlin Machine（TM）对内存处理架构进行介绍。TM的学习元素自动机映射到单个Y-Flash单元中，其中自动机的范围转移到了Y-Flash的导电性范围。通过全面的仿真，提出的硬件实现学习元件，特别是用于Tsetlin机器的自动机，已经展示了增强的可扩展性和边缘学习能力。

---

## A Unified Framework for Interpretable Transformers Using PDEs and Information Theory

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09523v1)

### Abstract (English)

This paper presents a novel unified theoretical framework for understanding
Transformer architectures by integrating Partial Differential Equations (PDEs),
Neural Information Flow Theory, and Information Bottleneck Theory. We model
Transformer information dynamics as a continuous PDE process, encompassing
diffusion, self-attention, and nonlinear residual components. Our comprehensive
experiments across image and text modalities demonstrate that the PDE model
effectively captures key aspects of Transformer behavior, achieving high
similarity (cosine similarity > 0.98) with Transformer attention distributions
across all layers. While the model excels in replicating general information
flow patterns, it shows limitations in fully capturing complex, non-linear
transformations. This work provides crucial theoretical insights into
Transformer mechanisms, offering a foundation for future optimizations in deep
learning architectural design. We discuss the implications of our findings,
potential applications in model interpretability and efficiency, and outline
directions for enhancing PDE models to better mimic the intricate behaviors
observed in Transformers, paving the way for more transparent and optimized AI
systems.

### 摘要 (中文)

这篇论文提出了一种新的统一理论框架来理解和分析Transformer架构，该框架综合了偏微分方程（PDE）、神经信息流动理论和信息阻塞理论。我们以连续的PDE过程模型化Transformer信息动力学，包括扩散、注意力和非线性残差组件。我们的跨模态全面实验展示了PDE模型有效地捕捉到了Transformer行为的关键方面，其在所有层之间Transformer注意力分布的相似度（余弦相似度>0.98）高达98%以上。虽然模型在复制一般信息流动模式方面表现出色，但在完全捕获复杂、非线性变换方面表现有限。本工作提供了对Transformer机制的重要理论洞察，为未来在深度学习架构设计方面的优化奠定了基础。我们讨论了我们发现的含义，潜在的应用于模型解释性和效率，并概述了改进PDE模型以更好地模仿观察到的Transformer复杂行为的方向，从而为更透明和优化的人工智能系统铺平道路。

---

## Say My Name_ a Model_s Bias Discovery Framework

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09570v1)

### Abstract (English)

In the last few years, due to the broad applicability of deep learning to
downstream tasks and end-to-end training capabilities, increasingly more
concerns about potential biases to specific, non-representative patterns have
been raised. Many works focusing on unsupervised debiasing usually leverage the
tendency of deep models to learn ``easier'' samples, for example by clustering
the latent space to obtain bias pseudo-labels. However, the interpretation of
such pseudo-labels is not trivial, especially for a non-expert end user, as it
does not provide semantic information about the bias features. To address this
issue, we introduce ``Say My Name'' (SaMyNa), the first tool to identify biases
within deep models semantically. Unlike existing methods, our approach focuses
on biases learned by the model. Our text-based pipeline enhances explainability
and supports debiasing efforts: applicable during either training or post-hoc
validation, our method can disentangle task-related information and proposes
itself as a tool to analyze biases. Evaluation on traditional benchmarks
demonstrates its effectiveness in detecting biases and even disclaiming them,
showcasing its broad applicability for model diagnosis.

### 摘要 (中文)

在过去的几年里，由于深度学习对下游任务的广泛适用性和端到端训练能力，越来越多的人开始关注特定非代表性模式潜在偏见的问题。许多专注于无监督去偏差化的工作通常利用深层模型倾向于“更容易”样本的学习趋势，例如通过聚类隐空间来获得偏伪标签。然而，解释这些偏伪标签并不容易，尤其是对于非专家用户来说，它们不能提供关于偏移特征的语义信息。为了应对这一问题，我们引入了“说我的名字”（SaMyNa），这是第一个识别深度模型内部潜伏偏见的工具。与现有方法不同，我们的方法侧重于模型学习的偏见。基于文本的管道增强了可解释性，并支持去偏差化努力：在训练期间或事后验证时均可应用，我们的方法可以解离任务相关的信息并将其作为分析偏见的工具。在传统基准上的评估展示了它在检测偏见和甚至否认它们方面的作用，显示了其适用于模型诊断的广泛适用性。

---

## Attention is a smoothed cubic spline

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09624v1)

### Abstract (English)

We highlight a perhaps important but hitherto unobserved insight: The
attention module in a transformer is a smoothed cubic spline. Viewed in this
manner, this mysterious but critical component of a transformer becomes a
natural development of an old notion deeply entrenched in classical
approximation theory. More precisely, we show that with ReLU-activation,
attention, masked attention, encoder-decoder attention are all cubic splines.
As every component in a transformer is constructed out of compositions of
various attention modules (= cubic splines) and feed forward neural networks (=
linear splines), all its components -- encoder, decoder, and encoder-decoder
blocks; multilayered encoders and decoders; the transformer itself -- are cubic
or higher-order splines. If we assume the Pierce-Birkhoff conjecture, then the
converse also holds, i.e., every spline is a ReLU-activated encoder. Since a
spline is generally just $C^2$, one way to obtain a smoothed $C^\infty$-version
is by replacing ReLU with a smooth activation; and if this activation is chosen
to be SoftMax, we recover the original transformer as proposed by Vaswani et
al. This insight sheds light on the nature of the transformer by casting it
entirely in terms of splines, one of the best known and thoroughly understood
objects in applied mathematics.

### 摘要 (中文)

我们强调了一个可能的重要但尚未被注意到的见解：Transformer中的注意力模块是一个光滑的三次样条。以这种方式看待，作为Transformer中一个神秘而至关重要的组件，它变成了经典近似理论深深根植的一个自然发展。更准确地说，我们证明了ReLU激活、注意力、掩码注意力和编码器-解码器注意力都是三次样条。在Transformer的所有组成部分都由各种注意力模块（= 三次样条）和前馈神经网络（= 线性样条）的组合构成的情况下，所有这些部件——编码器、解码器、编码器-解码块；多层编码器和解码器；整个变换器本身——都是三次或更高阶的样条。如果我们假设Pierce-Birkhoff猜想成立，则逆命题也成立，即每个样条都是ReLU激活的编码器。由于一般情况下只是$C^2$，为了获得光滑的$C^\infty$版本的一种方式是用光滑激活替换ReLU；如果选择这个激活函数为SoftMax，则我们可以恢复Vaswani等人提出的原始变体。这一洞察力揭示了变换器的本质，将其完全转化为样条，这是应用数学中最著名的对象之一。

---

## Meta-Learning on Augmented Gene Expression Profiles for Enhanced Lung Cancer Detection

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09635v1)

### Abstract (English)

Gene expression profiles obtained through DNA microarray have proven
successful in providing critical information for cancer detection classifiers.
However, the limited number of samples in these datasets poses a challenge to
employ complex methodologies such as deep neural networks for sophisticated
analysis. To address this "small data" dilemma, Meta-Learning has been
introduced as a solution to enhance the optimization of machine learning models
by utilizing similar datasets, thereby facilitating a quicker adaptation to
target datasets without the requirement of sufficient samples. In this study,
we present a meta-learning-based approach for predicting lung cancer from gene
expression profiles. We apply this framework to well-established deep learning
methodologies and employ four distinct datasets for the meta-learning tasks,
where one as the target dataset and the rest as source datasets. Our approach
is evaluated against both traditional and deep learning methodologies, and the
results show the superior performance of meta-learning on augmented source data
compared to the baselines trained on single datasets. Moreover, we conduct the
comparative analysis between meta-learning and transfer learning methodologies
to highlight the efficiency of the proposed approach in addressing the
challenges associated with limited sample sizes. Finally, we incorporate the
explainability study to illustrate the distinctiveness of decisions made by
meta-learning.

### 摘要 (中文)

通过DNA微阵列获得的基因表达模式已经证明，它们成功地提供了癌症检测分类器的关键信息。然而，在这些数据集中的样本数量有限，这使得使用深度神经网络等复杂方法进行深入分析变得具有挑战性。为了解决“小数据”困境，Meta-Learning作为一种解决方案，旨在利用相似的数据集来优化机器学习模型的优化过程，从而加快适应目标数据的速度，而无需足够的样本量。在本研究中，我们提出了一种基于Meta-Learning的方法来预测肺部癌症从基因表达模式。我们将这种方法应用于已建立的深度学习方法，并根据Meta-Learning任务应用四个不同的数据集。我们的方法被用于评估传统的和深度学习的方法，并显示了Meta-Learning对增强源数据比仅使用单个数据集训练的基准线有更好的性能。此外，我们进行了Meta-Learning与迁移学习方法之间的比较分析，以突出所提出的框架在解决有限样本大小带来的挑战时效率。最后，我们结合解释性研究来说明Meta-Learning做出的不同决策的独特性。

---

## LightWeather_ Harnessing Absolute Positional Encoding to Efficient and Scalable Global Weather Forecasting

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09695v1)

### Abstract (English)

Recently, Transformers have gained traction in weather forecasting for their
capability to capture long-term spatial-temporal correlations. However, their
complex architectures result in large parameter counts and extended training
times, limiting their practical application and scalability to global-scale
forecasting. This paper aims to explore the key factor for accurate weather
forecasting and design more efficient solutions. Interestingly, our empirical
findings reveal that absolute positional encoding is what really works in
Transformer-based weather forecasting models, which can explicitly model the
spatial-temporal correlations even without attention mechanisms. We
theoretically prove that its effectiveness stems from the integration of
geographical coordinates and real-world time features, which are intrinsically
related to the dynamics of weather. Based on this, we propose LightWeather, a
lightweight and effective model for station-based global weather forecasting.
We employ absolute positional encoding and a simple MLP in place of other
components of Transformer. With under 30k parameters and less than one hour of
training time, LightWeather achieves state-of-the-art performance on global
weather datasets compared to other advanced DL methods. The results underscore
the superiority of integrating spatial-temporal knowledge over complex
architectures, providing novel insights for DL in weather forecasting.

### 摘要 (中文)

最近，变体模型在天气预报中获得了关注，因为它们能够捕获长期的时空相关性。然而，其复杂的架构导致参数量大和训练时间长，限制了它们的实际应用和全球范围内的预测规模。本论文旨在探索准确的天气预报的关键因素，并设计更有效的解决方案。有趣的是，我们的实验证明绝对位置编码在基于Transformer的天气预报模型中真正有效，可以明确地建模空间-时间的相关性，而无需注意力机制。我们理论上证明其有效性源于地理坐标和现实世界的时间特征的整合，这本身就是天气动态的内在关联。基于这一点，我们提出了一种轻量级且有效的站基式全球天气预报模型——LightWeather。我们将绝对位置编码和其他组件替换为其他部件，如Transformer中的其他组件。用不到3万参数和不到一小时的训练时间，LightWeather在与最先进的DL方法相比，在全球天气数据集上实现了最先进的性能。结果强调了整合空间-时间知识的重要性，而不是复杂架构的优势，为天气预报领域的深度学习提供了新的见解。

---

## Unsupervised Machine Learning Hybrid Approach Integrating Linear Programming in Loss Function_ A Robust Optimization Technique

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09967v1)

### Abstract (English)

This paper presents a novel hybrid approach that integrates linear
programming (LP) within the loss function of an unsupervised machine learning
model. By leveraging the strengths of both optimization techniques and machine
learning, this method introduces a robust framework for solving complex
optimization problems where traditional methods may fall short. The proposed
approach encapsulates the constraints and objectives of a linear programming
problem directly into the loss function, guiding the learning process to adhere
to these constraints while optimizing the desired outcomes. This technique not
only preserves the interpretability of linear programming but also benefits
from the flexibility and adaptability of machine learning, making it
particularly well-suited for unsupervised or semi-supervised learning
scenarios.

### 摘要 (中文)

这篇论文提出了一种新的混合方法，它在无监督机器学习模型的损失函数中结合线性规划（LP）。通过利用优化技术与机器学习的双重优势，这种方法引入了一个解决复杂优化问题的强大框架，在传统方法可能失效的情况下。提出的这种方法直接将线性规划问题的约束和目标映射到损失函数中，引导学习过程遵循这些约束并优化期望结果。这种技术不仅保留了线性规划的可解释性，而且受益于机器学习的灵活性和适应性，使其特别适合于无监督或半监督学习场景。

---

## Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10075v1)

### Abstract (English)

Reinforcement Learning from Human Feedback (RLHF) is a powerful paradigm for
aligning foundation models to human values and preferences. However, current
RLHF techniques cannot account for the naturally occurring differences in
individual human preferences across a diverse population. When these
differences arise, traditional RLHF frameworks simply average over them,
leading to inaccurate rewards and poor performance for individual subgroups. To
address the need for pluralistic alignment, we develop a class of multimodal
RLHF methods. Our proposed techniques are based on a latent variable
formulation - inferring a novel user-specific latent and learning reward models
and policies conditioned on this latent without additional user-specific data.
While conceptually simple, we show that in practice, this reward modeling
requires careful algorithmic considerations around model architecture and
reward scaling. To empirically validate our proposed technique, we first show
that it can provide a way to combat underspecification in simulated control
problems, inferring and optimizing user-specific reward functions. Next, we
conduct experiments on pluralistic language datasets representing diverse user
preferences and demonstrate improved reward function accuracy. We additionally
show the benefits of this probabilistic framework in terms of measuring
uncertainty, and actively learning user preferences. This work enables learning
from diverse populations of users with divergent preferences, an important
challenge that naturally occurs in problems from robot learning to foundation
model alignment.

### 摘要 (中文)

强化学习从人类反馈（RLHF）是将基础模型与人类价值观和偏好相匹配的强大范式。然而，当前的RLHF技术无法考虑不同人口中个体人类偏好的自然差异。当这些差异出现时，传统RLHF框架仅对它们进行平均处理，导致针对特定子群的不准确奖励和不良表现。为了应对多元化的配对需求，我们开发了一类多模态RLHF方法。我们的提出的方法基于隐变量表述——推断出一种新型用户特定的隐变量和学习奖励模型及其政策，并且在没有额外用户特定数据的情况下无需额外用户特定数据。尽管概念上简单，但我们表明，在实践中，这种奖励建模需要围绕模型架构和奖励尺度进行仔细算法考量。为了实证验证我们的提出的技术，首先展示了它可以在模拟控制问题中提供对抗不足的方法，推断并优化用户特定的奖励函数。其次，我们在代表多样用户偏好的多语种数据集上进行了实验，展示了改进后的奖励函数精度。此外，还显示了这个概率框架在测量不确定性方面的益处，并积极地学习用户偏好。这项工作使从具有分歧偏好的人群中学习成为可能，这对于机器人学习到基础模型对齐等自然存在的问题都是一个重要的挑战。

---

## No Screening is More Efficient with Multiple Objects

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10077v1)

### Abstract (English)

We study efficient mechanism design for allocating multiple heterogeneous
objects. We aim to maximize the residual surplus, the total value generated
from an allocation minus the costs for screening agents' values. We discover a
robust trend indicating that no-screening mechanisms such as serial
dictatorship with exogenous priority order tend to perform better as the
variety of goods increases. We analyze the underlying reasons by characterizing
efficient mechanisms in a stylized environment. We also apply an automated
mechanism design approach to numerically derive efficient mechanisms and
validate the trend in general environments. Building on this implication, we
propose the register-invite-book system (RIB) as an efficient system for
scheduling vaccination against pandemic diseases.

### 摘要 (中文)

我们研究了分配多个异质性对象的高效机制设计。我们的目标是最大化剩余盈余，即从分配中产生的总价值减去筛选代理的价值成本。我们发现一个稳健的趋势表明，如外生优先顺序序列独裁者这样的无筛选机制倾向于随着商品种类增加而表现得更好。我们通过分析有效机制在简化环境中来分析这一原因。我们也应用自动机制设计方法对数值求解出有效的机制，并验证了这一趋势在一般环境中的普遍性。基于这一点，我们提出了注册邀请书系统（RIB），这是一种高效的疫苗预约系统，用于预防流行病。

---

## Learning Brave Assumption-Based Argumentation Frameworks via ASP

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10126v1)

### Abstract (English)

Assumption-based Argumentation (ABA) is advocated as a unifying formalism for
various forms of non-monotonic reasoning, including logic programming. It
allows capturing defeasible knowledge, subject to argumentative debate. While,
in much existing work, ABA frameworks are given up-front, in this paper we
focus on the problem of automating their learning from background knowledge and
positive/negative examples. Unlike prior work, we newly frame the problem in
terms of brave reasoning under stable extensions for ABA. We present a novel
algorithm based on transformation rules (such as Rote Learning, Folding,
Assumption Introduction and Fact Subsumption) and an implementation thereof
that makes use of Answer Set Programming. Finally, we compare our technique to
state-of-the-art ILP systems that learn defeasible knowledge.

### 摘要 (中文)

基于假设的论证（ABA）被倡导为非单调推理的各种形式的统一正式语言，包括逻辑编程。它允许捕获不坚定的知识，在争论中受到论证辩论。虽然，在现有的大多数工作中，ABA框架是在预先给出的，但在本论文中，我们专注于从背景知识和正面/负面例子自动学习它们的问题。与先前的工作不同，我们在ABA框架中重新定义了问题，将其视为稳定扩展下勇敢推理的新领域。我们提出了一种基于转换规则（例如Rote Learning、折叠、假设引入和事实归约）的新算法，并提供了一个使用答案集程序实现该算法的版本。最后，我们将我们的技术与最先进的ILP系统进行比较，这些系统学习不坚定的知识。

---

## KAN 2.0_ Kolmogorov-Arnold Networks Meet Science

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10205v1)

### Abstract (English)

A major challenge of AI + Science lies in their inherent incompatibility:
today's AI is primarily based on connectionism, while science depends on
symbolism. To bridge the two worlds, we propose a framework to seamlessly
synergize Kolmogorov-Arnold Networks (KANs) and science. The framework
highlights KANs' usage for three aspects of scientific discovery: identifying
relevant features, revealing modular structures, and discovering symbolic
formulas. The synergy is bidirectional: science to KAN (incorporating
scientific knowledge into KANs), and KAN to science (extracting scientific
insights from KANs). We highlight major new functionalities in the pykan
package: (1) MultKAN: KANs with multiplication nodes. (2) kanpiler: a KAN
compiler that compiles symbolic formulas into KANs. (3) tree converter: convert
KANs (or any neural networks) to tree graphs. Based on these tools, we
demonstrate KANs' capability to discover various types of physical laws,
including conserved quantities, Lagrangians, symmetries, and constitutive laws.

### 摘要 (中文)

人工智能+科学的挑战之一在于它们内在的不兼容性：
今天的AI主要基于联结主义，而科学依赖于符号。为了连接这两个世界，我们提出了一种框架，无缝整合科洛莫格罗夫-阿纳托尔神经网络（KANs）和科学。这个框架强调了KANs在三个方面进行科学研究的功能：识别相关特征、揭示模块结构以及发现符号公式。这种协同是双向的：科学到KAN（将科学知识融入KAN中），和KAN到科学（从KAN中提取科学洞察）。我们突出显示pykan包中的几个新功能：（1）多KAN：具有乘法节点的KANs。（2）kanpiler：一个编译器，将符号公式转换为KANs。（3）树转换器：将KANs（或任何神经网络）转换为树图。基于这些工具，我们展示了KANs发现各种物理定律的能力，包括守恒量、拉格朗日方程、对称性和构成定律。

---

## Confirmation Bias in Gaussian Mixture Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09718v1)

### Abstract (English)

Confirmation bias, the tendency to interpret information in a way that aligns
with one's preconceptions, can profoundly impact scientific research, leading
to conclusions that reflect the researcher's hypotheses even when the
observational data do not support them. This issue is especially critical in
scientific fields involving highly noisy observations, such as cryo-electron
microscopy.
  This study investigates confirmation bias in Gaussian mixture models. We
consider the following experiment: A team of scientists assumes they are
analyzing data drawn from a Gaussian mixture model with known signals
(hypotheses) as centroids. However, in reality, the observations consist
entirely of noise without any informative structure. The researchers use a
single iteration of the K-means or expectation-maximization algorithms, two
popular algorithms to estimate the centroids. Despite the observations being
pure noise, we show that these algorithms yield biased estimates that resemble
the initial hypotheses, contradicting the unbiased expectation that averaging
these noise observations would converge to zero. Namely, the algorithms
generate estimates that mirror the postulated model, although the hypotheses
(the presumed centroids of the Gaussian mixture) are not evident in the
observations. Specifically, among other results, we prove a positive
correlation between the estimates produced by the algorithms and the
corresponding hypotheses. We also derive explicit closed-form expressions of
the estimates for a finite and infinite number of hypotheses. This study
underscores the risks of confirmation bias in low signal-to-noise environments,
provides insights into potential pitfalls in scientific methodologies, and
highlights the importance of prudent data interpretation.

### 摘要 (中文)

确认偏见，即根据预设观点解读信息的现象，可以深刻影响科学研究，即使观察数据与之不相符，结论也反映在研究者的假设上。这一问题尤其严重，在涉及噪声观测的科学领域，如冷冻电子显微镜。

这项研究调查了高斯混合模型中的确认偏见。我们考虑以下实验：一群科学家认为他们正在分析来自已知信号（假设）作为中心点的数据的高斯混合模型。然而，在现实生活中，观测完全是由噪声没有结构的信息。研究人员使用单个迭代的K均值或期望最大化算法来估计中心点，这两种流行算法之一。尽管观察是纯粹的噪音，但我们证明这些算法生成的估计反映了初始假设，这与平均这些无结构观察量会收敛到零的无偏预期相矛盾。换句话说，算法生成的估计类似于预设模型，虽然假设（高斯混合模型的拟合中心）不在观察中明显。具体来说，除了其他结果外，我们证明了由算法产生的估计和相应假设之间的积极相关性。我们还导出了有限和无限数目的假设下对数的闭式表达形式的估计。这项研究强调了低信噪比环境中确认偏见的风险，提供了科学方法潜在陷阱的理解，并强调了谨慎的数据解释的重要性。

---

## Robust spectral clustering with rank statistics

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10136v1)

### Abstract (English)

This paper analyzes the statistical performance of a robust spectral
clustering method for latent structure recovery in noisy data matrices. We
consider eigenvector-based clustering applied to a matrix of nonparametric rank
statistics that is derived entrywise from the raw, original data matrix. This
approach is robust in the sense that, unlike traditional spectral clustering
procedures, it can provably recover population-level latent block structure
even when the observed data matrix includes heavy-tailed entries and has a
heterogeneous variance profile.
  Our main theoretical contributions are threefold and hold under flexible data
generating conditions. First, we establish that robust spectral clustering with
rank statistics can consistently recover latent block structure, viewed as
communities of nodes in a graph, in the sense that unobserved community
memberships for all but a vanishing fraction of nodes are correctly recovered
with high probability when the data matrix is large. Second, we refine the
former result and further establish that, under certain conditions, the
community membership of any individual, specified node of interest can be
asymptotically exactly recovered with probability tending to one in the
large-data limit. Third, we establish asymptotic normality results associated
with the truncated eigenstructure of matrices whose entries are rank
statistics, made possible by synthesizing contemporary entrywise matrix
perturbation analysis with the classical nonparametric theory of so-called
simple linear rank statistics. Collectively, these results demonstrate the
statistical utility of rank-based data transformations when paired with
spectral techniques for dimensionality reduction. Additionally, for a dataset
of human connectomes, our approach yields parsimonious dimensionality reduction
and improved recovery of ground-truth neuroanatomical cluster structure.

### 摘要 (中文)

这篇论文分析了在有噪声数据矩阵中恢复隐含结构的稳健谱聚类统计性能。我们考虑了一种基于特征向量的聚类方法，该方法是从原始数据矩阵中的每一行逐个提取非参数秩统计数据来实现的。这种方法是鲁棒的，在传统上，它只能证明当观察到的数据矩阵包含尾部重大的元素时，可以正确地恢复总体水平的隐块结构，即使这些观察到的数据矩阵中含有异质性方差分布。
我们的主要理论贡献是三方面的，并且在灵活的数据生成条件下保持不变。首先，我们证明了使用秩统计数据进行稳健谱聚类能够有效恢复隐块结构，即作为图中的节点群，当数据矩阵足够大时，对但几乎所有的节点而言，所有未观测的社区成员的归属都是正确的概率很高。其次，我们进一步完善了前者的结果，并进一步确立，对于感兴趣的任何单个节点，可以根据条件收敛于概率趋于一的极限值，其所属社区的归属可以准确地得到。最后，我们确立了与矩阵中元素是秩统计数据的相关联的截断主成分结构的渐近正态性结果，这是通过结合当代的行列式矩阵扰动分析和经典简单线性秩统计数据的古典非参数理论实现的。这些结果展示了当与降维技术（如谱聚类）一起使用秩统计数据时，数据变换的有效性。此外，对于人类连接脑成像数据集，我们的方法提供了简洁的维度降低和改进的真神经解剖结构的回归。

---

## Molecular Graph Representation Learning Integrating Large Language Models with Domain-specific Small Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10124v1)

### Abstract (English)

Molecular property prediction is a crucial foundation for drug discovery. In
recent years, pre-trained deep learning models have been widely applied to this
task. Some approaches that incorporate prior biological domain knowledge into
the pre-training framework have achieved impressive results. However, these
methods heavily rely on biochemical experts, and retrieving and summarizing
vast amounts of domain knowledge literature is both time-consuming and
expensive. Large Language Models (LLMs) have demonstrated remarkable
performance in understanding and efficiently providing general knowledge.
Nevertheless, they occasionally exhibit hallucinations and lack precision in
generating domain-specific knowledge. Conversely, Domain-specific Small Models
(DSMs) possess rich domain knowledge and can accurately calculate molecular
domain-related metrics. However, due to their limited model size and singular
functionality, they lack the breadth of knowledge necessary for comprehensive
representation learning. To leverage the advantages of both approaches in
molecular property prediction, we propose a novel Molecular Graph
representation learning framework that integrates Large language models and
Domain-specific small models (MolGraph-LarDo). Technically, we design a
two-stage prompt strategy where DSMs are introduced to calibrate the knowledge
provided by LLMs, enhancing the accuracy of domain-specific information and
thus enabling LLMs to generate more precise textual descriptions for molecular
samples. Subsequently, we employ a multi-modal alignment method to coordinate
various modalities, including molecular graphs and their corresponding
descriptive texts, to guide the pre-training of molecular representations.
Extensive experiments demonstrate the effectiveness of the proposed method.

### 摘要 (中文)

分子属性预测是药物发现的关键基础。近年来，预训练深度学习模型广泛应用于这一任务。一些将先验生物领域知识纳入预训练框架的方法取得了显著成果。然而，这些方法依赖于生化专家，检索和总结大量领域知识文献既耗时又昂贵。大型语言模型（LLMs）在理解和高效提供一般知识方面表现出色。不过，它们偶尔会出现幻觉，并且在生成特定领域的知识上缺乏精确性。相反，专门的小型模型（DSMs）拥有丰富的领域知识，可以准确计算与分子域相关的指标。然而，由于其较小的模型规模和单一功能，它们缺少全面表示学习所需的广度的知识。为了利用这两种方法在分子属性预测中的优势，我们提出了一种新型的分子图表示学习框架，该框架结合了大型语言模型和专门的小型模型（MolGraph-LarDo）。技术上，我们设计了一个引导策略，其中DSMs被引入到由LLMs提供的知识中进行校准，从而提高特定信息的准确性，从而允许LLMs为分子样本生成更精确的文本描述。随后，我们采用多模态配对方法来协调各种模式，包括分子图及其相应的描述文本，以指导分子表示的预训练。广泛的实验表明提出的策略的有效性。

---

## A Transcription Prompt-based Efficient Audio Large Language Model for Robust Speech Recognition

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09491v1)

### Abstract (English)

Audio-LLM introduces audio modality into a large language model (LLM) to
enable a powerful LLM to recognize, understand, and generate audio. However,
during speech recognition in noisy environments, we observed the presence of
illusions and repetition issues in audio-LLM, leading to substitution and
insertion errors. This paper proposes a transcription prompt-based audio-LLM by
introducing an ASR expert as a transcription tokenizer and a hybrid
Autoregressive (AR) Non-autoregressive (NAR) decoding approach to solve the
above problems. Experiments on 10k-hour WenetSpeech Mandarin corpus show that
our approach decreases 12.2% and 9.6% CER relatively on Test_Net and
Test_Meeting evaluation sets compared with baseline. Notably, we reduce the
decoding repetition rate on the evaluation set to zero, showing that the
decoding repetition problem has been solved fundamentally.

### 摘要 (中文)

音频LLM将音频模式引入大型语言模型（LLM），以便强大的LLM能够识别、理解并生成音频。然而，在嘈杂的环境中进行语音识别时，我们观察到音频LLM中存在幻觉和重复问题，导致替换和插入错误。本文提出了一种基于转录提示的音频LLM，通过引入ASR专家作为转录分词器和混合自回归（AR）非自回归（NAR）解码方法来解决上述问题。在10K小时WenetSpeech普通话语料库上进行实验，结果显示与基准线相比，我们的方法减少了12.2％和9.6％的CER相对值在测试网络和会议评估集上的测试分数分别降低了12.2％和9.6％。值得注意的是，我们在评价集中编码重复率降至零，显示基本解决了编码重复问题。

---

## Efficient Area-based and Speaker-Agnostic Source Separation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09810v1)

### Abstract (English)

This paper introduces an area-based source separation method designed for
virtual meeting scenarios. The aim is to preserve speech signals from an
unspecified number of sources within a defined spatial area in front of a
linear microphone array, while suppressing all other sounds. Therefore, we
employ an efficient neural network architecture adapted for multi-channel input
to encompass the predefined target area. To evaluate the approach, training
data and specific test scenarios including multiple target and interfering
speakers, as well as background noise are simulated. All models are rated
according to DNSMOS and scale-invariant signal-to-distortion ratio. Our
experiments show that the proposed method separates speech from multiple
speakers within the target area well, besides being of very low complexity,
intended for real-time processing. In addition, a power reduction heatmap is
used to demonstrate the networks' ability to identify sources located within
the target area. We put our approach in context with a well-established
baseline for speaker-speaker separation and discuss its strengths and
challenges.

### 摘要 (中文)

这篇论文介绍了一种针对虚拟会议场景的区域源分离方法。目标是保留前方线性麦克风阵列中定义的空间范围内指定数量的声源中的语音信号，同时抑制其他所有声音。因此，我们采用了一个适应多通道输入的高效神经网络架构来涵盖预定义的目标区域。为了评估这一方法，训练数据和包括多个目标和干扰源在内的具体测试场景以及背景噪音都被模拟出来。所有的模型根据DNSMOS和等价信号失真比的标准进行评级。我们的实验表明，该提出的方法在目标区域内很好地分离出多个演讲者的声音，并且具有非常低的复杂度，适合实时处理。此外，我们还使用功率降低热图来展示网络能够识别目标区域内处于预定位置的声源的能力。我们将这种方法与一个已建立的基础模型进行比较，讨论其优点和挑战。

---

## Auptimize_ Optimal Placement of Spatial Audio Cues for Extended Reality

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09320v1)

### Abstract (English)

Spatial audio in Extended Reality (XR) provides users with better awareness
of where virtual elements are placed, and efficiently guides them to events
such as notifications, system alerts from different windows, or approaching
avatars. Humans, however, are inaccurate in localizing sound cues, especially
with multiple sources due to limitations in human auditory perception such as
angular discrimination error and front-back confusion. This decreases the
efficiency of XR interfaces because users misidentify from which XR element a
sound is coming. To address this, we propose Auptimize, a novel computational
approach for placing XR sound sources, which mitigates such localization errors
by utilizing the ventriloquist effect. Auptimize disentangles the sound source
locations from the visual elements and relocates the sound sources to optimal
positions for unambiguous identification of sound cues, avoiding errors due to
inter-source proximity and front-back confusion. Our evaluation shows that
Auptimize decreases spatial audio-based source identification errors compared
to playing sound cues at the paired visual-sound locations. We demonstrate the
applicability of Auptimize for diverse spatial audio-based interactive XR
scenarios.

### 摘要 (中文)

在扩展现实（XR）中，空间音频为用户提供了虚拟元素放置位置的更好意识，并高效地引导他们识别来自不同窗口的通知、系统警告或其他逼近的拟人角色。然而，人类在定位声学线索方面存在误差，特别是在多源的情况下，由于人类听觉感知中的角度鉴别错误和前后混淆等因素而受到限制。这降低了XR接口的效率，因为用户从哪个XR元素听到声音误认为是来自其他XR元素。因此，我们提出了一种新的计算方法——Auptimize，用于放置XR声源，该方法通过利用哑剧效果来缓解这种定位误差。Auptimize将声音源的位置与视觉元素分开并重新定位到最优位置，以明确识别声学线索，避免由于近邻源或前后混淆导致的声音来源的定位误差。我们的评估显示，在空间音频基于的源识别错误方面，Auptimize比在对齐的视音频位置播放声音线索减少了空间音频。我们展示了Auptimize适用于各种基于空间音频的互动XR场景的应用性。

---

## __mathbb_BEHR__NOULLI_ A Binary EHR Data-Oriented Medication Recommendation System

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09410v1)

### Abstract (English)

The medical community believes binary medical event outcomes in EHR data
contain sufficient information for making a sensible recommendation. However,
there are two challenges to effectively utilizing such data: (1) modeling the
relationship between massive 0,1 event outcomes is difficult, even with expert
knowledge; (2) in practice, learning can be stalled by the binary values since
the equally important 0 entries propagate no learning signals. Currently, there
is a large gap between the assumed sufficient information and the reality that
no promising results have been shown by utilizing solely the binary data:
visiting or secondary information is often necessary to reach acceptable
performance. In this paper, we attempt to build the first successful binary EHR
data-oriented drug recommendation system by tackling the two difficulties,
making sensible drug recommendations solely using the binary EHR medical
records. To this end, we take a statistical perspective to view the EHR data as
a sample from its cohorts and transform them into continuous Bernoulli
probabilities. The transformed entries not only model a deterministic binary
event with a distribution but also allow reflecting \emph{event-event}
relationship by conditional probability. A graph neural network is learned on
top of the transformation. It captures event-event correlations while
emphasizing \emph{event-to-patient} features. Extensive results demonstrate
that the proposed method achieves state-of-the-art performance on large-scale
databases, outperforming baseline methods that use secondary information by a
large margin. The source code is available at
\url{https://github.com/chenzRG/BEHRMecom}

### 摘要 (中文)

医学界认为，基于电子病历（EHR）数据的二元医疗事件结果包含足够的信息，可以做出明智的推荐。然而，使用这些数据的有效利用存在两个挑战：(1)建模大规模0-1事件结果之间的关系是困难的，即使有专家知识；(2)在实践中，由于重要的0项会传播学习信号而使学习受阻。目前，假设的信息充足程度与实际中显示的无希望结果之间存在着巨大的差距：仅使用二进制数据时往往需要访问或辅助信息才能达到可接受的表现。在此文中，我们试图通过解决这两个难题来建立第一个成功的基于二进制EHR数据的药物推荐系统，只使用二进制EHR医疗记录进行明智的药物推荐。为此，我们从统计学的角度看待EHR数据，将其视为其群组的一个样本，并对其进行连续伯努利概率的转换。转换后的项目不仅模型了一个确定的二进制事件的概率分布，而且还可以通过条件概率来反映“事件-事件”关系。在顶部学习了图神经网络。它捕捉到了事件-事件的相关性，同时强调“事件到患者”的特征。广泛的成果表明，所提出的策略在大型数据库上取得了最先进的表现，比使用辅助信息的方法显示出较大的优势。源代码可在以下网址获取：
https://github.com/chenzRG/BEHRMecom

---

## ALS-HAR_ Harnessing Wearable Ambient Light Sensors to Enhance IMU-based HAR

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09527v1)

### Abstract (English)

Despite the widespread integration of ambient light sensors (ALS) in smart
devices commonly used for screen brightness adaptation, their application in
human activity recognition (HAR), primarily through body-worn ALS, is largely
unexplored. In this work, we developed ALS-HAR, a robust wearable light-based
motion activity classifier. Although ALS-HAR achieves comparable accuracy to
other modalities, its natural sensitivity to external disturbances, such as
changes in ambient light, weather conditions, or indoor lighting, makes it
challenging for daily use. To address such drawbacks, we introduce strategies
to enhance environment-invariant IMU-based activity classifications through
augmented multi-modal and contrastive classifications by transferring the
knowledge extracted from the ALS. Our experiments on a real-world activity
dataset for three different scenarios demonstrate that while ALS-HAR's accuracy
strongly relies on external lighting conditions, cross-modal information can
still improve other HAR systems, such as IMU-based classifiers.Even in
scenarios where ALS performs insufficiently, the additional knowledge enables
improved accuracy and macro F1 score by up to 4.2 % and 6.4 %, respectively,
for IMU-based classifiers and even surpasses multi-modal sensor fusion models
in two of our three experiment scenarios. Our research highlights the untapped
potential of ALS integration in advancing sensor-based HAR technology, paving
the way for practical and efficient wearable ALS-based activity recognition
systems with potential applications in healthcare, sports monitoring, and smart
indoor environments.

### 摘要 (中文)

尽管在用于屏幕亮度调整的智能手机等智能设备中广泛地安装了环境光传感器（ALS），但它们在人类活动识别（HAR）方面，主要是通过穿戴式ALS进行应用的研究相对较少。在此工作中，我们开发了一种名为ALS-HAR的可穿戴光源运动活动分类器，虽然ALS-HAR的准确率与其他模态相当，但由于外部干扰因素如照明条件的变化、天气条件或室内照明等因素的影响较大，这使得日常使用变得具有挑战性。为了应对这些缺点，我们将ALS知识从IMU中迁移至增强多模态和对抗类别的转移学习中，以提高环境适应性的IMU基于活动分类。我们在三个不同场景的真实世界活动中数据集上进行了实验，发现ALS-HAR的准确性主要依赖于外部光线条件，而交叉模式信息仍可以改善其他HAR系统，例如IMU基分类器。即使ALS表现不佳，在仅需要额外的知识的情况下，IMU基分类器的准确性和macro F1分数也有显著提升，分别可达4.2%和6.4%，对于IMU基分类器而言，甚至超过了我们的三个实验场景中的融合传感器模型。

我们的研究揭示了ALS集成在推进基于传感器的人体活动识别技术方面的潜力未被充分利用，开辟了实际可行且高效的穿戴式ALS-基础人体活动识别系统的道路，该系统有望应用于医疗保健、体育监测以及智能室内环境等领域。

---

## PA-LLaVA_ A Large Language-Vision Assistant for Human Pathology Image Understanding

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09530v1)

### Abstract (English)

The previous advancements in pathology image understanding primarily involved
developing models tailored to specific tasks. Recent studies has demonstrated
that the large vision-language model can enhance the performance of various
downstream tasks in medical image understanding. In this study, we developed a
domain-specific large language-vision assistant (PA-LLaVA) for pathology image
understanding. Specifically, (1) we first construct a human pathology
image-text dataset by cleaning the public medical image-text data for
domain-specific alignment; (2) Using the proposed image-text data, we first
train a pathology language-image pretraining (PLIP) model as the specialized
visual encoder for pathology image, and then we developed scale-invariant
connector to avoid the information loss caused by image scaling; (3) We adopt
two-stage learning to train PA-LLaVA, first stage for domain alignment, and
second stage for end to end visual question \& answering (VQA) task. In
experiments, we evaluate our PA-LLaVA on both supervised and zero-shot VQA
datasets, our model achieved the best overall performance among multimodal
models of similar scale. The ablation experiments also confirmed the
effectiveness of our design. We posit that our PA-LLaVA model and the datasets
presented in this work can promote research in field of computational
pathology. All codes are available at:
https://github.com/ddw2AIGROUP2CQUPT/PA-LLaVA}{https://github.com/ddw2AIGROUP2CQUPT/PA-LLaVA

### 摘要 (中文)

之前在病理图像理解领域取得的进步主要集中在开发针对特定任务的模型上。最近的研究已经展示了大型视觉语言模型在医学影像理解的各种下游任务中可以显著提升性能。在此研究中，我们开发了一个专用于病理图像理解的人工智能助手（PA-LLaVA）。具体来说，(1)我们首先通过清除公共医疗图像文本数据中的特定域对齐来构建人类病理图像-文本数据集；(2)使用提出的图像-文本数据，我们首先训练一种专用于病理图像的语言-图像预训练（PLIP）模型作为病理图像的专门视觉编码器，并然后开发一个尺度不变连接器以避免由于图像缩放导致的信息损失；(3)采用两阶段学习的方法训练PA-LLaVA，第一阶段进行领域对齐，第二阶段为端到端的视觉问题和回答（VQA）任务。在实验中，我们在监督式和零样本VQA数据集上的评估中比较了我们的PA-LLaVA模型，在相似规模的多模态模型中取得了最佳的整体性能。拆分实验也证实了设计的有效性。我们认为，我们的PA-LLaVA模型及其呈现的工作可以在计算病理学领域的研究推广中发挥重要作用。所有代码都可在以下链接找到：
https://github.com/ddw2AIGROUP2CQUPT/PA-LLaVA{https://github.com/ddw2AIGROUP2CQUPT/PA-LLaVA

---

## SynTraC_ A Synthetic Dataset for Traffic Signal Control from Traffic Monitoring Cameras

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09588v1)

### Abstract (English)

This paper introduces SynTraC, the first public image-based traffic signal
control dataset, aimed at bridging the gap between simulated environments and
real-world traffic management challenges. Unlike traditional datasets for
traffic signal control which aim to provide simplified feature vectors like
vehicle counts from traffic simulators, SynTraC provides real-style images from
the CARLA simulator with annotated features, along with traffic signal states.
This image-based dataset comes with diverse real-world scenarios, including
varying weather and times of day. Additionally, SynTraC also provides different
reward values for advanced traffic signal control algorithms like reinforcement
learning. Experiments with SynTraC demonstrate that it is still an open
challenge to image-based traffic signal control methods compared with
feature-based control methods, indicating our dataset can further guide the
development of future algorithms. The code for this paper can be found in
\url{https://github.com/DaRL-LibSignal/SynTraC}.SynTraC

### 摘要 (中文)

这篇论文介绍了第一张公共的基于图像的交通信号控制数据集，名为SynTraC，旨在填补模拟环境和实际交通管理挑战之间的差距。与传统用于交通信号控制的数据集不同，这些数据集的目标是提供简化特征向量，如从交通仿真器中获取的车辆计数等。相反，SynTraC通过使用来自CARLA模拟器的真实样式图片，并带有标注特征，以及交通信号状态，提供了实拍的图片数据集。这个基于图片的数据集包含各种真实世界场景，包括不同的天气条件和不同时段。此外，SynTraC还为先进的交通信号控制系统算法（例如强化学习）提供了不同的奖励值。实验表明，相比基于特征的控制方法，基于图像的交通信号控制方法仍然是一个开放的研究课题，这表明我们的数据集可以进一步指导未来算法的发展。本文代码可在以下链接找到：\url{https://github.com/DaRL-LibSignal/SynTraC}

---

## Moonshine_ Distilling Game Content Generators into Steerable Generative Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09594v1)

### Abstract (English)

Procedural Content Generation via Machine Learning (PCGML) has enhanced game
content creation, yet challenges in controllability and limited training data
persist. This study addresses these issues by distilling a constructive PCG
algorithm into a controllable PCGML model. We first generate a large amount of
content with a constructive algorithm and label it using a Large Language Model
(LLM). We use these synthetic labels to condition two PCGML models for
content-specific generation, a diffusion model and the five-dollar model. This
neural network distillation process ensures that the generation aligns with the
original algorithm while introducing controllability through plain text. We
define this text-conditioned PCGML as a Text-to-game-Map (T2M) task, offering
an alternative to prevalent text-to-image multi-modal tasks. We compare our
distilled models with the baseline constructive algorithm. Our analysis of the
variety, accuracy, and quality of our generation demonstrates the efficacy of
distilling constructive methods into controllable text-conditioned PCGML
models.

### 摘要 (中文)

通过机器学习（PCGML）增强游戏内容生成，但控制能力挑战和有限训练数据问题仍然存在。本研究通过将建设性PCG算法提炼成可控的PCGML模型来解决这些问题。我们首先使用一个构建性的算法生成大量内容，并利用大型语言模型（LLM）对其进行标签。我们将这些合成标签用于条件两个PCGML模型进行特定内容生成，一个是扩散模型，另一个是五美元模型。这个神经网络微调过程确保了生成与原始算法保持一致，同时引入文本控制以实现自然语言处理。我们将这种带有文本条件的PCGML定义为文本到游戏地图（T2M）任务，提供了一种替代现有基于多模态任务的文本到图像任务。我们将提炼后的模型与基线构建性算法进行了比较。我们的分析显示，对各种、准确性和质量的生成演示了将构建方法提炼为可控文本条件下的PCGML模型的有效性。

---

## Does Thought Require Sensory Grounding_ From Pure Thinkers to Large Language Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09605v1)

### Abstract (English)

Does the capacity to think require the capacity to sense? A lively debate on
this topic runs throughout the history of philosophy and now animates
discussions of artificial intelligence. I argue that in principle, there can be
pure thinkers: thinkers that lack the capacity to sense altogether. I also
argue for significant limitations in just what sort of thought is possible in
the absence of the capacity to sense. Regarding AI, I do not argue directly
that large language models can think or understand, but I rebut one important
argument (the argument from sensory grounding) that they cannot. I also use
recent results regarding language models to address the question of whether or
how sensory grounding enhances cognitive capacities.

### 摘要 (中文)

关于思考能力是否需要感知能力这一话题，哲学史上一直有激烈的争论，并且现在正在人工智能讨论中激化。我主张，在理论上，可以存在纯粹的思考者：没有完全丧失感知能力的思考者。我也认为在没有感知能力的情况下，可能存在的思维类型有限制。至于人工智能，我没有直接提出大语言模型能否思考或理解的观点，但我反驳了一个重要的论点（感觉地根植论），即它们不能。我还用最近有关语言模型的结果来回答“感官根植是否增强认知能力”的问题。

---

## On the Foundations of Conflict-Driven Solving for Hybrid MKNF Knowledge Bases

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09626v1)

### Abstract (English)

Hybrid MKNF Knowledge Bases (HMKNF-KBs) constitute a formalism for tightly
integrated reasoning over closed-world rules and open-world ontologies. This
approach allows for accurate modeling of real-world systems, which often rely
on both categorical and normative reasoning. Conflict-driven solving is the
leading approach for computationally hard problems, such as satisfiability
(SAT) and answer set programming (ASP), in which MKNF is rooted. This paper
investigates the theoretical underpinnings required for a conflict-driven
solver of HMKNF-KBs. The approach defines a set of completion and loop
formulas, whose satisfaction characterizes MKNF models. This forms the basis
for a set of nogoods, which in turn can be used as the backbone for a
conflict-driven solver.

### 摘要 (中文)

混合MKNF知识库（HMKNF-KB）是一种封闭世界规则和开放世界元知识的正式语言。这种方法允许对现实世界的系统进行准确建模，这些系统经常依赖于既有的逻辑推理和规范性推理。冲突驱动解决问题是计算难题中解决Satisfiability（SAT）和答案集编程（ASP）问题的主要方法，其中MKNF起源于这里。本论文研究了用于HMKNF-KB冲突驱动求解器所需的理论基础。该方法定义了一组完成和循环公式，其满足度决定了MKNF模型。这形成了一个无硝石的基础，后者可以作为冲突驱动求解器的核心。

---

## Simulating Field Experiments with Large Language Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09682v1)

### Abstract (English)

Prevailing large language models (LLMs) are capable of human responses
simulation through its unprecedented content generation and reasoning
abilities. However, it is not clear whether and how to leverage LLMs to
simulate field experiments. In this paper, we propose and evaluate two
prompting strategies: the observer mode that allows a direct prediction on main
conclusions and the participant mode that simulates distributions of responses
from participants. Using this approach, we examine fifteen well cited field
experimental papers published in INFORMS and MISQ, finding encouraging
alignments between simulated experimental results and the actual results in
certain scenarios. We further identify topics of which LLMs underperform,
including gender difference and social norms related research. Additionally,
the automatic and standardized workflow proposed in this paper enables the
possibility of a large-scale screening of more papers with field experiments.
This paper pioneers the utilization of large language models (LLMs) for
simulating field experiments, presenting a significant extension to previous
work which focused solely on lab environments. By introducing two novel
prompting strategies, observer and participant modes, we demonstrate the
ability of LLMs to both predict outcomes and replicate participant responses
within complex field settings. Our findings indicate a promising alignment with
actual experimental results in certain scenarios, achieving a stimulation
accuracy of 66% in observer mode. This study expands the scope of potential
applications for LLMs and illustrates their utility in assisting researchers
prior to engaging in expensive field experiments. Moreover, it sheds light on
the boundaries of LLMs when used in simulating field experiments, serving as a
cautionary note for researchers considering the integration of LLMs into their
experimental toolkit.

### 摘要 (中文)

当前的大语言模型（LLM）在内容生成和推理能力上具有人类水平的模拟能力，但如何利用这些模型来模拟实地实验仍然不清楚。本研究提出并评估两种提示策略：观察者模式，允许直接预测主要结论；参与者模式，模拟来自参与者的响应分布。通过这种方法，我们审查了15篇在INFORMS和MISQ发表的被引用的实地实验论文，发现模拟实验结果与实际结果之间存在鼓励的匹配情况。此外，提出的自动标准化工作流使大规模筛选更多关于实地实验的研究成为可能。本文开辟了使用大语言模型（LLM）进行实地实验模拟的新途径，这是之前仅关注实验室环境工作的扩展。通过引入两种新颖的提示策略，观察者和参与者模式，我们展示了LMMs不仅能够预测结果，而且能够在复杂场所以内的参与者响应中复制参与者。我们的发现表明，在某些情况下，模拟实验的结果与实际结果有积极的匹配，观察者模式下的刺激准确率为66%。这一研究表明，对于LMMs来说，扩大其潜在应用范围，并展示它们在研究人员在开始昂贵的实地实验前协助研究者的优势是很有意义的。此外，它还揭示了当用于模拟实地实验时，LLMs的边界，这为考虑将LMMs集成到他们的实验工具包中的研究人员提供了一个警告。

---

## Partial-Multivariate Model for Forecasting

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09703v1)

### Abstract (English)

When solving forecasting problems including multiple time-series features,
existing approaches often fall into two extreme categories, depending on
whether to utilize inter-feature information: univariate and
complete-multivariate models. Unlike univariate cases which ignore the
information, complete-multivariate models compute relationships among a
complete set of features. However, despite the potential advantage of
leveraging the additional information, complete-multivariate models sometimes
underperform univariate ones. Therefore, our research aims to explore a middle
ground between these two by introducing what we term Partial-Multivariate
models where a neural network captures only partial relationships, that is,
dependencies within subsets of all features. To this end, we propose PMformer,
a Transformer-based partial-multivariate model, with its training algorithm. We
demonstrate that PMformer outperforms various univariate and
complete-multivariate models, providing a theoretical rationale and empirical
analysis for its superiority. Additionally, by proposing an inference technique
for PMformer, the forecasting accuracy is further enhanced. Finally, we
highlight other advantages of PMformer: efficiency and robustness under missing
features.

### 摘要 (中文)

在解决包括多个时间序列特征在内的预测问题时，现有的方法往往倾向于两种极端的分类方式：单变量和完整多变量模型。与单一变量的情况不同，完整的多变量模型计算的是所有特征之间的一组完整关系。然而，尽管利用额外信息的优势是显而易见的，但有时完整的多变量模型的表现却不如单一变量模型。因此，我们的研究旨在探索这两个极端之间的中间地带，通过引入我们称之为部分多变量模型的概念，其中神经网络仅捕捉部分关系，即特定子集内的依赖关系。为此，我们提出了PMformer，一种基于Transformer的部分多变量模型，并提出了一种训练算法。我们展示了PMformer在各种单一变量和完整多变量模型中均优于它们，提供了理论依据和实证分析来支持其优越性。此外，通过提出对PMformer进行推断技术，预测准确性得到了进一步提高。最后，我们强调了PMformer的一些其他优势：对于缺失特征的效率和鲁棒性。

---

## MalLight_ Influence-Aware Coordinated Traffic Signal Control for Traffic Signal Malfunctions

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09768v1)

### Abstract (English)

Urban traffic is subject to disruptions that cause extended waiting time and
safety issues at signalized intersections. While numerous studies have
addressed the issue of intelligent traffic systems in the context of various
disturbances, traffic signal malfunction, a common real-world occurrence with
significant repercussions, has received comparatively limited attention. The
primary objective of this research is to mitigate the adverse effects of
traffic signal malfunction, such as traffic congestion and collision, by
optimizing the control of neighboring functioning signals. To achieve this
goal, this paper presents a novel traffic signal control framework (MalLight),
which leverages an Influence-aware State Aggregation Module (ISAM) and an
Influence-aware Reward Aggregation Module (IRAM) to achieve coordinated control
of surrounding traffic signals. To the best of our knowledge, this study
pioneers the application of a Reinforcement Learning(RL)-based approach to
address the challenges posed by traffic signal malfunction. Empirical
investigations conducted on real-world datasets substantiate the superior
performance of our proposed methodology over conventional and deep
learning-based alternatives in the presence of signal malfunction, with
reduction of throughput alleviated by as much as 48.6$\%$.

### 摘要 (中文)

城市交通受阻扰所造成的延长等待时间和安全问题。尽管有多项研究关注了各种干扰下的智能交通系统的主题，但信号故障这一常见的现实发生频率却相对较少受到关注。本研究的主要目标是通过优化邻近正常工作的信号控制来减轻由信号故障引起的负面影响，如交通拥堵和碰撞。为此，本文提出了一种新型的交通信号控制系统（MalLight），它利用感知影响的知识聚集模块（ISAM）和知识聚合奖励模块（IRAM）实现周围交通信号的协调控制。到目前为止，我们对所有已知的研究进行了深入分析，发现本研究开创性地应用了基于强化学习（RL）的方法来解决由信号故障带来的挑战。实证调查结果表明，在存在信号故障的情况下，我们的提出的解决方案在性能上优于传统和深度学习方法，特别是在缓解因信号故障导致的吞吐量减少方面，可提高48.6%。

---

## World Models Increase Autonomy in Reinforcement Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09807v1)

### Abstract (English)

Reinforcement learning (RL) is an appealing paradigm for training intelligent
agents, enabling policy acquisition from the agent's own autonomously acquired
experience. However, the training process of RL is far from automatic,
requiring extensive human effort to reset the agent and environments. To tackle
the challenging reset-free setting, we first demonstrate the superiority of
model-based (MB) RL methods in such setting, showing that a straightforward
adaptation of MBRL can outperform all the prior state-of-the-art methods while
requiring less supervision. We then identify limitations inherent to this
direct extension and propose a solution called model-based reset-free
(MoReFree) agent, which further enhances the performance. MoReFree adapts two
key mechanisms, exploration and policy learning, to handle reset-free tasks by
prioritizing task-relevant states. It exhibits superior data-efficiency across
various reset-free tasks without access to environmental reward or
demonstrations while significantly outperforming privileged baselines that
require supervision. Our findings suggest model-based methods hold significant
promise for reducing human effort in RL. Website:
https://sites.google.com/view/morefree

### 摘要 (中文)

强化学习（RL）是训练智能代理的吸引人范例，使政策从代理自身自动生成的经验自动获取。然而，RL的训练过程远非自动化的，需要大量的人力来重置代理和环境。为了应对挑战性的无重置设置，我们首先展示了在这样的设置中，基于模型（MB）RL方法的优势，证明了在不依赖监督的情况下，直接扩展MBRL可以超越所有先前的最先进的方法，同时要求更少的监督。然后，我们识别到这种直接延伸中存在的局限性，并提出了解决方案称为基于模型的重置免费（MoReFree）代理，进一步提高了性能。MoReFree通过优先处理任务相关的状态机制，利用重置的任务来调整探索和政策学习，以在没有访问环境奖励或示范的情况下，在各种重置任务中表现出色，显著优于依赖监督的特权基线。我们的发现表明基于模型的方法对减少RL中的人类努力具有重要意义。网站：https://sites.google.com/view/morefree

---

## TDNetGen_ Empowering Complex Network Resilience Prediction with Generative Augmentation of Topology and Dynamics

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09825v1)

### Abstract (English)

Predicting the resilience of complex networks, which represents the ability
to retain fundamental functionality amidst external perturbations or internal
failures, plays a critical role in understanding and improving real-world
complex systems. Traditional theoretical approaches grounded in nonlinear
dynamical systems rely on prior knowledge of network dynamics. On the other
hand, data-driven approaches frequently encounter the challenge of insufficient
labeled data, a predicament commonly observed in real-world scenarios. In this
paper, we introduce a novel resilience prediction framework for complex
networks, designed to tackle this issue through generative data augmentation of
network topology and dynamics. The core idea is the strategic utilization of
the inherent joint distribution present in unlabeled network data, facilitating
the learning process of the resilience predictor by illuminating the
relationship between network topology and dynamics. Experiment results on three
network datasets demonstrate that our proposed framework TDNetGen can achieve
high prediction accuracy up to 85%-95%. Furthermore, the framework still
demonstrates a pronounced augmentation capability in extreme low-data regimes,
thereby underscoring its utility and robustness in enhancing the prediction of
network resilience. We have open-sourced our code in the following link,
https://github.com/tsinghua-fib-lab/TDNetGen.

### 摘要 (中文)

预测复杂网络的韧性，即能够抵抗外部扰动或内部故障保持基本功能的能力，在理解并改善现实世界复杂系统方面起着至关重要的作用。基于非线性动力学系统的传统理论框架依赖于对网络动态知识的先验了解。另一方面，数据驱动的方法经常面临标签数据不足的问题，这是在实际场景中常见的困境。本论文提出了一种针对复杂网络的韧性预测框架，旨在通过调整网络拓扑和动力学的数据生成来解决这一问题。核心思想是利用未标记网络数据中的内在联合分布的策略利用，这有助于增强韧性的预测模型的学习过程，并揭示网络拓扑与动力学之间的关系。实验结果表明，我们的提出框架TDNetGen可以在三个网络数据集上达到高达85%-95%的预测精度。此外，该框架在极端低数据条件下仍表现出显著的数据增强能力，从而强调了其在增强网络韧性预测方面的实用性和鲁棒性。我们已经开源了我们的代码，链接如下：https://github.com/tsinghua-fib-lab/TDNetGen。

---

## Minor DPO reject penalty to increase training robustness

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09834v1)

### Abstract (English)

Learning from human preference is a paradigm used in large-scale language
model (LLM) fine-tuning step to better align pretrained LLM to human preference
for downstream task. In the past it uses reinforcement learning from human
feedback (RLHF) algorithm to optimize the LLM policy to align with these
preferences and not to draft too far from the original model. Recently, Direct
Preference Optimization (DPO) has been proposed to solve the alignment problem
with a simplified RL-free method. Using preference pairs of chosen and reject
data, DPO models the relative log probability as implicit reward function and
optimize LLM policy using a simple binary cross entropy objective directly. DPO
is quite straight forward and easy to be understood. It perform efficiently and
well in most cases. In this article, we analyze the working mechanism of
$\beta$ in DPO, disclose its syntax difference between RL algorithm and DPO,
and understand the potential shortage brought by the DPO simplification. With
these insights, we propose MinorDPO, which is better aligned to the original RL
algorithm, and increase the stability of preference optimization process.

### 摘要 (中文)

从人类偏好学习是大型语言模型（LLM）微调阶段的一种模式，以更好地与下游任务的人类偏好相匹配。在过去，它使用强化学习从人类反馈（RLHF）算法优化LLM政策来使其与这些偏好保持一致，而不是偏离原始模型太远。最近，直接偏好优化（DPO）被提出作为解决这一匹配问题的简化无监督方法。通过选择和拒绝数据之间的偏好对比如今和过去的数据，DPO模型将相对逻辑概率表示为隐含奖励函数，并使用简单的二元交叉熵目标直接优化LLM策略。DPO非常简单直观，易于理解，在大多数情况下都能高效且良好地工作。本文分析了DPO中的$\beta$在DPO中的作用、揭示了RL算法和DPO中语法的区别以及简化的DPO带来的潜在不足。凭借这些见解，我们提出了MinorDPO，这是一种更符合原始RL算法的改进版，并提高了偏好优化过程的稳定性。

---

## Demystifying Reinforcement Learning in Production Scheduling via Explainable AI

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09841v1)

### Abstract (English)

Deep Reinforcement Learning (DRL) is a frequently employed technique to solve
scheduling problems. Although DRL agents ace at delivering viable results in
short computing times, their reasoning remains opaque. We conduct a case study
where we systematically apply two explainable AI (xAI) frameworks, namely SHAP
(DeepSHAP) and Captum (Input x Gradient), to describe the reasoning behind
scheduling decisions of a specialized DRL agent in a flow production. We find
that methods in the xAI literature lack falsifiability and consistent
terminology, do not adequately consider domain-knowledge, the target audience
or real-world scenarios, and typically provide simple input-output explanations
rather than causal interpretations. To resolve this issue, we introduce a
hypotheses-based workflow. This approach enables us to inspect whether
explanations align with domain knowledge and match the reward hypotheses of the
agent. We furthermore tackle the challenge of communicating these insights to
third parties by tailoring hypotheses to the target audience, which can serve
as interpretations of the agent's behavior after verification. Our proposed
workflow emphasizes the repeated verification of explanations and may be
applicable to various DRL-based scheduling use cases.

### 摘要 (中文)

深度强化学习（Deep Reinforcement Learning，简称DRL）是一种经常被用来解决调度问题的技术。虽然DRL代理在短计算时间下能出色地交付可行的结果，但他们的推理仍然是模糊的。我们进行了一个案例研究，其中系统性地应用了两种可解释人工智能（xAI）框架，即DeepSHAP和Captum（输入x梯度），来描述专门DRL代理在流程生产中制定决策时推理背后的逻辑。我们发现文献中的xAIL缺乏可验证性和一致的语言，不充分考虑知识域、目标受众或现实世界场景，并通常提供简单的输入-输出解释，而不是因果解释。为了解决这个问题，我们引入了一种基于假设的工作流。这种方法使我们能够检查是否解释与知识域相匹配，并且是否符合代理奖励的假设。此外，我们还解决了向第三方传达这些见解的问题，通过调整假设以适应目标受众，这可以作为代理行为的后验解释。我们的提出工作流强调了对解释的反复验证，并可能适用于各种基于DRL的调度使用案例。

---

## Contextual Importance and Utility in Python_ New Functionality and Insights with the py-ciu Package

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09957v1)

### Abstract (English)

The availability of easy-to-use and reliable software implementations is
important for allowing researchers in academia and industry to test, assess and
take into use eXplainable AI (XAI) methods. This paper describes the
\texttt{py-ciu} Python implementation of the Contextual Importance and Utility
(CIU) model-agnostic, post-hoc explanation method and illustrates capabilities
of CIU that go beyond the current state-of-the-art that could be useful for XAI
practitioners in general.

### 摘要 (中文)

易于使用和可靠的软件实现对于允许学术界和工业界的研究人员测试、评估和采用可解释人工智能（XAI）方法至关重要。本文描述了\texttt{py-ciu}Python实现的无特定后处理CIU模型中立解释方法，展示了CIU超越当前最先进的能力，这可能是XAI实践者通用有用的。

---

## MSDiagnosis_ An EMR-based Dataset for Clinical Multi-Step Diagnosis

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10039v1)

### Abstract (English)

Clinical diagnosis is critical in medical practice, typically requiring a
continuous and evolving process that includes primary diagnosis, differential
diagnosis, and final diagnosis. However, most existing clinical diagnostic
tasks are single-step processes, which does not align with the complex
multi-step diagnostic procedures found in real-world clinical settings. In this
paper, we propose a multi-step diagnostic task and annotate a clinical
diagnostic dataset (MSDiagnosis). This dataset includes primary diagnosis,
differential diagnosis, and final diagnosis questions. Additionally, we propose
a novel and effective framework. This framework combines forward inference,
backward inference, reflection, and refinement, enabling the LLM to
self-evaluate and adjust its diagnostic results. To assess the effectiveness of
our proposed method, we design and conduct extensive experiments. The
experimental results demonstrate the effectiveness of the proposed method. We
also provide a comprehensive experimental analysis and suggest future research
directions for this task.

### 摘要 (中文)

临床诊断在医疗实践中至关重要，通常需要一个连续和不断发展的过程，包括初步诊断、鉴别诊断和最终诊断。然而，目前大多数临床诊断任务都是单步过程，这并不符合现实生活中的复杂多步诊断程序。在这篇论文中，我们提出一个多步骤诊断任务，并标注了一个临床诊断数据集（MSDiagnosis）。这个数据集包括初步诊断、鉴别诊断和最终诊断问题。此外，我们还提出一种新型有效的框架。这一框架结合了向前推理、向后推理、反射和精炼，使LLM能够自我评估并调整其诊断结果。为了评估我们提出的办法的有效性，我们设计并进行了广泛的实验。实验结果显示，该方法的有效性。我们也提供了全面的实验分析，并提出了此任务未来研究方向。

---

## The Practimum-Optimum Algorithm for Manufacturing Scheduling_ A Paradigm Shift Leading to Breakthroughs in Scale and Performance

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10040v1)

### Abstract (English)

The Practimum-Optimum (P-O) algorithm represents a paradigm shift in
developing automatic optimization products for complex real-life business
problems such as large-scale manufacturing scheduling. It leverages deep
business domain expertise to create a group of virtual human expert (VHE)
agents with different "schools of thought" on how to create high-quality
schedules. By computerizing them into algorithms, P-O generates many valid
schedules at far higher speeds than human schedulers are capable of. Initially,
these schedules can also be local optimum peaks far away from high-quality
schedules. By submitting these schedules to a reinforced machine learning
algorithm (RL), P-O learns the weaknesses and strengths of each VHE schedule,
and accordingly derives reward and punishment changes in the Demand Set that
will modify the relative priorities for time and resource allocation that jobs
received in the prior iteration that led to the current state of the schedule.
These cause the core logic of the VHE algorithms to explore, in the subsequent
iteration, substantially different parts of the schedules universe and
potentially find higher-quality schedules. Using the hill climbing analogy,
this may be viewed as a big jump, shifting from a given local peak to a faraway
promising start point equipped with knowledge embedded in the demand set for
future iterations. This is a fundamental difference from most contemporary
algorithms, which spend considerable time on local micro-steps restricted to
the neighbourhoods of local peaks they visit. This difference enables a
breakthrough in scale and performance for fully automatic manufacturing
scheduling in complex organizations. The P-O algorithm is at the heart of
Plataine Scheduler that, in one click, routinely schedules 30,000-50,000 tasks
for real-life complex manufacturing operations.

### 摘要 (中文)

Practicum Optimum（P-O）算法代表了在复杂现实生活商业问题中开发自动优化产品的一种范式转变，例如大规模制造调度。它利用深奥的业务领域专业知识创建一组虚拟人类专家（VHE）代理，他们有不同的“思想”来创造高质量的日程安排。通过将其计算机化成算法，P-O可以生成比人类调度员更快速的有效日程安排，速度是后者无法达到的。最初，这些日程安排也可能远离高质量日程安排的局部最优峰。通过将这些日程提交给强化机器学习算法（RL），P-O会学习每个VHE日程的弱点和优势，并相应地调整需求集中的奖励和惩罚变化，以改变前一迭代中导致当前状态的资源分配相对优先权的时间和资源分配。这导致VHE算法的核心逻辑探索，在随后的迭代中，对日程宇宙进行大幅度不同的部分搜索，可能找到更好的日程。使用爬山法的比喻，这可能被视为一个大跳跃，从给定的局部高峰跳到一个装备有未来迭代需求集知识的知识点远端充满知识的地方。这是大多数当代算法所不具备的突破性进展，它们花费大量的时间在限制于访问的局部顶峰附近的微小步骤上。这种差异使大型组织完全自动化的制造业调度达到了革命性的进展。P-O算法是Plataine Scheduler的核心，只需一键就能为真实生活中的复杂制造业操作调度3000至5000项任务。

---

## ARMADA_ Attribute-Based Multimodal Data Augmentation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10086v1)

### Abstract (English)

In Multimodal Language Models (MLMs), the cost of manually annotating
high-quality image-text pair data for fine-tuning and alignment is extremely
high. While existing multimodal data augmentation frameworks propose ways to
augment image-text pairs, they either suffer from semantic inconsistency
between texts and images, or generate unrealistic images, causing knowledge gap
with real world examples. To address these issues, we propose Attribute-based
Multimodal Data Augmentation (ARMADA), a novel multimodal data augmentation
method via knowledge-guided manipulation of visual attributes of the mentioned
entities. Specifically, we extract entities and their visual attributes from
the original text data, then search for alternative values for the visual
attributes under the guidance of knowledge bases (KBs) and large language
models (LLMs). We then utilize an image-editing model to edit the images with
the extracted attributes. ARMADA is a novel multimodal data generation
framework that: (i) extracts knowledge-grounded attributes from symbolic KBs
for semantically consistent yet distinctive image-text pair generation, (ii)
generates visually similar images of disparate categories using neighboring
entities in the KB hierarchy, and (iii) uses the commonsense knowledge of LLMs
to modulate auxiliary visual attributes such as backgrounds for more robust
representation of original entities. Our empirical results over four downstream
tasks demonstrate the efficacy of our framework to produce high-quality data
and enhance the model performance. This also highlights the need to leverage
external knowledge proxies for enhanced interpretability and real-world
grounding.

### 摘要 (中文)

在多模态语言模型（MLMs）中，手动标注高质量的图像-文本对进行微调和配准的成本非常高。虽然现有的多模态数据增强框架提出了通过指导知识库（KB）和大型语言模型（LLM）来增强图像-文本对的方法，但它们要么存在文字与图像之间的语义不一致性，要么生成不现实的图像，导致知识差距与现实生活中的例子。为了应对这些问题，我们提出一种名为Attribute-based Multimodal Data Augmentation（ARMADA）的新多模态数据增强方法，这是一种基于知识引导的视觉属性操作的知识导向式操纵的新多模态数据增强方法。具体来说，我们从原始文本数据中提取实体及其视觉属性，并根据知识库（KB）和大型语言模型（LLM）的指导搜索替代值。然后，我们利用图像编辑模型使用提取到的属性编辑图像。ARMADA是一种新颖的多模态数据生成框架，它具有以下特点：(i)从符号KB中抽取符号性的知识地化属性以产生语义一致且独特的图像-文本对；(ii)使用相邻实体在KB层次结构中的不同类别使用视觉上相似的图像；(iii)利用LLM的常识知识调节辅助视觉属性，如背景，以增强原始实体的更 robust表示能力。我们的四个下游任务的实验证明了我们的框架可以生产高质量的数据并提高模型性能的有效性。这也突出了需要利用外部知识代理来提升解释性和现实世界扎根的需求。

---

## Enhancing Reinforcement Learning Through Guided Search

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10113v1)

### Abstract (English)

With the aim of improving performance in Markov Decision Problem in an
Off-Policy setting, we suggest taking inspiration from what is done in Offline
Reinforcement Learning (RL). In Offline RL, it is a common practice during
policy learning to maintain proximity to a reference policy to mitigate
uncertainty, reduce potential policy errors, and help improve performance. We
find ourselves in a different setting, yet it raises questions about whether a
similar concept can be applied to enhance performance ie, whether it is
possible to find a guiding policy capable of contributing to performance
improvement, and how to incorporate it into our RL agent. Our attention is
particularly focused on algorithms based on Monte Carlo Tree Search (MCTS) as a
guide.MCTS renowned for its state-of-the-art capabilities across various
domains, catches our interest due to its ability to converge to equilibrium in
single-player and two-player contexts. By harnessing the power of MCTS as a
guide for our RL agent, we observed a significant performance improvement,
surpassing the outcomes achieved by utilizing each method in isolation. Our
experiments were carried out on the Atari 100k benchmark.

### 摘要 (中文)

在Markov决策问题的无监督设置中，我们建议从在线强化学习（RL）中的做法中汲取灵感。在线RL中，在政策学习期间维护与参考策略之间的接近以减轻不确定性、减少潜在的政策错误并帮助提高性能是一个常见的实践。然而，我们的设置不同，这引发了一个问题：是否可以应用类似的概念来增强性能？换句话说，是否有能力找到一个指导政策，它能够促进性能的改善，并如何将其融入到我们的RL代理中？

我们的注意力特别集中在基于蒙特卡洛树搜索（MCTS）的算法上，MCTS因其在各种领域的出色表现而闻名于世，因此它的能力吸引了我们的注意。通过利用MCTS作为我们的RL代理的指南，我们在实验中观察到了显著的表现改进，超过了使用每种方法单独使用的结果。我们的实验是在Atari 100K基准上进行的。

---

## Geometry Informed Tokenization of Molecules for Language Model Generation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10120v1)

### Abstract (English)

We consider molecule generation in 3D space using language models (LMs),
which requires discrete tokenization of 3D molecular geometries. Although
tokenization of molecular graphs exists, that for 3D geometries is largely
unexplored. Here, we attempt to bridge this gap by proposing the Geo2Seq, which
converts molecular geometries into $SE(3)$-invariant 1D discrete sequences.
Geo2Seq consists of canonical labeling and invariant spherical representation
steps, which together maintain geometric and atomic fidelity in a format
conducive to LMs. Our experiments show that, when coupled with Geo2Seq, various
LMs excel in molecular geometry generation, especially in controlled generation
tasks.

### 摘要 (中文)

我们使用语言模型（LM）在三维空间中生成分子，这需要对三维分子几何进行离散化。虽然分子图的分词存在，但对于三维形状的分词却是未被探索的领域。在这里，我们试图通过提出Geo2Seq来填补这一空白，Geo2Seq将分子几何转换成$SE(3)$不变性的1D离散序列。Geo2Seq由可归一化的标记和等价球表示步骤组成，这些步骤一起保持了几何和原子的忠实性，并且有利于LMS的格式。我们的实验表明，在与Geo2Seq结合时，各种LM在分子几何生成方面表现出色，特别是在控制生成任务中尤其明显。

---

## YOLOv1 to YOLOv10_ The fastest and most accurate real-time object detection systems

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09332v1)

### Abstract (English)

This is a comprehensive review of the YOLO series of systems. Different from
previous literature surveys, this review article re-examines the
characteristics of the YOLO series from the latest technical point of view. At
the same time, we also analyzed how the YOLO series continued to influence and
promote real-time computer vision-related research and led to the subsequent
development of computer vision and language models.We take a closer look at how
the methods proposed by the YOLO series in the past ten years have affected the
development of subsequent technologies and show the applications of YOLO in
various fields. We hope this article can play a good guiding role in subsequent
real-time computer vision development.

### 摘要 (中文)

这是YOLO系列系统的全面回顾。与以往的文献综述不同，这一综述文章从最新的技术角度重新审视了YOLO系列系统的特点。同时，我们也分析了YOLO系列如何继续影响和推动实时计算机视觉相关研究，并引领后续计算机视觉和语言模型的发展。我们更深入地探讨了YOLO系列在过去十年中提出的各种方法是如何影响后续技术的发展，以及YOLO在各种领域的应用。我们希望这篇论文能起到良好的指导作用，在未来实时计算机视觉发展过程中发挥重要作用。

---

## Elite360M_ Efficient 360 Multi-task Learning via Bi-projection Fusion and Cross-task Collaboration

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09336v1)

### Abstract (English)

360 cameras capture the entire surrounding environment with a large FoV,
exhibiting comprehensive visual information to directly infer the 3D
structures, e.g., depth and surface normal, and semantic information
simultaneously. Existing works predominantly specialize in a single task,
leaving multi-task learning of 3D geometry and semantics largely unexplored.
Achieving such an objective is, however, challenging due to: 1) inherent
spherical distortion of planar equirectangular projection (ERP) and
insufficient global perception induced by 360 image's ultra-wide FoV; 2)
non-trivial progress in effectively merging geometry and semantics among
different tasks to achieve mutual benefits. In this paper, we propose a novel
end-to-end multi-task learning framework, named Elite360M, capable of inferring
3D structures via depth and surface normal estimation, and semantics via
semantic segmentation simultaneously. Our key idea is to build a representation
with strong global perception and less distortion while exploring the inter-
and cross-task relationships between geometry and semantics. We incorporate the
distortion-free and spatially continuous icosahedron projection (ICOSAP) points
and combine them with ERP to enhance global perception. With a negligible cost,
a Bi-projection Bi-attention Fusion module is thus designed to capture the
semantic- and distance-aware dependencies between each pixel of the
region-aware ERP feature and the ICOSAP point feature set. Moreover, we propose
a novel Cross-task Collaboration module to explicitly extract task-specific
geometric and semantic information from the learned representation to achieve
preliminary predictions. It then integrates the spatial contextual information
among tasks to realize cross-task fusion. Extensive experiments demonstrate the
effectiveness and efficacy of Elite360M.

### 摘要 (中文)

通过大视场（FoV）捕捉整个周围环境，可以提供全面的视觉信息，直接推断三维结构，如深度和表面法向量，并同时提供语义信息。现有的工作主要专注于单一任务，导致三维几何和语义多任务学习几乎未被探索。然而，实现这样的目标是具有挑战性的，因为存在以下因素：

1. 平面等角投影（ERP）中存在的内在球形畸变以及超宽视野限制了全局感知；

2. 在不同的任务中有效地融合几何和语义以实现互惠互利的进展仍然非常困难。

在本文中，我们提出了一种名为Elite360M的新端到端多任务学习框架，该框架能够通过深度和表面法向量估计推断三维结构，并通过语义分割同时推断语义信息。我们的关键想法是在保持强烈全局感知的同时减少扭曲，同时探索几何和语义之间的交互和交叉任务关系。我们将无损地结合分布连续的十二面体投影（ICOSAP）点并将其与ERP相结合，从而增强全局感知。因此，设计了一个Bi-projection Bi-attention Fusion模块来捕获区域感知特征和ICOSAP点特征集之间的像素依赖性距离。此外，我们提出了一个跨任务合作模块，从学习的表示中明确提取每个任务特定的几何和语义信息，以便获得初步预测。然后，它将空间上下文信息整合到任务之间，实现跨任务融合。大量的实验表明了Elite360M的有效性和有效性。

---

## S_3D-NeRF_ Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09347v1)

### Abstract (English)

Talking head synthesis is a practical technique with wide applications.
Current Neural Radiance Field (NeRF) based approaches have shown their
superiority on driving one-shot talking heads with videos or signals regressed
from audio. However, most of them failed to take the audio as driven
information directly, unable to enjoy the flexibility and availability of
speech. Since mapping audio signals to face deformation is non-trivial, we
design a Single-Shot Speech-Driven Neural Radiance Field (S^3D-NeRF) method in
this paper to tackle the following three difficulties: learning a
representative appearance feature for each identity, modeling motion of
different face regions with audio, and keeping the temporal consistency of the
lip area. To this end, we introduce a Hierarchical Facial Appearance Encoder to
learn multi-scale representations for catching the appearance of different
speakers, and elaborate a Cross-modal Facial Deformation Field to perform
speech animation according to the relationship between the audio signal and
different face regions. Moreover, to enhance the temporal consistency of the
important lip area, we introduce a lip-sync discriminator to penalize the
out-of-sync audio-visual sequences. Extensive experiments have shown that our
S^3D-NeRF surpasses previous arts on both video fidelity and audio-lip
synchronization.

### 摘要 (中文)

合成头像是一个实用的技术，有着广泛的应用。

当前基于神经光线场（NeRF）的方法在视频或音频驱动的语音信号上取得了显著的优势。然而，大多数方法都无法直接使用音频作为驱动信息，无法享受语音的灵活性和可用性。由于将音频信号映射到面部变形是不简单的任务，我们在本文中设计了一个单序列语音驱动神经光线场（S^3D-NeRF）方法来解决以下三个困难：学习每个身份代表性的外观特征，根据音频信号的不同面部区域进行运动模型，以及保持嘴唇区域的时间一致性。为了实现这一点，我们引入了一个多尺度面部表情编码器，用于捕获不同演讲者的外观，并详细地介绍了跨模态面部变形场，以根据音频信号和不同面部区域之间的关系执行语音动画。此外，为了增强重要唇部区域的时间一致性，我们引入了一种唇同步鉴别器，对不在同一时间线上的音频-视觉序列施加惩罚。大量的实验已经证明，在视频清晰度和唇同步方面，我们的S^3D-NeRF超越了以前的艺术。

---

## Hyperstroke_ A Novel High-quality Stroke Representation for Assistive Artistic Drawing

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09348v1)

### Abstract (English)

Assistive drawing aims to facilitate the creative process by providing
intelligent guidance to artists. Existing solutions often fail to effectively
model intricate stroke details or adequately address the temporal aspects of
drawing. We introduce hyperstroke, a novel stroke representation designed to
capture precise fine stroke details, including RGB appearance and alpha-channel
opacity. Using a Vector Quantization approach, hyperstroke learns compact
tokenized representations of strokes from real-life drawing videos of artistic
drawing. With hyperstroke, we propose to model assistive drawing via a
transformer-based architecture, to enable intuitive and user-friendly drawing
applications, which are experimented in our exploratory evaluation.

### 摘要 (中文)

辅助绘图旨在通过提供智能指导来促进艺术家的创作过程。现有的解决方案往往无法有效模拟复杂笔触细节或妥善处理绘画中的时间性方面。我们引入了新型的笔触表示法，即超笔触，以捕捉精确的精细笔触细节，包括RGB显示和透明度通道。使用矢量量化方法，超笔触从现实生活艺术绘画中学习到实况视频中的紧凑标记化笔触表示。借助超笔触，我们提出利用基于Transformer的架构模型辅助绘图，以便于直观易用的绘制应用，我们在探索性的评估中进行了实验。

---

## Boundary-Recovering Network for Temporal Action Detection

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09354v1)

### Abstract (English)

Temporal action detection (TAD) is challenging, yet fundamental for
real-world video applications. Large temporal scale variation of actions is one
of the most primary difficulties in TAD. Naturally, multi-scale features have
potential in localizing actions of diverse lengths as widely used in object
detection. Nevertheless, unlike objects in images, actions have more ambiguity
in their boundaries. That is, small neighboring objects are not considered as a
large one while short adjoining actions can be misunderstood as a long one. In
the coarse-to-fine feature pyramid via pooling, these vague action boundaries
can fade out, which we call 'vanishing boundary problem'. To this end, we
propose Boundary-Recovering Network (BRN) to address the vanishing boundary
problem. BRN constructs scale-time features by introducing a new axis called
scale dimension by interpolating multi-scale features to the same temporal
length. On top of scale-time features, scale-time blocks learn to exchange
features across scale levels, which can effectively settle down the issue. Our
extensive experiments demonstrate that our model outperforms the
state-of-the-art on the two challenging benchmarks, ActivityNet-v1.3 and
THUMOS14, with remarkably reduced degree of the vanishing boundary problem.

### 摘要 (中文)

时间动作检测（Temporal Action Detection，TAD）是一个挑战性的问题，但对于现实世界视频应用来说是至关重要的。在TAD中，大规模的行动时长变化是最主要的困难之一。因此，在对象检测中广泛使用的多尺度特征可能有助于定位各种长度的动作边界。然而，与图像中的物体不同，动作的边界更具有模糊性。换句话说，小邻域的对象不被视为一个大的一个，而短相邻的动作可能会被误解为一个较长的一个。通过池化操作在粗到细的特征金字塔中，这些模糊的动作边界可以消失，我们称其为“消逝边界问题”。为了应对这个问题，我们提出了Boundary-Recovering Network（BRN），以解决消逝边界问题。BRN通过引入一个新的轴称为尺度维度来构建由新轴组成的尺度-时间特征。在尺度-时间特征之上，学习从不同的尺度水平交换特征的能力，这可以在一定程度上解决问题。我们的大量实验表明，我们的模型在两个挑战性的基准活动网和THUMOS14上超越了最先进的性能，并且显著降低了消逝边界问题的程度。

---

## Joint Temporal Pooling for Improving Skeleton-based Action Recognition

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09356v1)

### Abstract (English)

In skeleton-based human action recognition, temporal pooling is a critical
step for capturing spatiotemporal relationship of joint dynamics. Conventional
pooling methods overlook the preservation of motion information and treat each
frame equally. However, in an action sequence, only a few segments of frames
carry discriminative information related to the action. This paper presents a
novel Joint Motion Adaptive Temporal Pooling (JMAP) method for improving
skeleton-based action recognition. Two variants of JMAP, frame-wise pooling and
joint-wise pooling, are introduced. The efficacy of JMAP has been validated
through experiments on the popular NTU RGB+D 120 and PKU-MMD datasets.

### 摘要 (中文)

在基于骨骼的人体动作识别中，时间池化是捕捉关节动态空间-时序关系的关键步骤。常规的池化方法忽略了运动信息的保全，并且把每一帧都视为相同对待。然而，在一个动作序列中，只有少数片段的帧携带与动作相关的特征。本论文提出了一个新的名为Joint Motion Adaptive Temporal Pooling（JMAP）的方法来改善基于骨骼的动作识别。两种变体的JMAP，帧间池化和联合池化，被引入了进来。通过NTU RGB+D 120和PKU-MMD数据集上的实验验证了JMAP的有效性。

---

## Angle of Arrival Estimation with Transformer_ A Sparse and Gridless Method with Zero-Shot Capability

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09362v1)

### Abstract (English)

Automotive Multiple-Input Multiple-Output (MIMO) radars have gained
significant traction in Advanced Driver Assistance Systems (ADAS) and
Autonomous Vehicles (AV) due to their cost-effectiveness, resilience to
challenging operating conditions, and extended detection range. To fully
leverage the advantages of MIMO radars, it is crucial to develop an Angle of
Arrival (AOA) algorithm that delivers high performance with reasonable
computational workload. This work introduces AAETR (Angle of Arrival Estimation
with TRansformer) for high performance gridless AOA estimation. Comprehensive
evaluations across various signal-to-noise ratios (SNRs) and multi-target
scenarios demonstrate AAETR's superior performance compared to super resolution
AOA algorithms such as Iterative Adaptive Approach (IAA). The proposed
architecture features efficient, scalable, sparse and gridless angle-finding
capability, overcoming the issues of high computational cost and straddling
loss in SNR associated with grid-based IAA. AAETR requires fewer tunable
hyper-parameters and is end-to-end trainable in a deep learning radar
perception pipeline. When trained on large-scale simulated datasets then
evaluated on real dataset, AAETR exhibits remarkable zero-shot sim-to-real
transferability and emergent sidelobe suppression capability. This highlights
the effectiveness of the proposed approach and its potential as a drop-in
module in practical systems.

### 摘要 (中文)

多输入多输出（MIMO）雷达在先进的驾驶辅助系统（ADAS）和自动驾驶汽车（AV）中获得了显著的进展，由于其成本效益、对挑战性操作条件的抗干扰能力和更远的检测范围。为了充分利用MIMO雷达的优势，开发一个估计角度到达（AOA）算法至关重要，该算法具有合理的计算负载提供高性能。这项工作引入了AAETR（Angle of Arrival Estimation with Transfomer）以高精度进行网格状AOA估计。全面的评估表明，AAETR在各种信号噪声比（SNR）和多目标场景中优于超分辨率AOA算法，如迭代适应方法（IAA）。提出的架构具有高效、可扩展、稀疏且无网格的角度寻找能力，克服了与基于网格的IAA相关的高计算成本和SNR跨越问题。AAETR需要较少调参的超参数，并可在深度学习雷达感知管道的端到端训练中作为端点完成训练。当使用大规模仿真数据集进行训练并用于真实数据集进行评估时，AAETR表现出令人印象深刻的零样本重定向转移能力和突发旁瓣抑制能力。这突出了提出的方法的有效性和将其视为实际系统中的降级模块的可能性。

---

## OU-CoViT_ Copula-Enhanced Bi-Channel Multi-Task Vision Transformers with Dual Adaptation for OU-UWF Images

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09395v1)

### Abstract (English)

Myopia screening using cutting-edge ultra-widefield (UWF) fundus imaging and
joint modeling of multiple discrete and continuous clinical scores presents a
promising new paradigm for multi-task problems in Ophthalmology. The bi-channel
framework that arises from the Ophthalmic phenomenon of ``interocular
asymmetries'' of both eyes (OU) calls for new employment on the SOTA
transformer-based models. However, the application of copula models for
multiple mixed discrete-continuous labels on deep learning (DL) is challenging.
Moreover, the application of advanced large transformer-based models to small
medical datasets is challenging due to overfitting and computational resource
constraints. To resolve these challenges, we propose OU-CoViT: a novel
Copula-Enhanced Bi-Channel Multi-Task Vision Transformers with Dual Adaptation
for OU-UWF images, which can i) incorporate conditional correlation information
across multiple discrete and continuous labels within a deep learning framework
(by deriving the closed form of a novel Copula Loss); ii) take OU inputs
subject to both high correlation and interocular asymmetries using a bi-channel
model with dual adaptation; and iii) enable the adaptation of large vision
transformer (ViT) models to small medical datasets. Solid experiments
demonstrate that OU-CoViT significantly improves prediction performance
compared to single-channel baseline models with empirical loss. Furthermore,
the novel architecture of OU-CoViT allows generalizability and extensions of
our dual adaptation and Copula Loss to various ViT variants and large DL models
on small medical datasets. Our approach opens up new possibilities for joint
modeling of heterogeneous multi-channel input and mixed discrete-continuous
clinical scores in medical practices and has the potential to advance
AI-assisted clinical decision-making in various medical domains beyond
Ophthalmology.

### 摘要 (中文)

使用最先进的超宽视野（UWF）视网膜成像和多条目连续的断续临床评分联合模型，为眼科领域的多任务问题提供了令人鼓舞的新范式。眼科现象中的“双眼差异”要求我们对现有基于变换器的模型进行新的就业。然而，在深度学习（DL）中应用Copula模型处理多个混合离散-连续标签具有挑战性。此外，由于过拟合和计算资源限制，将大型变体基于模型应用于小型医疗数据集具有挑战性。为了解决这些问题，我们提出了一种名为OU-CoViT的新架构，它结合了条件相关性信息，并在深学习框架中进行了双适应性双通道多任务视觉变换器，可以实现以下目标：i) 在一个深度学习框架中，可以将多条目离散和连续标签之间的条件相关性信息纳入闭合形式的新型Copula损失；ii) 可以在高相关性和双眼差异下使用双通道模型并行训练；iii) 允许大型视觉变换器（ViT）模型在小医疗数据集上进行自适应。实验证明，与单通道基准模型相比，采用OU-CoViT显著提高了预测性能。此外，OU-CoViT的新型架构允许通用化我们的双适应性和Copula损失到各种大小和类型的ViT变体以及大型DL模型的小型医疗数据集上。我们的方法为医疗实践中的异构多通道输入和混合离散-连续临床评分的联合建模开辟了新可能性，有望在医学领域以外的各个医疗领域推进AI辅助临床决策能力的进步。

---

## Combo_ Co-speech holistic 3D human motion generation and efficient customizable adaptation in harmony

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09397v1)

### Abstract (English)

In this paper, we propose a novel framework, Combo, for harmonious co-speech
holistic 3D human motion generation and efficient customizable adaption. In
particular, we identify that one fundamental challenge as the
multiple-input-multiple-output (MIMO) nature of the generative model of
interest. More concretely, on the input end, the model typically consumes both
speech signals and character guidance (e.g., identity and emotion), which not
only poses challenge on learning capacity but also hinders further adaptation
to varying guidance; on the output end, holistic human motions mainly consist
of facial expressions and body movements, which are inherently correlated but
non-trivial to coordinate in current data-driven generation process. In
response to the above challenge, we propose tailored designs to both ends. For
the former, we propose to pre-train on data regarding a fixed identity with
neutral emotion, and defer the incorporation of customizable conditions
(identity and emotion) to fine-tuning stage, which is boosted by our novel
X-Adapter for parameter-efficient fine-tuning. For the latter, we propose a
simple yet effective transformer design, DU-Trans, which first divides into two
branches to learn individual features of face expression and body movements,
and then unites those to learn a joint bi-directional distribution and directly
predicts combined coefficients. Evaluated on BEAT2 and SHOW datasets, Combo is
highly effective in generating high-quality motions but also efficient in
transferring identity and emotion. Project website:
\href{https://xc-csc101.github.io/combo/}{Combo}.

### 摘要 (中文)

在本文中，我们提出了一种新的框架Compo，用于和谐的声波综合3D人体运动生成和高效的自定义适应。具体来说，我们发现一个基本挑战是兴趣模型的多输入输出（MIMO）性质。更具体地说，在输入端，该模型通常需要同时处理语音信号和人物引导（例如身份和情绪），这不仅提出了学习能力的问题，而且也限制了后续对不同指导的进一步适应；在输出端，主要的人体动作主要包括面部表情和身体动作，这些动作本身就存在相关性，但在当前的数据驱动生成过程中却难以协调。面对上述挑战，我们提出了两种面向两端的设计。对于前者，我们提出预训练数据集，其中包含固定身份和中立情感的身份信息，然后在微调阶段延迟集成可定制条件（身份和情绪），这是由我们的新型X-Adaptor参数高效微调设计来增强的。对于后者，我们提出一种简单而有效的Transformer设计DU-Trans，首先将其分为两个分支以学习面部表情和个人特征的身体移动的个体特性和联合分布，然后将它们结合起来学习一个双向分布，并直接预测联合系数。在BEAT2和SHOW数据集上评估后，Compo在生成高质量的动作方面非常有效，同时也非常高效地传递身份和情绪。项目网站：
https://xc-csc101.github.io/combo/

---

## VrdONE_ One-stage Video Visual Relation Detection

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09408v1)

### Abstract (English)

Video Visual Relation Detection (VidVRD) focuses on understanding how
entities interact over time and space in videos, a key step for gaining deeper
insights into video scenes beyond basic visual tasks. Traditional methods for
VidVRD, challenged by its complexity, typically split the task into two parts:
one for identifying what relation categories are present and another for
determining their temporal boundaries. This split overlooks the inherent
connection between these elements. Addressing the need to recognize entity
pairs' spatiotemporal interactions across a range of durations, we propose
VrdONE, a streamlined yet efficacious one-stage model. VrdONE combines the
features of subjects and objects, turning predicate detection into 1D instance
segmentation on their combined representations. This setup allows for both
relation category identification and binary mask generation in one go,
eliminating the need for extra steps like proposal generation or
post-processing. VrdONE facilitates the interaction of features across various
frames, adeptly capturing both short-lived and enduring relations.
Additionally, we introduce the Subject-Object Synergy (SOS) module, enhancing
how subjects and objects perceive each other before combining. VrdONE achieves
state-of-the-art performances on the VidOR benchmark and ImageNet-VidVRD,
showcasing its superior capability in discerning relations across different
temporal scales. The code is available at
\textcolor[RGB]{228,58,136}{\href{https://github.com/lucaspk512/vrdone}{https://github.com/lucaspk512/vrdone}}.

### 摘要 (中文)

视频视觉关系检测（VidVRD）专注于理解视频中实体随着时间和空间的互动，这是获取对视频场景更深入见解的关键步骤。对于传统的VidVRD方法来说，由于其复杂性，通常将其分为两个部分：一是识别存在的关系类别，二是确定它们的时间边界。这种分割忽略了这些元素之间的内在联系。针对需要在不同持续时间范围内识别实体对时空交互的需求，我们提出了VrdONE，这是一个简化而有效的单一阶段模型。VrdONE结合了主体和对象的特征，将谓词检测转换为他们的联合表示上的一维实例分割。这一设置允许同时进行关系类别的识别和二值掩模生成，在一次操作中消除额外的步骤，如提议生成或后处理。VrdONE使特征跨越各种帧之间的相互作用变得有效，巧妙地捕捉到短时性和持久性的关系。此外，我们引入了Subject-Object Synergy（SOS）模块，增强当合并之前主体和对象如何感知彼此的方式。VrdONE在VidOR基准测试集和ImageNet-VidVRD上取得了最先进的性能，展示了它在不同时间尺度下区分关系的能力上的优越能力。代码可在以下链接获取：
\(\textcolor[RGB]{228,58,136}{\href{https://github.com/lucaspk512/vrdone}{https://github.com/lucaspk512/vrdone})\)。

---

## OPPH_ A Vision-Based Operator for Measuring Body Movements for Personal Healthcare

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09409v1)

### Abstract (English)

Vision-based motion estimation methods show promise in accurately and
unobtrusively estimating human body motion for healthcare purposes. However,
these methods are not specifically designed for healthcare purposes and face
challenges in real-world applications. Human pose estimation methods often lack
the accuracy needed for detecting fine-grained, subtle body movements, while
optical flow-based methods struggle with poor lighting conditions and unseen
real-world data. These issues result in human body motion estimation errors,
particularly during critical medical situations where the body is motionless,
such as during unconsciousness. To address these challenges and improve the
accuracy of human body motion estimation for healthcare purposes, we propose
the OPPH operator designed to enhance current vision-based motion estimation
methods. This operator, which considers human body movement and noise
properties, functions as a multi-stage filter. Results tested on two real-world
and one synthetic human motion dataset demonstrate that the operator
effectively removes real-world noise, significantly enhances the detection of
motionless states, maintains the accuracy of estimating active body movements,
and maintains long-term body movement trends. This method could be beneficial
for analyzing both critical medical events and chronic medical conditions.

### 摘要 (中文)

基于视觉的运动估计方法在医疗用途中具有准确和无侵入性估计人体动作的潜力。然而，这些方法并不专门设计用于医疗目的，并且在实际应用中面临挑战。人类姿势识别方法往往缺乏检测细微、微妙身体动作所需的精确度，而基于光学流的方法则难以处理不良照明条件和未见的真实世界数据。这些问题导致了人体动作估算错误，特别是在关键医疗情况下，如意识丧失时，当身体静止不动的情况下尤其如此。为了解决这些问题并提高医疗目的下人体动作估算的准确性，我们提出了OPPH操作符，该操作符旨在增强当前基于视觉的运动估计方法。这个操作符考虑到了人体移动和噪声属性，作为一个多阶段过滤器。在两个真实世界和一个合成的人体运动数据集上进行测试的结果表明，该操作符有效地去除现实世界的噪声，显著提高了对静止状态的检测能力，保持了活动身体动作的精度，同时保持了长期的身体运动趋势。这种方法对于分析既包括关键医疗事件也包括慢性医学状况的事件都有益。

---

## Weakly Supervised Lymph Nodes Segmentation Based on Partial Instance Annotations with Pre-trained Dual-branch Network and Pseudo Label Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09411v1)

### Abstract (English)

Assessing the presence of potentially malignant lymph nodes aids in
estimating cancer progression, and identifying surrounding benign lymph nodes
can assist in determining potential metastatic pathways for cancer. For
quantitative analysis, automatic segmentation of lymph nodes is crucial.
However, due to the labor-intensive and time-consuming manual annotation
process required for a large number of lymph nodes, it is more practical to
annotate only a subset of the lymph node instances to reduce annotation costs.
In this study, we propose a pre-trained Dual-Branch network with Dynamically
Mixed Pseudo label (DBDMP) to learn from partial instance annotations for lymph
nodes segmentation. To obtain reliable pseudo labels for lymph nodes that are
not annotated, we employ a dual-decoder network to generate different outputs
that are then dynamically mixed. We integrate the original weak partial
annotations with the mixed pseudo labels to supervise the network. To further
leverage the extensive amount of unannotated voxels, we apply a self-supervised
pre-training strategy to enhance the model's feature extraction capability.
Experiments on the mediastinal Lymph Node Quantification (LNQ) dataset
demonstrate that our method, compared to directly learning from partial
instance annotations, significantly improves the Dice Similarity Coefficient
(DSC) from 11.04% to 54.10% and reduces the Average Symmetric Surface Distance
(ASSD) from 20.83 $mm$ to 8.72 $mm$. The code is available at
https://github.com/WltyBY/LNQ2023_training_code.git

### 摘要 (中文)

评估可能恶性的淋巴结有助于估计癌症的进展，并识别周围的良性淋巴结可以帮助确定癌症潜在转移途径。对于定量分析，自动分割淋巴结是至关重要的。然而，由于对大量淋巴结进行手动标注所需的劳动密集型和耗时的过程，更实用的是只标记部分淋巴结实例以减少标注成本。在本研究中，我们提出了一种预训练的双分支网络，通过动态混合伪标签（DBDMP）学习从部分实例注释来学习淋巴结的分割。为了获得不注释淋巴结的可靠伪标签，我们使用双解码网络生成不同的输出，然后动态混合。我们将原始弱部分注释与混合伪标签整合到监督网络中。为了进一步利用大量的未标注像素量，我们应用自监督预训练策略来增强模型的特征提取能力。在中位肺淋巴结量化（LNQ）数据集上的实验表明，与直接从部分实例注释学习相比，我们的方法显著提高了Dice相似系数（DSC），从11.04％提高到54.10％；降低了平均对称表面距离（ASSD），从20.83毫米降低到8.72毫米。代码可在 https://github.com/WltyBY/LNQ2023_training_code.git 上获取。

---

## OVOSE_ Open-Vocabulary Semantic Segmentation in Event-Based Cameras

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09424v1)

### Abstract (English)

Event cameras, known for low-latency operation and superior performance in
challenging lighting conditions, are suitable for sensitive computer vision
tasks such as semantic segmentation in autonomous driving. However, challenges
arise due to limited event-based data and the absence of large-scale
segmentation benchmarks. Current works are confined to closed-set semantic
segmentation, limiting their adaptability to other applications. In this paper,
we introduce OVOSE, the first Open-Vocabulary Semantic Segmentation algorithm
for Event cameras. OVOSE leverages synthetic event data and knowledge
distillation from a pre-trained image-based foundation model to an event-based
counterpart, effectively preserving spatial context and transferring
open-vocabulary semantic segmentation capabilities. We evaluate the performance
of OVOSE on two driving semantic segmentation datasets DDD17, and
DSEC-Semantic, comparing it with existing conventional image open-vocabulary
models adapted for event-based data. Similarly, we compare OVOSE with
state-of-the-art methods designed for closed-set settings in unsupervised
domain adaptation for event-based semantic segmentation. OVOSE demonstrates
superior performance, showcasing its potential for real-world applications. The
code is available at https://github.com/ram95d/OVOSE.

### 摘要 (中文)

事件摄像头因其低延迟操作和挑战性光照条件下的出色性能而闻名，因此非常适合敏感的计算机视觉任务，例如自动驾驶中的语义分割。然而，由于有限的事件数据和缺乏大规模分割基准的问题，出现了挑战。当前的工作局限于封闭集语义分割，限制了它们适应其他应用的能力。在本文中，我们引入了OVOSE，这是第一个面向事件摄像头的开放式词汇语义分割算法。OVOSE利用合成的事件数据和基于图像的基础模型的知识蒸馏到事件基版，有效地保留了空间上下文，并转移了开放词汇语义分割能力。我们评估了OVOSE在两个驾驶语义分割数据集DDD17和DSEC-Semantic上的表现，与现有用于事件数据的传统图像开放词汇模型进行了比较。同样，我们与设计用于闭合集设置的无监督域适应事件基版语义分割方法进行了比较。OVOSE表现出显著的优势，展示了其在实际世界应用中的潜力。代码可在https://github.com/ram95d/OVOSE获取。

---

## Adversarial Attacked Teacher for Unsupervised Domain Adaptive Object Detection

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09431v1)

### Abstract (English)

Object detectors encounter challenges in handling domain shifts. Cutting-edge
domain adaptive object detection methods use the teacher-student framework and
domain adversarial learning to generate domain-invariant pseudo-labels for
self-training. However, the pseudo-labels generated by the teacher model tend
to be biased towards the majority class and often mistakenly include
overconfident false positives and underconfident false negatives. We reveal
that pseudo-labels vulnerable to adversarial attacks are more likely to be
low-quality. To address this, we propose a simple yet effective framework named
Adversarial Attacked Teacher (AAT) to improve the quality of pseudo-labels.
Specifically, we apply adversarial attacks to the teacher model, prompting it
to generate adversarial pseudo-labels to correct bias, suppress overconfidence,
and encourage underconfident proposals. An adaptive pseudo-label regularization
is introduced to emphasize the influence of pseudo-labels with high certainty
and reduce the negative impacts of uncertain predictions. Moreover, robust
minority objects verified by pseudo-label regularization are oversampled to
minimize dataset imbalance without introducing false positives. Extensive
experiments conducted on various datasets demonstrate that AAT achieves
superior performance, reaching 52.6 mAP on Clipart1k, surpassing the previous
state-of-the-art by 6.7%.

### 摘要 (中文)

在处理领域变换时，对象检测器面临挑战。最先进的领域适应对象检测方法使用教师-学生框架和域对抗学习生成自训练的领域无关伪标签。然而，由教师模型生成的伪标签倾向于偏向多数类，并且常常包括过自信的假正例和不足自信的假负例。我们揭示了易受攻击的伪标签更有可能是低质量的。为了应对这一问题，我们提出了一种名为Adversarial Attacked Teacher（AAT）的简单而有效的框架，以提高伪标签的质量。具体来说，我们将攻击性攻击应用于教师模型，促使它生成攻击性伪标签来纠正偏见、抑制过度自信并鼓励不足自信的提议。引入可调节伪标签规范化，强调高确定性的伪标签的影响，并减少不确定预测的负面影响。此外，通过伪标签规范验证的稳健少数对象被超额采样，以最小化数据失衡而不引入虚假阳性。在各种数据集上进行的广泛实验表明，AAT实现了优异的表现，在Clipart1k上达到52.6 mAP，超过以前的最佳状态3.7%。

---

## CLIP-CID_ Efficient CLIP Distillation via Cluster-Instance Discrimination

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09441v1)

### Abstract (English)

Contrastive Language-Image Pre-training (CLIP) has achieved excellent
performance over a wide range of tasks. However, the effectiveness of CLIP
heavily relies on a substantial corpus of pre-training data, resulting in
notable consumption of computational resources. Although knowledge distillation
has been widely applied in single modality models, how to efficiently expand
knowledge distillation to vision-language foundation models with extensive data
remains relatively unexplored. In this paper, we introduce CLIP-CID, a novel
distillation mechanism that effectively transfers knowledge from a large
vision-language foundation model to a smaller model. We initially propose a
simple but efficient image semantic balance method to reduce transfer learning
bias and improve distillation efficiency. This method filters out 43.7% of
image-text pairs from the LAION400M while maintaining superior performance.
After that, we leverage cluster-instance discrimination to facilitate knowledge
transfer from the teacher model to the student model, thereby empowering the
student model to acquire a holistic semantic comprehension of the pre-training
data. Experimental results demonstrate that CLIP-CID achieves state-of-the-art
performance on various downstream tasks including linear probe and zero-shot
classification.

### 摘要 (中文)

对比语言图像预训练（CLIP）在广泛的任务上取得了优异的表现。然而，CLIP的高效性很大程度上依赖于大量预训练数据的消耗，这导致了显著的计算资源消耗。尽管知识蒸馏已经在单模态模型中广泛应用于，如何有效地扩展知识蒸馏到具有大量数据的视觉语言基础模型仍然是相对未探索的领域。本文提出了一种名为CLIP-CID的新蒸馏机制，该机制有效地从大型视觉语言基础模型转移到较小的模型。我们首先提出了一个简单但高效的图像语义平衡方法来减少转移学习偏见并提高蒸馏效率。这种方法过滤掉了LAION400M中的43.7％的图片和文本对，同时保持了出色的性能。然后，我们利用集群实例鉴别力来促进教师模型与学生模型之间的知识传递，从而赋予学生模型对其预训练数据的整体语义理解能力。实验结果表明，CLIP-CID在各种下游任务，如线性探测和零shot分类等方面实现了最先进的表现。

---

## G2Face_ High-Fidelity Reversible Face Anonymization via Generative and Geometric Priors

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09458v1)

### Abstract (English)

Reversible face anonymization, unlike traditional face pixelization, seeks to
replace sensitive identity information in facial images with synthesized
alternatives, preserving privacy without sacrificing image clarity. Traditional
methods, such as encoder-decoder networks, often result in significant loss of
facial details due to their limited learning capacity. Additionally, relying on
latent manipulation in pre-trained GANs can lead to changes in ID-irrelevant
attributes, adversely affecting data utility due to GAN inversion inaccuracies.
This paper introduces G\textsuperscript{2}Face, which leverages both generative
and geometric priors to enhance identity manipulation, achieving high-quality
reversible face anonymization without compromising data utility. We utilize a
3D face model to extract geometric information from the input face, integrating
it with a pre-trained GAN-based decoder. This synergy of generative and
geometric priors allows the decoder to produce realistic anonymized faces with
consistent geometry. Moreover, multi-scale facial features are extracted from
the original face and combined with the decoder using our novel identity-aware
feature fusion blocks (IFF). This integration enables precise blending of the
generated facial patterns with the original ID-irrelevant features, resulting
in accurate identity manipulation. Extensive experiments demonstrate that our
method outperforms existing state-of-the-art techniques in face anonymization
and recovery, while preserving high data utility. Code is available at
https://github.com/Harxis/G2Face.

### 摘要 (中文)

可逆面部匿名化，不同于传统的面部像素化，旨在通过合成替代品来替换面部图像中的敏感身份信息，并在不牺牲图像清晰度的情况下保护隐私。传统方法，如编码器解码网络，由于其有限的学习能力，往往导致面部细节损失显著。此外，依赖于预训练GAN的潜伏操作可能会影响ID无关属性，这会严重影响数据使用性，因为GAN反向传播错误。本论文介绍G\textsuperscript{2}Face，它利用生成和几何先验增强身份操纵，实现高质 可逆面部匿名化的同时保持数据使用的准确性。我们利用三维人脸模型从输入面部中提取几何信息，并将其与预先训练的GAN基底解码器相结合。这种生成和几何先验的协同作用允许解码器产生具有一致形状的真实匿名面孔。此外，从原始面部中提取多尺度面部特征，并结合解码器使用我们的新型身份感知特征融合块（IFF），这一集成使生成的面部模式与ID无关的特征精确地混合在一起，从而实现了准确的身份操纵。广泛的实验表明，在面部匿名化和恢复方面，我们的方法优于现有最先进的技术，同时保留了高度的数据实用性。代码可在 https://github.com/Harxis/G2Face 下获取。

---

## Fine-Grained Building Function Recognition from Street-View Images via Geometry-Aware Semi-Supervised Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09460v1)

### Abstract (English)

In this work, we propose a geometry-aware semi-supervised method for
fine-grained building function recognition. This method leverages the geometric
relationships between multi-source data to improve the accuracy of pseudo
labels in semi-supervised learning, extending the task's scope and making it
applicable to cross-categorization systems of building function recognition.
Firstly, we design an online semi-supervised pre-training stage, which
facilitates the precise acquisition of building facade location information in
street-view images. In the second stage, we propose a geometry-aware coarse
annotation generation module. This module effectively combines GIS data and
street-view data based on the geometric relationships, improving the accuracy
of pseudo annotations. In the third stage, we combine the newly generated
coarse annotations with the existing labeled dataset to achieve fine-grained
functional recognition of buildings across multiple cities at a large scale.
Extensive experiments demonstrate that our proposed framework exhibits superior
performance in fine-grained functional recognition of buildings. Within the
same categorization system, it achieves improvements of 7.6% and 4.8% compared
to fully-supervised methods and state-of-the-art semi-supervised methods,
respectively. Additionally, our method also performs well in cross-city tasks,
i.e., extending the model trained on OmniCity (New York) to new areas (i.e.,
Los Angeles and Boston). This study provides a novel solution for the
fine-grained function recognition of large-scale buildings across multiple
cities, offering essential data for understanding urban infrastructure
planning, human activity patterns, and the interactions between humans and
buildings.

### 摘要 (中文)

在这一工作中，我们提出了一种基于几何感知的半监督方法来识别精细的建筑功能。这种方法利用多源数据之间的几何关系来提高伪标签在半监督学习中的准确性，扩展了任务的范围，并使其适用于建筑物功能识别交叉分类系统的应用。首先，我们设计了一个在线半监督预训练阶段，该阶段有助于精确获取街道视图图像中建筑物立面位置信息的精确获取。第二阶段，我们提出了一个基于几何关系的粗注释生成模块。这个模块有效地结合GIS数据和街道视图数据，提高了伪注释的准确率。第三阶段，我们将新生成的粗注释与现有标注集相结合，以实现大规模城市范围内对多个城市的建筑物进行细粒度功能识别。广泛的实验表明，我们的提出的框架在大规模建筑物的精细功能识别中表现出显著性能优势。在同一类别系统中，它比完全监督方法和当前最先进的半监督方法分别提高了7.6％和4.8％。此外，我们的方法也表现良好，在跨城任务方面，即模型在OmniCity（纽约）上训练后，可以延伸到新的地区（洛杉矶和波士顿）。这项研究提供了一种大型城市范围内多个城市的建筑物精细功能识别的新解决方案，对于理解和城市基础设施规划、人类活动模式以及人与建筑之间的相互作用提供了关键的数据。

---

## 3C_ Confidence-Guided Clustering and Contrastive Learning for Unsupervised Person Re-Identification

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09464v1)

### Abstract (English)

Unsupervised person re-identification (Re-ID) aims to learn a feature network
with cross-camera retrieval capability in unlabelled datasets. Although the
pseudo-label based methods have achieved great progress in Re-ID, their
performance in the complex scenario still needs to sharpen up. In order to
reduce potential misguidance, including feature bias, noise pseudo-labels and
invalid hard samples, accumulated during the learning process, in this pa per,
a confidence-guided clustering and contrastive learning (3C) framework is
proposed for unsupervised person Re-ID. This 3C framework presents three
confidence degrees. i) In the clustering stage, the confidence of the
discrepancy between samples and clusters is proposed to implement a harmonic
discrepancy clustering algorithm (HDC). ii) In the forward-propagation training
stage, the confidence of the camera diversity of a cluster is evaluated via a
novel camera information entropy (CIE). Then, the clusters with high CIE values
will play leading roles in training the model. iii) In the back-propagation
training stage, the confidence of the hard sample in each cluster is designed
and further used in a confidence integrated harmonic discrepancy (CHD), to
select the informative sample for updating the memory in contrastive learning.
Extensive experiments on three popular Re-ID benchmarks demonstrate the
superiority of the proposed framework. Particularly, the 3C framework achieves
state-of-the-art results: 86.7%/94.7%, 45.3%/73.1% and 47.1%/90.6% in terms of
mAP/Rank-1 accuracy on Market-1501, the com plex datasets MSMT17 and VeRi-776,
respectively. Code is available at https://github.com/stone5265/3C-reid.

### 摘要 (中文)

无监督人像重识别（Re-ID）旨在学习在未标记数据集中的跨摄像头检索能力的特征网络。尽管基于伪标签的方法在Re-ID方面取得了很大的进步，但在复杂场景中仍然需要进一步提升性能。为了减少潜在误导因素，包括特征偏见、噪声伪标签和无效硬样本等，在本文中提出了一种信任引导聚类与对抗学习（3C）框架来解决无监督人像重识别问题。该3C框架提出了三个信心度。 i) 在聚类阶段，提出了一种和谐差异聚类算法（HDC），用于实现散点和簇之间的差异性；ii) 在前向训练阶段，通过一种新颖的相机信息熵（CIE）评估集群内的摄像机多样性，并根据高CIE值的集群选择模型的主要角色；iii) 在后向训练阶段，设计了每个集群中硬样本的信心，并将其用于一个自信集成的和谐差异（CHD），以选择更新对比学习时内存的信息样本。在市场上1501、MSMT17和Veri-776三种流行的人像重识别基准测试集上进行的广泛实验展示了提出的框架的优越性。特别地，3C框架在Market-1501、MSMT17和Veri-776这三个复杂数据集上的mAP/Rank-1精度达到86.7%/94.7%，45.3%/73.1%和47.1%/90.6%的结果。代码可在https://github.com/stone5265/3C-reid中获取。

---

## Source-Free Test-Time Adaptation For Online Surface-Defect Detection

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09494v1)

### Abstract (English)

Surface defect detection is significant in industrial production. However,
detecting defects with varying textures and anomaly classes during the test
time is challenging. This arises due to the differences in data distributions
between source and target domains. Collecting and annotating new data from the
target domain and retraining the model is time-consuming and costly. In this
paper, we propose a novel test-time adaptation surface-defect detection
approach that adapts pre-trained models to new domains and classes during
inference. Our approach involves two core ideas. Firstly, we introduce a
supervisor to filter samples and select only those with high confidence to
update the model. This ensures that the model is not excessively biased by
incorrect data. Secondly, we propose the augmented mean prediction to generate
robust pseudo labels and a dynamically-balancing loss to facilitate the model
in effectively integrating classification and segmentation results to improve
surface-defect detection accuracy. Our approach is real-time and does not
require additional offline retraining. Experiments demonstrate it outperforms
state-of-the-art techniques.

### 摘要 (中文)

表面缺陷检测在工业生产中至关重要。然而，在测试时间内，随着测试数据分布的差异，检测不同纹理和异常类别的缺陷是具有挑战性的。这主要是由于源域与目标域之间数据分布的不同造成的。从目标域收集并标注新数据，并重新训练模型需要时间和成本。本论文提出了一种新的测试时间适应性表面缺陷检测方法，该方法在推理时根据预训练模型对新领域和新类别进行调整。我们的方法包含两个核心思想。首先，我们引入了一个监督者来过滤样本，选择那些有高置信度的样本更新模型。这样可以确保模型不会因为错误的数据而过早地偏向。其次，我们提出了增强平均预测来生成稳健伪标签以及动态平衡损失以促进模型有效地整合分类和分割结果以提高表面缺陷检测精度。我们的方法实时且不需要额外的离线重训练。实验表明，它比最先进的技术更优。

---

## StyleBrush_ Style Extraction and Transfer from a Single Image

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09496v1)

### Abstract (English)

Stylization for visual content aims to add specific style patterns at the
pixel level while preserving the original structural features. Compared with
using predefined styles, stylization guided by reference style images is more
challenging, where the main difficulty is to effectively separate style from
structural elements. In this paper, we propose StyleBrush, a method that
accurately captures styles from a reference image and ``brushes'' the extracted
style onto other input visual content. Specifically, our architecture consists
of two branches: ReferenceNet, which extracts style from the reference image,
and Structure Guider, which extracts structural features from the input image,
thus enabling image-guided stylization. We utilize LLM and T2I models to create
a dataset comprising 100K high-quality style images, encompassing a diverse
range of styles and contents with high aesthetic score. To construct training
pairs, we crop different regions of the same training image. Experiments show
that our approach achieves state-of-the-art results through both qualitative
and quantitative analyses. We will release our code and dataset upon acceptance
of the paper.

### 摘要 (中文)

视觉内容的样式化是为了在像素级别添加特定的样式模式，同时保留原始结构特征。与使用预定义风格相比，由参考图像引导的样式化更具挑战性，主要困难是有效地分离样式和结构元素。本文提出了一种名为StyleBrush的方法，该方法准确地从参考图像中捕获样式，并将其应用于输入的其他输入视觉内容上。具体而言，我们的架构包括两个分支：ReferenceNet，它从参考图像中提取样式；Structure Guider，则从输入图像中提取结构特征，从而允许基于图像的样式化。我们利用LLM和T2I模型创建了一个包含10万高质量样例图片的数据集，涵盖了多样化的风格和内容，并具有很高的审美评分。为了构建训练对，我们将相同训练图像的不同区域进行裁剪。实验表明，通过质性和定量分析，我们的方法在量化评估方面取得了最先进的结果。我们将根据论文接受后的发表，在代码和数据集上发布。

---

## NAVERO_ Unlocking Fine-Grained Semantics for Video-Language Compositionality

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09511v1)

### Abstract (English)

We study the capability of Video-Language (VidL) models in understanding
compositions between objects, attributes, actions and their relations.
Composition understanding becomes particularly challenging for video data since
the compositional relations rapidly change over time in videos. We first build
a benchmark named AARO to evaluate composition understanding related to actions
on top of spatial concepts. The benchmark is constructed by generating negative
texts with incorrect action descriptions for a given video and the model is
expected to pair a positive text with its corresponding video. Furthermore, we
propose a training method called NAVERO which utilizes video-text data
augmented with negative texts to enhance composition understanding. We also
develop a negative-augmented visual-language matching loss which is used
explicitly to benefit from the generated negative text. We compare NAVERO with
other state-of-the-art methods in terms of compositional understanding as well
as video-text retrieval performance. NAVERO achieves significant improvement
over other methods for both video-language and image-language composition
understanding, while maintaining strong performance on traditional text-video
retrieval tasks.

### 摘要 (中文)

我们研究了视频语言（VidL）模型在理解对象、属性和动作及其关系之间的能力。由于视频中的构成关系随着时间的推移迅速变化，因此对于视频数据来说，构成的理解变得特别具有挑战性。首先，我们建立了一个名为AARO的基准来评估与动作相关的构成理解。该基准是通过生成给定视频中错误的动作描述的否定文本来构建的，而模型则期望以正面文本与其对应的视频进行配对。此外，我们提出了一个称为NAVERO的方法，它利用带有否定文本的视频-文字数据增强来提升构成理解。我们还开发了一种负增强视觉语言匹配损失，这是明确用于受益于生成的否定文本的。我们将NAVERO与其他最先进的方法在构成理解和视频-文字检索性能方面进行了比较。在视频-语言和图像-语言的构成理解上，NAVERO均取得了显著的改进，同时保持了传统文本-视频检索任务的强大表现。

---

## AnomalyFactory_ Regard Anomaly Generation as Unsupervised Anomaly Localization

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09533v1)

### Abstract (English)

Recent advances in anomaly generation approaches alleviate the effect of data
insufficiency on task of anomaly localization. While effective, most of them
learn multiple large generative models on different datasets and cumbersome
anomaly prediction models for different classes. To address the limitations, we
propose a novel scalable framework, named AnomalyFactory, that unifies
unsupervised anomaly generation and localization with same network
architecture. It starts with a BootGenerator that combines structure of a
target edge map and appearance of a reference color image with the guidance of
a learned heatmap. Then, it proceeds with a FlareGenerator that receives
supervision signals from the BootGenerator and reforms the heatmap to indicate
anomaly locations in the generated image. Finally, it easily transforms the
same network architecture to a BlazeDetector that localizes anomaly pixels with
the learned heatmap by converting the anomaly images generated by the
FlareGenerator to normal images. By manipulating the target edge maps and
combining them with various reference images, AnomalyFactory generates
authentic and diversity samples cross domains. Comprehensive experiments
carried on 5 datasets, including MVTecAD, VisA, MVTecLOCO, MADSim and RealIAD,
demonstrate that our approach is superior to competitors in generation
capability and scalability.

### 摘要 (中文)

最近，异常生成方法的进步缓解了数据不足对异常定位任务的影响。虽然有效，但它们学习不同数据集上多个大型生成模型和不同的类的冗长异常预测模型。为了应对这些限制，我们提出了一种新的可扩展框架，名为AnomalyFactory，它将监督无结构异常生成和定位与相同网络架构统一起来。它从BootGenerator开始，该生成器结合了目标边框图的结构和参考颜色图像的外观，并在学习热图的指导下进行引导。然后，它进行了FlareGenerator，该生成器接收来自BootGenerator的监督信号并重新形成热图来指示生成图像中的异常位置。最后，它很容易将同一网络架构转换成BlazeDetector，通过将FlareGenerator生成的异常图像转换为正常图像来定位异常像素。通过操纵目标边缘地图并将它们与各种参考图像组合，AnomalyFactory可以在跨域中产生真实且多样性的样本。我们在包括MVTecAD、VisA、MVTecLOCO、MADSim和RealIAD在内的5个数据集上进行了全面实验，展示了我们的方法在生成能力方面的优势和扩展性。

---

## Generating Automatically Print_Scan Textures for Morphing Attack Detection Applications

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09558v1)

### Abstract (English)

Morphing Attack Detection (MAD) is a relevant topic that aims to detect
attempts by unauthorised individuals to access a "valid" identity. One of the
main scenarios is printing morphed images and submitting the respective print
in a passport application process. Today, small datasets are available to train
the MAD algorithm because of privacy concerns and the limitations resulting
from the effort associated with the printing and scanning of images at large
numbers. In order to improve the detection capabilities and spot such morphing
attacks, it will be necessary to have a larger and more realistic dataset
representing the passport application scenario with the diversity of devices
and the resulting printed scanned or compressed images. Creating training data
representing the diversity of attacks is a very demanding task because the
training material is developed manually. This paper proposes two different
methods based on transfer-transfer for automatically creating digital
print/scan face images and using such images in the training of a Morphing
Attack Detection algorithm. Our proposed method can reach an Equal Error Rate
(EER) of 3.84% and 1.92% on the FRGC/FERET database when including our
synthetic and texture-transfer print/scan with 600 dpi to handcrafted images,
respectively.

### 摘要 (中文)

变形攻击检测（MAD）是一个相关主题，旨在检测未经授权的个人尝试访问“有效”身份的企图。主要场景之一是打印并提交相应的印制护照申请流程中伪造的照片。今天，由于隐私考虑和大数量扫描和印刷图像时所需的努力限制，小数据集可用以训练MAD算法。为了提高检测能力，并发现此类变形攻击，有必要拥有一个更大、更现实的数据集代表护照申请流程中的多样性设备以及由此产生的打印、扫描或压缩图像。创建代表各种攻击类型的培训数据是一项非常艰巨的任务，因为这些材料是手动开发的。本论文提出了两种不同的基于转移-转移的方法，用于自动创建数字打印/扫描人脸图像，并使用这些图像训练变形攻击检测算法。我们的提出方法可以达到等误率误差（EER）分别为3.84%和1.92%在FRGC/FERET数据库中包括我们合成和纹理转移的打印/扫描与手工艺品图像时。

---

## Enhancing ASL Recognition with GCNs and Successive Residual Connections

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09567v1)

### Abstract (English)

This study presents a novel approach for enhancing American Sign Language
(ASL) recognition using Graph Convolutional Networks (GCNs) integrated with
successive residual connections. The method leverages the MediaPipe framework
to extract key landmarks from each hand gesture, which are then used to
construct graph representations. A robust preprocessing pipeline, including
translational and scale normalization techniques, ensures consistency across
the dataset. The constructed graphs are fed into a GCN-based neural
architecture with residual connections to improve network stability. The
architecture achieves state-of-the-art results, demonstrating superior
generalization capabilities with a validation accuracy of 99.14%.

### 摘要 (中文)

这项研究提出了一种新的方法，使用集成成功余连接的图卷积网络（Graph Convolutional Networks，GCNs）增强美国手语（ASL）识别。该方法利用MediaPipe框架从每个手势的手部动作中提取关键标记，然后用于构建图表示。一个强大的预处理管道，包括移动和缩放规范化技术，确保了数据集中的一致性。构造的图通过保留连接的GCN神经架构输入到网络以提高网络稳定性。架构达到最先进的结果，展示了在验证准确率99.14%时的优越通用能力。

---

## The First Competition on Resource-Limited Infrared Small Target Detection Challenge_ Methods and Results

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09615v1)

### Abstract (English)

In this paper, we briefly summarize the first competition on resource-limited
infrared small target detection (namely, LimitIRSTD). This competition has two
tracks, including weakly-supervised infrared small target detection (Track 1)
and lightweight infrared small target detection (Track 2). 46 and 60 teams
successfully registered and took part in Tracks 1 and Track 2, respectively.
The top-performing methods and their results in each track are described with
details. This competition inspires the community to explore the tough problems
in the application of infrared small target detection, and ultimately promote
the deployment of this technology under limited resource.

### 摘要 (中文)

在这篇文章中，我们简要总结了第一个红外小目标检测（限资源红外小目标检测）的首次竞争。这个比赛有两条轨道，包括弱监督红外小目标检测（轨道1）和轻量级红外小目标检测（轨道2）。分别有46个和60个团队成功注册并参与了轨道1和轨道2。在每条轨道上，描述了表现最佳的方法及其结果。这个比赛激发了社区探索红外小目标检测应用中的严峻问题，并最终推动这一技术在有限资源下部署。

---

## C2P-CLIP_ Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09647v1)

### Abstract (English)

This work focuses on AIGC detection to develop universal detectors capable of
identifying various types of forgery images. Recent studies have found large
pre-trained models, such as CLIP, are effective for generalizable deepfake
detection along with linear classifiers. However, two critical issues remain
unresolved: 1) understanding why CLIP features are effective on deepfake
detection through a linear classifier; and 2) exploring the detection potential
of CLIP. In this study, we delve into the underlying mechanisms of CLIP's
detection capabilities by decoding its detection features into text and
performing word frequency analysis. Our finding indicates that CLIP detects
deepfakes by recognizing similar concepts (Fig. \ref{fig:fig1} a). Building on
this insight, we introduce Category Common Prompt CLIP, called C2P-CLIP, which
integrates the category common prompt into the text encoder to inject
category-related concepts into the image encoder, thereby enhancing detection
performance (Fig. \ref{fig:fig1} b). Our method achieves a 12.41\% improvement
in detection accuracy compared to the original CLIP, without introducing
additional parameters during testing. Comprehensive experiments conducted on
two widely-used datasets, encompassing 20 generation models, validate the
efficacy of the proposed method, demonstrating state-of-the-art performance.
The code is available at
\url{https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection}

### 摘要 (中文)

该工作专注于开发能够识别各种伪造图像的通用检测器。最近的研究发现，如CLIP这样的大预训练模型在深度伪造检测中具有良好的通用性，并且线性分类器也有效。然而，仍然存在两个关键问题未得到解决：1）通过线性分类器理解为什么CLIP特征在深度伪造检测中有用；2）探索CLIP的检测潜力。在此研究中，我们深入分析了CLIP检测能力的底层机制，将其检测特征编码成文本，并进行词频分析。我们的发现表明，CLIP通过对相似概念的识别来检测假货（图 \(\ref{fig:fig1}\)a）。基于这一洞察，我们引入Category Common Prompt CLIP，称为C2P-CLIP，它整合了共同提示语句到文本编码器中，在图像编码器中注入与类别相关的概念，从而增强了检测性能（图 \(\ref{fig:fig1}\)b）。我们的方法在测试期间没有引入额外参数的情况下，在原始CLIP的基础上提高了检测准确率12.41％。对两种广泛使用的数据集进行了综合实验，包括20个生成模型，验证了提出的策略的有效性，展示了最先进的表现。代码可在以下链接下载：
\(\href{https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection}{https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection}\)

---

## CHASE_ 3D-Consistent Human Avatars with Sparse Inputs via Gaussian Splatting and Contrastive Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09663v1)

### Abstract (English)

Recent advancements in human avatar synthesis have utilized radiance fields
to reconstruct photo-realistic animatable human avatars. However, both
NeRFs-based and 3DGS-based methods struggle with maintaining 3D consistency and
exhibit suboptimal detail reconstruction, especially with sparse inputs. To
address this challenge, we propose CHASE, which introduces supervision from
intrinsic 3D consistency across poses and 3D geometry contrastive learning,
achieving performance comparable with sparse inputs to that with full inputs.
Following previous work, we first integrate a skeleton-driven rigid deformation
and a non-rigid cloth dynamics deformation to coordinate the movements of
individual Gaussians during animation, reconstructing basic avatar with coarse
3D consistency. To improve 3D consistency under sparse inputs, we design
Dynamic Avatar Adjustment(DAA) to adjust deformed Gaussians based on a selected
similar pose/image from the dataset. Minimizing the difference between the
image rendered by adjusted Gaussians and the image with the similar pose serves
as an additional form of supervision for avatar. Furthermore, we propose a 3D
geometry contrastive learning strategy to maintain the 3D global consistency of
generated avatars. Though CHASE is designed for sparse inputs, it surprisingly
outperforms current SOTA methods \textbf{in both full and sparse settings} on
the ZJU-MoCap and H36M datasets, demonstrating that our CHASE successfully
maintains avatar's 3D consistency, hence improving rendering quality.

### 摘要 (中文)

最近在人类头像合成方面取得的进步，利用辐射场来重建照片真实感的可动人偶。然而，基于NERF和3DGS的方法都面临保持三维一致性的问题，并且表现出细节重建的不佳效果，尤其是在稀疏输入时尤其如此。针对这一挑战，我们提出CHASE，引入了从每个姿态之间的内在三维一致性监督和三维几何对称学习，性能与全输入相比相当。遵循前人的工作，首先整合骨架驱动的刚性变形和非线性的布料动力学变形到单个高斯运动中的移动协调中，使用粗糙的三维一致性重建基本的人形。为了改善稀疏输入下的三维一致性，我们设计动态人形调整（DAA）来根据选定的相似姿势/图像调整受变形的高斯分布。最小化调整后的高斯分布与具有相同姿势/图像的图像之间差异作为额外的形式监督对于虚拟人形。此外，我们提出了一个三维几何对称学习策略以维持生成的虚拟人形的三维全局一致性。虽然CHASE是为稀疏输入设计的，但它在ZJU-MoCap和H36M数据集上意外地在全输入和稀疏输入下优于当前SOTA方法，在这些任务上取得了更好的表现。

---

## SG-GS_ Photo-realistic Animatable Human Avatars with Semantically-Guided Gaussian Splatting

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09665v1)

### Abstract (English)

Reconstructing photo-realistic animatable human avatars from monocular videos
remains challenging in computer vision and graphics. Recently, methods using 3D
Gaussians to represent the human body have emerged, offering faster
optimization and real-time rendering. However, due to ignoring the crucial role
of human body semantic information which represents the intrinsic structure and
connections within the human body, they fail to achieve fine-detail
reconstruction of dynamic human avatars. To address this issue, we propose
SG-GS, which uses semantics-embedded 3D Gaussians, skeleton-driven rigid
deformation, and non-rigid cloth dynamics deformation to create photo-realistic
animatable human avatars from monocular videos. We then design a Semantic
Human-Body Annotator (SHA) which utilizes SMPL's semantic prior for efficient
body part semantic labeling. The generated labels are used to guide the
optimization of Gaussian semantic attributes. To address the limited receptive
field of point-level MLPs for local features, we also propose a 3D network that
integrates geometric and semantic associations for human avatar deformation. We
further implement three key strategies to enhance the semantic accuracy of 3D
Gaussians and rendering quality: semantic projection with 2D regularization,
semantic-guided density regularization and semantic-aware regularization with
neighborhood consistency. Extensive experiments demonstrate that SG-GS achieves
state-of-the-art geometry and appearance reconstruction performance.

### 摘要 (中文)

从单目视频重建逼真的可动人体拟人角色仍然是计算机视觉和图形学中极具挑战性的任务。最近，使用三维高斯来表示人类身体的方法已经出现，提供了更快的优化和实时渲染。然而，由于忽略了代表人体内在结构和连接的关键的人体语义信息，它们无法实现动态人体拟人的精细细节重建。为了应对这一问题，我们提出SG-GS，它使用嵌入了语义的三维高斯、骨骼驱动的刚性变形和非线性的布料动力学变形来创建逼真的可动人体拟人角色从单目视频。然后设计一个语义人体标注器（SHA），利用SMPL的语义先验进行高效的身体部分语义标签标记。生成的标签用于指导语义属性的优化。为了解决点级MLP在局部特征有限感知范围中的限制，我们还提出了一种3D网络，该网络结合几何和语义关联以对人类虚拟角色进行变形。我们进一步实施了三个关键策略来增强三维高斯的语义准确性和渲染质量：二维正则化下的语义投影、基于语义引导的密度正则化以及基于邻居一致性的语义感知正则化。广泛的实验展示了SG-GS在几何和外观重构性能方面达到了最先进的水平。

---

## Implicit Grid Convolution for Multi-Scale Image Super-Resolution

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09674v1)

### Abstract (English)

Recently, Super-Resolution (SR) achieved significant performance improvement
by employing neural networks. Most SR methods conventionally train a single
model for each targeted scale, which increases redundancy in training and
deployment in proportion to the number of scales targeted. This paper
challenges this conventional fixed-scale approach. Our preliminary analysis
reveals that, surprisingly, encoders trained at different scales extract
similar features from images. Furthermore, the commonly used scale-specific
upsampler, Sub-Pixel Convolution (SPConv), exhibits significant inter-scale
correlations. Based on these observations, we propose a framework for training
multiple integer scales simultaneously with a single model. We use a single
encoder to extract features and introduce a novel upsampler, Implicit Grid
Convolution~(IGConv), which integrates SPConv at all scales within a single
module to predict multiple scales. Our extensive experiments demonstrate that
training multiple scales with a single model reduces the training budget and
stored parameters by one-third while achieving equivalent inference latency and
comparable performance. Furthermore, we propose IGConv$^{+}$, which addresses
spectral bias and input-independent upsampling and uses ensemble prediction to
improve performance. As a result, SRFormer-IGConv$^{+}$ achieves a remarkable
0.25dB improvement in PSNR at Urban100$\times$4 while reducing the training
budget, stored parameters, and inference cost compared to the existing
SRFormer.

### 摘要 (中文)

最近，超分辨率（SR）通过使用神经网络取得了显著的性能改进。大多数SR方法通常为每个目标尺度训练一个模型，这增加了在多个尺度上训练和部署的冗余性。本论文挑战了这种固定尺度的方法论。我们的初步分析揭示了一个令人惊讶的事实，即不同尺度下编码器从图像中提取相似特征。此外，广泛使用的尺度特定插值子像素卷积（SPConv）显示具有明显的跨尺度相关性。基于这些观察，我们提出了一种同时训练多个整数尺度的框架，采用单一模型。我们使用单个编码器来提取特征，并引入一种新型的插值器，即隐式网格卷积（IGConv），该插值器集成于单个模块中的所有尺度内，以预测多个尺度。我们的大量实验表明，在与现有SRFormer相同的推理延迟和类似性能的同时，采用单一模型同时训练多个尺度可以减少三分之一的训练预算和存储参数，而获得等效的推理延迟和类似的表现。此外，我们提出了IGConv+，它解决了谱偏斜和输入独立插值，并使用ensemble预测改善性能。结果，SRFormer-IGConv+在Urban100$\times$4时PSNR提高了0.25dB，同时比现有SRFormer的训练预算、存储参数和推理成本降低了30%。

---

## Image-based Freeform Handwriting Authentication with Energy-oriented Self-Supervised Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09676v1)

### Abstract (English)

Freeform handwriting authentication verifies a person's identity from their
writing style and habits in messy handwriting data. This technique has gained
widespread attention in recent years as a valuable tool for various fields,
e.g., fraud prevention and cultural heritage protection. However, it still
remains a challenging task in reality due to three reasons: (i) severe damage,
(ii) complex high-dimensional features, and (iii) lack of supervision. To
address these issues, we propose SherlockNet, an energy-oriented two-branch
contrastive self-supervised learning framework for robust and fast freeform
handwriting authentication. It consists of four stages: (i) pre-processing:
converting manuscripts into energy distributions using a novel plug-and-play
energy-oriented operator to eliminate the influence of noise; (ii) generalized
pre-training: learning general representation through two-branch momentum-based
adaptive contrastive learning with the energy distributions, which handles the
high-dimensional features and spatial dependencies of handwriting; (iii)
personalized fine-tuning: calibrating the learned knowledge using a small
amount of labeled data from downstream tasks; and (iv) practical application:
identifying individual handwriting from scrambled, missing, or forged data
efficiently and conveniently. Considering the practicality, we construct EN-HA,
a novel dataset that simulates data forgery and severe damage in real
applications. Finally, we conduct extensive experiments on six benchmark
datasets including our EN-HA, and the results prove the robustness and
efficiency of SherlockNet.

### 摘要 (中文)

自由手写认证技术从个人书写风格和习惯的乱写的数据中验证一个人的身份。近年来，由于在各种领域（如欺诈预防和文化遗产保护）获得广泛的关注，它已成为一项有价值的工具。然而，在现实生活中，仍然存在三个挑战：（i）严重的损坏；（ii）复杂高维特征；（iii）缺乏监督。针对这些问题，我们提出SherlockNet框架，这是一个基于能量导向的双分支对比学习自监督学习框架，用于稳健快速的手写认证。它包括四个阶段：（i）预处理：使用一个全新的插件与即用型能量导向操作将手稿转换为能量分布，以消除噪音的影响；（ii）一般化预训练：通过两个分支的 Momentum 基于适应性对比学习来学习通用表示，同时处理高维特征和手写空间依赖性；（iii）个性化微调：利用少量来自下游任务标签的数据对所学知识进行校准；（iv）实际应用：高效便捷地从乱写、缺失或伪造的数据中识别出个人的书写。考虑到实用性，我们构建了一个名为EN-HA的新数据集，该数据集模拟了现实生活中的数据伪造和严重损害。最后，我们在六个基准测试数据集上进行了大量实验，包括我们的EN-HA，并证明了SherlockNet的稳健性和效率。

---

## MePT_ Multi-Representation Guided Prompt Tuning for Vision-Language Model

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09706v1)

### Abstract (English)

Recent advancements in pre-trained Vision-Language Models (VLMs) have
highlighted the significant potential of prompt tuning for adapting these
models to a wide range of downstream tasks. However, existing prompt tuning
methods typically map an image to a single representation, limiting the model's
ability to capture the diverse ways an image can be described. To address this
limitation, we investigate the impact of visual prompts on the model's
generalization capability and introduce a novel method termed
Multi-Representation Guided Prompt Tuning (MePT). Specifically, MePT employs a
three-branch framework that focuses on diverse salient regions, uncovering the
inherent knowledge within images which is crucial for robust generalization.
Further, we employ efficient self-ensemble techniques to integrate these
versatile image representations, allowing MePT to learn all conditional,
marginal, and fine-grained distributions effectively. We validate the
effectiveness of MePT through extensive experiments, demonstrating significant
improvements on both base-to-novel class prediction and domain generalization
tasks.

### 摘要 (中文)

最近，预训练视觉语言模型（VLM）的最新进展强调了通过调优提示来适应多种下游任务的巨大潜力。然而，现有的提示调优方法通常将图像映射到单一表示，限制了模型捕获图像多样性的能力。为了应对这一局限性，我们研究了视觉提示对模型泛化能力的影响，并提出了一个新的方法称为多模态引导提示调优（MePT）。具体来说，MePT采用了一个三分支框架，专注于多样化的关键区域，揭示了图像中蕴藏的关键知识，这对稳健的泛化至关重要。此外，我们还采用了高效自集成技术来整合这些多样化的人像表示，使MePT能够有效地学习所有条件分布、边际分布和细粒度分布。通过广泛的实验验证了MePT的有效性，证明了它在基线到新领域类预测以及一般化任务上的显著改进。

---

## Dataset Distillation for Histopathology Image Classification

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09709v1)

### Abstract (English)

Deep neural networks (DNNs) have exhibited remarkable success in the field of
histopathology image analysis. On the other hand, the contemporary trend of
employing large models and extensive datasets has underscored the significance
of dataset distillation, which involves compressing large-scale datasets into a
condensed set of synthetic samples, offering distinct advantages in improving
training efficiency and streamlining downstream applications. In this work, we
introduce a novel dataset distillation algorithm tailored for histopathology
image datasets (Histo-DD), which integrates stain normalisation and model
augmentation into the distillation progress. Such integration can substantially
enhance the compatibility with histopathology images that are often
characterised by high colour heterogeneity. We conduct a comprehensive
evaluation of the effectiveness of the proposed algorithm and the generated
histopathology samples in both patch-level and slide-level classification
tasks. The experimental results, carried out on three publicly available WSI
datasets, including Camelyon16, TCGA-IDH, and UniToPath, demonstrate that the
proposed Histo-DD can generate more informative synthetic patches than previous
coreset selection and patch sampling methods. Moreover, the synthetic samples
can preserve discriminative information, substantially reduce training efforts,
and exhibit architecture-agnostic properties. These advantages indicate that
synthetic samples can serve as an alternative to large-scale datasets.

### 摘要 (中文)

深度神经网络（DNN）在病理图像分析领域表现出色。另一方面，当前的趋势是采用大型模型和大量数据，强调了数据微调的重要性，即压缩大规模数据集形成一组合成样本的压缩过程，可以显著提高训练效率，并简化下游应用。在这项工作中，我们引入了一个专门针对病理图像数据集（Histo-DD）的独特数据微调算法，该算法整合了染色正态化和模型增强，以促进数据微调的进步。这种集成可以显著增强与往往具有高色彩异质性的病理图像相兼容性。我们在三个公开可用的WSI数据集中（Camelyon16、TCGA-IDH和UniToPath）进行了全面评估，包括拟议的Histo-DD生成的病理切片比先前的核心选择和片样采样方法更有效。此外，合成样本可以保留区分信息，大大减少培训努力，且架构无关属性。这些优势表明，合成样本可以作为一种替代大规模数据集。

---

## TraDiffusion_ Trajectory-Based Training-Free Image Generation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09739v1)

### Abstract (English)

In this work, we propose a training-free, trajectory-based controllable T2I
approach, termed TraDiffusion. This novel method allows users to effortlessly
guide image generation via mouse trajectories. To achieve precise control, we
design a distance awareness energy function to effectively guide latent
variables, ensuring that the focus of generation is within the areas defined by
the trajectory. The energy function encompasses a control function to draw the
generation closer to the specified trajectory and a movement function to
diminish activity in areas distant from the trajectory. Through extensive
experiments and qualitative assessments on the COCO dataset, the results reveal
that TraDiffusion facilitates simpler, more natural image control. Moreover, it
showcases the ability to manipulate salient regions, attributes, and
relationships within the generated images, alongside visual input based on
arbitrary or enhanced trajectories.

### 摘要 (中文)

在这一工作中，我们提出了一种无需训练的基于轨迹的可操控T2I方法，称为TraDiffusion。这种新颖的方法允许用户轻松地通过鼠标轨迹引导图像生成。为了实现精确控制，我们设计了一个距离感知能量函数来有效地指导隐变量，确保生成的重点位于轨迹定义的区域内。该能量函数包括一个控制功能以将生成拉近到指定的轨迹，并包含一个移动功能以减少远离轨迹活动区域的活动量。通过对COCO数据集进行广泛的实验和对生成图像的定性评估，结果表明，TraDiffusion有助于更简单、更自然的图像控制。此外，它展示了操纵生成图像中的突出部位、属性以及基于任意或增强轨迹的关联的能力，同时结合根据任意或增强轨迹的视觉输入。

---

## RealCustom___ Representing Images as Real-Word for Real-Time Customization

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09744v1)

### Abstract (English)

Text-to-image customization, which takes given texts and images depicting
given subjects as inputs, aims to synthesize new images that align with both
text semantics and subject appearance. This task provides precise control over
details that text alone cannot capture and is fundamental for various
real-world applications, garnering significant interest from academia and
industry. Existing works follow the pseudo-word paradigm, which involves
representing given subjects as pseudo-words and combining them with given texts
to collectively guide the generation. However, the inherent conflict and
entanglement between the pseudo-words and texts result in a dual-optimum
paradox, where subject similarity and text controllability cannot be optimal
simultaneously. We propose a novel real-words paradigm termed RealCustom++ that
instead represents subjects as non-conflict real words, thereby disentangling
subject similarity from text controllability and allowing both to be optimized
simultaneously. Specifically, RealCustom++ introduces a novel "train-inference"
decoupled framework: (1) During training, RealCustom++ learns the alignment
between vision conditions and all real words in the text, ensuring high
subject-similarity generation in open domains. This is achieved by the
cross-layer cross-scale projector to robustly and finely extract subject
features, and a curriculum training recipe that adapts the generated subject to
diverse poses and sizes. (2) During inference, leveraging the learned general
alignment, an adaptive mask guidance is proposed to only customize the
generation of the specific target real word, keeping other subject-irrelevant
regions uncontaminated to ensure high text-controllability in real-time.

### 摘要 (中文)

文本到图像定制化任务，即根据给定的文本和描绘给定主题的图像作为输入，旨在合成与文本语义和主题外观都相符的新图像。这项任务提供了文本无法捕捉的细节方面的精确控制，并对各种实际世界应用至关重要，吸引了学术界和工业界的极大关注。现有工作遵循伪词范式，涉及将给定的主题表示为伪词，并将其与给定的文本相结合，以共同引导生成。然而，伪词和文本之间的内在冲突和纠缠导致了双重最优悖论，其中主题相似性和可控性不能同时达到最佳值。我们提出了一种名为RealCustom++的新实词范式，它代表主题为非冲突的现实词语，从而解除了主题相似性的干扰，使两者可以同时优化。具体来说，RealCustom++引入了一个新的“训练推断”解耦框架：（1）在训练过程中，RealCustom++学习了视觉条件与所有文本中所有现实词语之间的匹配，确保开放域中的高主题相似度生成。这通过跨层跨尺度投影器来稳健且精细地提取主题特征，以及一个适应生成目标主题的不同姿态和大小的课程教学食谱来实现。（2）在推理期间，利用所学的一般匹配，提出了一种自适应掩模指引策略，仅定制特定目标真实词的生成，保持其他不相关的主题区域不受污染，以确保实时中的文本可控性。

---

## A Unified Framework for Iris Anti-Spoofing_ Introducing IrisGeneral Dataset and Masked-MoE Method

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09752v1)

### Abstract (English)

Iris recognition is widely used in high-security scenarios due to its
stability and distinctiveness. However, the acquisition of iris images
typically requires near-infrared illumination and near-infrared band filters,
leading to significant and consistent differences in imaging across devices.
This underscores the importance of developing cross-domain capabilities in iris
anti-spoofing methods. Despite this need, there is no dataset available that
comprehensively evaluates the generalization ability of the iris anti-spoofing
task. To address this gap, we propose the IrisGeneral dataset, which includes
10 subsets, belonging to 7 databases, published by 4 institutions, collected
with 6 types of devices. IrisGeneral is designed with three protocols, aimed at
evaluating average performance, cross-racial generalization, and cross-device
generalization of iris anti-spoofing models. To tackle the challenge of
integrating multiple sub-datasets in IrisGeneral, we employ multiple parameter
sets to learn from the various subsets. Specifically, we utilize the Mixture of
Experts (MoE) to fit complex data distributions using multiple sub-neural
networks. To further enhance the generalization capabilities, we introduce a
novel method Masked-MoE (MMoE). It randomly masks a portion of tokens for some
experts and requires their outputs to be similar to the unmasked experts, which
improves the generalization ability and effectively mitigates the overfitting
issue produced by MoE. We selected ResNet50, VIT-B/16, CLIP, and FLIP as
representative models and benchmarked them on the IrisGeneral dataset.
Experimental results demonstrate that our proposed MMoE with CLIP achieves the
best performance on IrisGeneral.

### 摘要 (中文)

虹膜识别技术在高安全场景中得到了广泛应用，因为它的稳定性及独特性。然而，获取虹膜图像通常需要近红外照明和近红外滤波器，这导致了设备间图像的显著且一致差异。这一现象强调了开发跨域能力的重要性。尽管如此，尚无全面评估虹膜反欺骗任务泛化能力的数据集可供选择。为了应对这一空白，我们提出了一种名为IrisGeneral的数据集，它包括7个数据库中的10个子集，由4个机构收集，并使用6种设备进行采集。IrisGeneral设计有三个协议，旨在评估平均性能、跨种族泛化能力和跨设备泛化能力的虹膜反欺骗模型。为解决集成多个子集在IrisGeneral中的挑战，我们利用混合专家（MoE）学习复杂数据分布，采用多神经网络的多种参数设置。具体而言，我们将利用MoE来训练复杂的数据分布，通过多个子神经网络。为了进一步增强泛化能力，我们引入了一个新的方法Masked-MoE（MMoE）。它随机屏蔽某些专家的部分令牌，要求他们的输出与未屏蔽的专家相似，从而提高了泛化能力和有效缓解了MoE产生的过拟合问题。我们选择了ResNet50、VIT-B/16、CLIP和FLIP作为代表性模型，并将其基准测试在IrisGeneral数据集中。实验结果表明，我们的新型MMoE结合CLIP在IrisGeneral上取得了最佳性能。

---

## Cross-composition Feature Disentanglement for Compositional Zero-shot Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09786v1)

### Abstract (English)

Disentanglement of visual features of primitives (i.e., attributes and
objects) has shown exceptional results in Compositional Zero-shot Learning
(CZSL). However, due to the feature divergence of an attribute (resp. object)
when combined with different objects (resp. attributes), it is challenging to
learn disentangled primitive features that are general across different
compositions. To this end, we propose the solution of cross-composition feature
disentanglement, which takes multiple primitive-sharing compositions as inputs
and constrains the disentangled primitive features to be general across these
compositions. More specifically, we leverage a compositional graph to define
the overall primitive-sharing relationships between compositions, and build a
task-specific architecture upon the recently successful large pre-trained
vision-language model (VLM) CLIP, with dual cross-composition disentangling
adapters (called L-Adapter and V-Adapter) inserted into CLIP's frozen text and
image encoders, respectively. Evaluation on three popular CZSL benchmarks shows
that our proposed solution significantly improves the performance of CZSL, and
its components have been verified by solid ablation studies.

### 摘要 (中文)

在构建复合零-shot学习（Compositional Zero-shot Learning，CZSL）的视觉特征时，解耦原始属性（即属性和对象）的视觉特征显示了非常出色的结果。然而，由于属性与不同对象相结合时（或反之亦然），解耦原始属性变得具有挑战性，使其难以学习通用的解耦原始属性。为此，我们提出了跨组合特征解耦的方法，该方法接受多个共享原始成分的输入，并将解耦的原始属性约束为这些组合中的一般性。更具体地说，我们将一个构成图定义为各个组合作用之间的总体共享关系，然后在最近成功的大型预训练视觉语言模型（VLM）CLIP上建立任务特定架构，其中双交叉组合解耦适配器（称为L-适配器和V-适配器）分别插入到CLIP的冻结文本编码器和图像编码器中。对三个流行CZSL基准进行评估后发现，我们的解决方案显著提高了CZSL的表现，并且其组件通过坚实的微调研究得到了验证。

---

## Latent Diffusion for Guided Document Table Generation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09800v1)

### Abstract (English)

Obtaining annotated table structure data for complex tables is a challenging
task due to the inherent diversity and complexity of real-world document
layouts. The scarcity of publicly available datasets with comprehensive
annotations for intricate table structures hinders the development and
evaluation of models designed for such scenarios. This research paper
introduces a novel approach for generating annotated images for table structure
by leveraging conditioned mask images of rows and columns through the
application of latent diffusion models. The proposed method aims to enhance the
quality of synthetic data used for training object detection models.
Specifically, the study employs a conditioning mechanism to guide the
generation of complex document table images, ensuring a realistic
representation of table layouts. To evaluate the effectiveness of the generated
data, we employ the popular YOLOv5 object detection model for training. The
generated table images serve as valuable training samples, enriching the
dataset with diverse table structures. The model is subsequently tested on the
challenging pubtables-1m testset, a benchmark for table structure recognition
in complex document layouts. Experimental results demonstrate that the
introduced approach significantly improves the quality of synthetic data for
training, leading to YOLOv5 models with enhanced performance. The mean Average
Precision (mAP) values obtained on the pubtables-1m testset showcase results
closely aligned with state-of-the-art methods. Furthermore, low FID results
obtained on the synthetic data further validate the efficacy of the proposed
methodology in generating annotated images for table structure.

### 摘要 (中文)

由于真实世界文档布局的内在多样性与复杂性，获取复杂表格结构的数据标注结构是一个具有挑战性的任务。缺乏包含全面注释的复杂表格结构的公开可用数据集阻碍了设计此类场景模型的发展和评估。本研究论文提出了一种新颖的方法来生成表结构的注释图像，通过应用隐式扩散模型对行和列进行条件化遮罩图像。提出的该方法旨在增强用于训练对象检测模型合成数据的质量。具体来说，研究采用一种指导机制引导复杂文档表格图像的生成，以确保表布局的真实表示。为了评估生成数据的有效性，我们使用流行的YOLOv5对象检测模型进行培训。生成的表格图像作为有价值的训练样本，丰富了数据集中的多样化的表格结构。随后，该模型被测试在公共的pubtables-1m测试集上，这是复杂文档布局中识别表格结构的基准。实验结果表明，引入的方法显著提高了合成数据用于训练的质量，从而获得了YOLOV5模型有改进性能的YOLOV5模型。在pubtables-1m测试集上的平均精度（mAP）值展示了与当前最先进的方法结果接近的结果。此外，合成数据获得的低FID结果进一步验证了使用生成的表结构图像进行注释图像产生的有效性和可行性。

---

## SurgicaL-CD_ Generating Surgical Images via Unpaired Image Translation with Latent Consistency Diffusion Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09822v1)

### Abstract (English)

Computer-assisted surgery (CAS) systems are designed to assist surgeons
during procedures, thereby reducing complications and enhancing patient care.
Training machine learning models for these systems requires a large corpus of
annotated datasets, which is challenging to obtain in the surgical domain due
to patient privacy concerns and the significant labeling effort required from
doctors. Previous methods have explored unpaired image translation using
generative models to create realistic surgical images from simulations.
However, these approaches have struggled to produce high-quality, diverse
surgical images. In this work, we introduce \emph{SurgicaL-CD}, a
consistency-distilled diffusion method to generate realistic surgical images
with only a few sampling steps without paired data. We evaluate our approach on
three datasets, assessing the generated images in terms of quality and utility
as downstream training datasets. Our results demonstrate that our method
outperforms GANs and diffusion-based approaches. Our code is available at
\url{https://gitlab.com/nct_tso_public/gan2diffusion}.

### 摘要 (中文)

计算机辅助手术（CAS）系统旨在协助医生在操作过程中，从而减少并发症并提高患者护理水平。为了训练这些系统，需要大量的标注数据集，而在手术领域由于隐私顾虑和医生所需的巨大标注工作，这使得获取大量注释数据变得具有挑战性。以往的方法尝试使用生成模型通过无对齐图像翻译来创建从模拟中生成的现实主义外科图像。然而，这些方法未能产生高质量、多样化的外科图像。本工作中，我们引入了一种名为SurgicaL-CD的等同式扩散方法，仅需几步采样步骤即可生成高质量、多样的外科图像。我们在三个数据集中评估生成的图像，在下游培训数据方面作为质量和实用性的标准。我们的结果表明，我们的方法优于GANs和基于扩散的方法。我们的代码可在https://gitlab.com/nct_tso_public/gan2diffusion中获得。

---

## OccMamba_ Semantic Occupancy Prediction with State Space Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09859v1)

### Abstract (English)

Training deep learning models for semantic occupancy prediction is
challenging due to factors such as a large number of occupancy cells, severe
occlusion, limited visual cues, complicated driving scenarios, etc. Recent
methods often adopt transformer-based architectures given their strong
capability in learning input-conditioned weights and long-range relationships.
However, transformer-based networks are notorious for their quadratic
computation complexity, seriously undermining their efficacy and deployment in
semantic occupancy prediction. Inspired by the global modeling and linear
computation complexity of the Mamba architecture, we present the first
Mamba-based network for semantic occupancy prediction, termed OccMamba.
However, directly applying the Mamba architecture to the occupancy prediction
task yields unsatisfactory performance due to the inherent domain gap between
the linguistic and 3D domains. To relieve this problem, we present a simple yet
effective 3D-to-1D reordering operation, i.e., height-prioritized 2D Hilbert
expansion. It can maximally retain the spatial structure of point clouds as
well as facilitate the processing of Mamba blocks. Our OccMamba achieves
state-of-the-art performance on three prevalent occupancy prediction
benchmarks, including OpenOccupancy, SemanticKITTI and SemanticPOSS. Notably,
on OpenOccupancy, our OccMamba outperforms the previous state-of-the-art Co-Occ
by 3.1% IoU and 3.2% mIoU, respectively. Codes will be released upon
publication.

### 摘要 (中文)

训练深度学习模型进行语义占据预测是一项挑战，因为有大量占位符细胞、严重的遮挡、有限的视觉线索等因素。最近的方法通常采用Transformer架构，因其在学习输入条件权重和长距离关系方面的能力强。然而，Transformer网络以其线性计算复杂度而臭名昭著，严重影响了它们在语义占据预测中的有效性以及部署。我们受到了马梅亚架构全局建模和线性计算复杂度的启发，提出了第一个基于马梅亚架构的语义占据预测模型，称为OccMamba。然而，直接应用马梅亚架构到占用预测任务中却无法获得满意的表现，因为语言域与三维域之间存在内在的领域差距。为了缓解这个问题，我们提出了一种简单且有效的三维到一维重新排序操作，即高度优先的二维希尔伯特扩展。它可以最大限度地保留点云的空间结构，并有助于处理马梅亚块的处理。我们的OccMamba在三个流行的占用预测基准上取得了最先进的表现，包括OpenOccupancy、SemanticKITTI和SemanticPOSS。值得注意的是，在OpenOccupancy上，我们的OccMamba相对于Co-Occ的IOU和mIoU分别提高了3.1%和3.2%。代码将在发表后发布。

---

## SAM-UNet_Enhancing Zero-Shot Segmentation of SAM for Universal Medical Images

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09886v1)

### Abstract (English)

Segment Anything Model (SAM) has demonstrated impressive performance on a
wide range of natural image segmentation tasks. However, its performance
significantly deteriorates when directly applied to medical domain, due to the
remarkable differences between natural images and medical images. Some
researchers have attempted to train SAM on large scale medical datasets.
However, poor zero-shot performance is observed from the experimental results.
In this context, inspired by the superior performance of U-Net-like models in
medical image segmentation, we propose SAMUNet, a new foundation model which
incorporates U-Net to the original SAM, to fully leverage the powerful
contextual modeling ability of convolutions. To be specific, we parallel a
convolutional branch in the image encoder, which is trained independently with
the vision Transformer branch frozen. Additionally, we employ multi-scale
fusion in the mask decoder, to facilitate accurate segmentation of objects with
different scales. We train SAM-UNet on SA-Med2D-16M, the largest 2-dimensional
medical image segmentation dataset to date, yielding a universal pretrained
model for medical images. Extensive experiments are conducted to evaluate the
performance of the model, and state-of-the-art result is achieved, with a dice
similarity coefficient score of 0.883 on SA-Med2D-16M dataset. Specifically, in
zero-shot segmentation experiments, our model not only significantly
outperforms previous large medical SAM models across all modalities, but also
substantially mitigates the performance degradation seen on unseen modalities.
It should be highlighted that SAM-UNet is an efficient and extensible
foundation model, which can be further fine-tuned for other downstream tasks in
medical community. The code is available at
https://github.com/Hhankyangg/sam-unet.

### 摘要 (中文)

SegmentAnything模型（SAM）在自然图像分割任务上表现出色，但在直接应用于医学领域时性能显著下降，因为自然图像和医学图像之间存在巨大的差异。一些研究人员尝试使用大规模的医疗数据集来训练SAM。然而，在实验结果中观察到零样本性能较差。在此背景下，我们受到了U-Net类似模型在医学图像分割中的出色表现的启发，提出了一种新的基础模型SAMUNet，它结合了U-Net，以充分利用卷积的强大上下文建模能力。具体来说，我们在图像编码器并行了一个卷积分支，该分支与视觉Transformer分支冻结训练。此外，我们还采用了多尺度融合在掩码解码器中，以便准确地对不同尺度的对象进行分割。我们将SAM-UNet用于SA-Med2D-16M，这是迄今为止最大的二维医学图像分割数据集，生成了医疗图像的通用预训练模型。进行了广泛的实验来评估模型的表现，并取得了优异的结果，在SA-Med2D-16M数据集上的Dice相似系数分数为0.883。具体而言，在零样本分割实验中，我们的模型不仅在所有模式上比以前的大规模医学SAM模型表现出更优的性能，而且也有效地缓解了未见模式下的性能退化。应该强调的是，SAM-UNet是一种高效且可扩展的基础模型，可以在医学社区进一步微调其他下游任务。代码可在以下链接获取：https://github.com/Hhankyangg/sam-unet。

---

## Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09919v1)

### Abstract (English)

Procedural activity videos often exhibit a long-tailed action distribution
due to varying action frequencies and durations. However, state-of-the-art
temporal action segmentation methods overlook the long tail and fail to
recognize tail actions. Existing long-tail methods make class-independent
assumptions and struggle to identify tail classes when applied to temporal
segmentation frameworks. This work proposes a novel group-wise temporal logit
adjustment~(G-TLA) framework that combines a group-wise softmax formulation
while leveraging activity information and action ordering for logit adjustment.
The proposed framework significantly improves in segmenting tail actions
without any performance loss on head actions.

### 摘要 (中文)

由于行动频率和持续时间的不一致，流程活动视频经常表现出长尾的动作分布。然而，最先进的时序动作分割方法忽略了长尾并无法识别尾部动作。现有的长尾方法假设是无差别的，并且在应用到时序分割框架时，难以识别尾类。本工作提出了一种新颖的组际时序逻辑调整(G-TLA)架构，该架构结合了组际softmax表达式，同时利用活动信息和动作顺序进行逻辑调整。提出的架构显著提高了对尾部动作的分割能力，而不会对头部动作造成任何性能损失。

---

## Boosting Open-Domain Continual Learning via Leveraging Intra-domain Category-aware Prototype

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09984v1)

### Abstract (English)

Despite recent progress in enhancing the efficacy of Open-Domain Continual
Learning (ODCL) in Vision-Language Models (VLM), failing to (1) correctly
identify the Task-ID of a test image and (2) use only the category set
corresponding to the Task-ID, while preserving the knowledge related to each
domain, cannot address the two primary challenges of ODCL: forgetting old
knowledge and maintaining zero-shot capabilities, as well as the confusions
caused by category-relatedness between domains. In this paper, we propose a
simple yet effective solution: leveraging intra-domain category-aware
prototypes for ODCL in CLIP (DPeCLIP), where the prototype is the key to
bridging the above two processes. Concretely, we propose a training-free
Task-ID discriminator method, by utilizing prototypes as classifiers for
identifying Task-IDs. Furthermore, to maintain the knowledge corresponding to
each domain, we incorporate intra-domain category-aware prototypes as domain
prior prompts into the training process. Extensive experiments conducted on 11
different datasets demonstrate the effectiveness of our approach, achieving
2.37% and 1.14% average improvement in class-incremental and task-incremental
settings, respectively.

### 摘要 (中文)

尽管在视觉语言模型(VLM)中增强Open Domain Continual Learning(ODCL)的有效性取得了进展，但（1）无法正确识别测试图像的任务ID，并且仅使用与任务ID对应的类别集来保留每个领域相关的知识，都无法解决ODCL的两个主要挑战：遗忘旧的知识和保持零-shot能力，以及由于域间相关性的困惑。在此文中，我们提出了一种简单而有效的解决方案：利用CLIP中的内嵌类别的感知原型在ODCL中，其中原型是连接这两个过程的关键。具体来说，我们提出了一个无监督的任务ID鉴别器方法，通过使用原型作为分类器来识别任务ID。此外，为了维持每个领域的知识，我们将内嵌类别的感知原型纳入训练过程中作为域先验提示。针对11个不同数据集进行的广泛实验展示了我们的方法的有效性，分别实现了2.37％和1.14％的平均提升，在增量学习设置中。

---

## P3P_ Pseudo-3D Pre-training for Scaling 3D Masked Autoencoders

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10007v1)

### Abstract (English)

3D pre-training is crucial to 3D perception tasks. However, limited by the
difficulties in collecting clean 3D data, 3D pre-training consistently faced
data scaling challenges. Inspired by semi-supervised learning leveraging
limited labeled data and a large amount of unlabeled data, in this work, we
propose a novel self-supervised pre-training framework utilizing the real 3D
data and the pseudo-3D data lifted from images by a large depth estimation
model. Another challenge lies in the efficiency. Previous methods such as
Point-BERT and Point-MAE, employ k nearest neighbors to embed 3D tokens,
requiring quadratic time complexity. To efficiently pre-train on such a large
amount of data, we propose a linear-time-complexity token embedding strategy
and a training-efficient 2D reconstruction target. Our method achieves
state-of-the-art performance in 3D classification and few-shot learning while
maintaining high pre-training and downstream fine-tuning efficiency.

### 摘要 (中文)

三维预训练对于三维感知任务至关重要。然而，由于收集干净的三维数据的困难，三维预训练一直面临数据缩放挑战。在这一工作中，我们提出了一种利用真实三维数据和由深度估计模型提取的伪三维数据来预训练的新自监督预训练框架。另一个挑战是效率问题。与点BERT和点MSE等方法不同，它们采用k最近邻来嵌入三维令牌，需要线性时间复杂度。为了高效地在如此大量数据上进行预训练，我们提出了一个线性时间复杂度的令牌嵌入策略和一个训练效率高的二维重建目标。我们的方法在三维分类和少量样本学习方面取得了最先进的性能，同时保持了高预训练和下游微调效率。

---

## CLIPCleaner_ Cleaning Noisy Labels with CLIP

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10012v1)

### Abstract (English)

Learning with Noisy labels (LNL) poses a significant challenge for the
Machine Learning community. Some of the most widely used approaches that select
as clean samples for which the model itself (the in-training model) has high
confidence, e.g., `small loss', can suffer from the so called
`self-confirmation' bias. This bias arises because the in-training model, is at
least partially trained on the noisy labels. Furthermore, in the classification
case, an additional challenge arises because some of the label noise is between
classes that are visually very similar (`hard noise'). This paper addresses
these challenges by proposing a method (\textit{CLIPCleaner}) that leverages
CLIP, a powerful Vision-Language (VL) model for constructing a zero-shot
classifier for efficient, offline, clean sample selection. This has the
advantage that the sample selection is decoupled from the in-training model and
that the sample selection is aware of the semantic and visual similarities
between the classes due to the way that CLIP is trained. We provide theoretical
justifications and empirical evidence to demonstrate the advantages of CLIP for
LNL compared to conventional pre-trained models. Compared to current methods
that combine iterative sample selection with various techniques,
\textit{CLIPCleaner} offers a simple, single-step approach that achieves
competitive or superior performance on benchmark datasets. To the best of our
knowledge, this is the first time a VL model has been used for sample selection
to address the problem of Learning with Noisy Labels (LNL), highlighting their
potential in the domain.

### 摘要 (中文)

学习有噪声标签（LNL）对机器学习社区提出了重大挑战。一些被选为干净样本的最广泛使用的方法，其中模型自身（训练集模型）具有很高的信心，例如“小损失”，可能会受到所谓的“自我确认”偏差的影响。这个偏差是因为在训练集中部分模型是在噪声标签上进行的。此外，在分类情况下，还存在另一个挑战，即一部分标签噪音是两个视觉非常相似的类之间的噪音（硬噪音）。本论文通过提出一种方法（CLIPCleaner），利用强大的视觉语言（VL）模型CLIP，来解决这些挑战。这种方法的优势在于，样本选择与训练集模型解耦，并且由于CLIP的培训方式，该方法能够意识到不同类别的语义和视觉相似性。我们提供了理论依据和实证证据来证明CLIP在LNL中优于预训练模型的优势。与当前结合迭代样本选择和其他技术的各种方法相比，本文提出的CLIPCleaner提供了一种简单、一步到位的方法，可以在基准数据集上达到或超过竞争或优越的表现。到目前为止，这是我们第一次使用VL模型用于处理学习有噪声标签（LNL）问题，强调了它们在领域中的潜力。

---

## Towards Robust Federated Image Classification_ An Empirical Study of Weight Selection Strategies in Manufacturing

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10024v1)

### Abstract (English)

In the realm of Federated Learning (FL), particularly within the
manufacturing sector, the strategy for selecting client weights for server
aggregation is pivotal for model performance. This study investigates the
comparative effectiveness of two weight selection strategies: Final Epoch
Weight Selection (FEWS) and Optimal Epoch Weight Selection (OEWS). Designed for
manufacturing contexts where collaboration typically involves a limited number
of partners (two to four clients), our research focuses on federated image
classification tasks. We employ various neural network architectures, including
EfficientNet, ResNet, and VGG, to assess the impact of these weight selection
strategies on model convergence and robustness.
  Our research aims to determine whether FEWS or OEWS enhances the global FL
model's performance across communication rounds (CRs). Through empirical
analysis and rigorous experimentation, we seek to provide valuable insights for
optimizing FL implementations in manufacturing, ensuring that collaborative
efforts yield the most effective and reliable models with a limited number of
participating clients. The findings from this study are expected to refine FL
practices significantly in manufacturing, thereby enhancing the efficiency and
performance of collaborative machine learning endeavors in this vital sector.

### 摘要 (中文)

在联邦学习（Federated Learning，简称FL）的领域中，特别是在制造业中，选择客户端权重对服务器聚合策略至关重要。本研究旨在比较两种权重选择策略：最终时期权重选择（Final Epoch Weight Selection，FEWS）和最优时期权重选择（Optimal Epoch Weight Selection，OEWS）。我们的研究专注于制造环境中合作通常涉及有限数量的合作伙伴（两个到四个客户）的联邦图像分类任务。我们使用各种神经网络架构，包括EfficientNet、ResNet和VGG，来评估这些权重选择策略对模型收敛性和鲁棒性的影响。
我们的研究目标是确定FEWS或OEWS是否能提高全球FL模型在通信轮次（CRs）中的性能。通过实证分析和严格实验，我们希望提供优化制造业FL实施的关键见解，以确保有限参与客户的协作努力能够产生最有效和可靠的最佳模型。从这项研究中得出的发现预计将显著改进制造业的FL实践，从而增强这一重要部门内高效和高性能的合作机器学习努力。

该研究的结果预期会对制造业的FL实践产生重大影响，从而提升在这一关键领域的高效和高性能的合作机器学习努力。

---

## Dynamic Label Injection for Imbalanced Industrial Defect Segmentation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10031v1)

### Abstract (English)

In this work, we propose a simple yet effective method to tackle the problem
of imbalanced multi-class semantic segmentation in deep learning systems. One
of the key properties for a good training set is the balancing among the
classes. When the input distribution is heavily imbalanced in the number of
instances, the learning process could be hindered or difficult to carry on. To
this end, we propose a Dynamic Label Injection (DLI) algorithm to impose a
uniform distribution in the input batch. Our algorithm computes the current
batch defect distribution and re-balances it by transferring defects using a
combination of Poisson-based seamless image cloning and cut-paste techniques. A
thorough experimental section on the Magnetic Tiles dataset shows better
results of DLI compared to other balancing loss approaches also in the
challenging weakly-supervised setup. The code is available at
https://github.com/covisionlab/dynamic-label-injection.git

### 摘要 (中文)

在这一工作中，我们提出了一种简单而有效的方法来解决深度学习系统中不平衡多类语义分割问题。一个良好的训练集的一个重要属性是类别之间的平衡。当输入分布中的实例数量严重失衡时，学习过程可能会受阻或难以进行。为此，我们提出了动态标签注入（DLI）算法以强制输入批次中的均匀分布。我们的算法计算当前批次的缺陷分布，并通过使用基于Poisson的无缝图像克隆和剪切粘贴技术重新平衡它。磁性瓷砖数据集上详细实验部分的结果表明，在弱监督设置中DLI与其他平衡损失方法相比有更好的结果。代码可在 https://github.com/covisionlab/dynamic-label-injection.git 下获取。

---

## SHARP_ Segmentation of Hands and Arms by Range using Pseudo-Depth for Enhanced Egocentric 3D Hand Pose Estimation and Action Recognition

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10037v1)

### Abstract (English)

Hand pose represents key information for action recognition in the egocentric
perspective, where the user is interacting with objects. We propose to improve
egocentric 3D hand pose estimation based on RGB frames only by using
pseudo-depth images. Incorporating state-of-the-art single RGB image depth
estimation techniques, we generate pseudo-depth representations of the frames
and use distance knowledge to segment irrelevant parts of the scene. The
resulting depth maps are then used as segmentation masks for the RGB frames.
Experimental results on H2O Dataset confirm the high accuracy of the estimated
pose with our method in an action recognition task. The 3D hand pose, together
with information from object detection, is processed by a transformer-based
action recognition network, resulting in an accuracy of 91.73%, outperforming
all state-of-the-art methods. Estimations of 3D hand pose result in competitive
performance with existing methods with a mean pose error of 28.66 mm. This
method opens up new possibilities for employing distance information in
egocentric 3D hand pose estimation without relying on depth sensors.

### 摘要 (中文)

手部姿势在视角中代表了动作识别的关键信息，用户与对象进行交互。我们提出通过仅使用伪深度图像来改进基于RGB帧的自洽三维手部姿势估计的方法。结合最先进的单像素RGB图像深度估计技术，我们可以生成伪深度表示，并利用距离知识来分割场景中的无关部分。最终，这些深度地图被用于RGB帧的分割掩码。H2O数据集上的实验结果证实，在动作识别任务中，以我们的方法获得的手部姿势估计具有很高的准确度。3D手部姿势，以及从物体检测的信息，由一个基于Transformer的动作识别网络处理，准确性达到91.73%，超过了所有现有最先进的方法。3D手部姿势的估计结果在现有的方法中获得了竞争性的性能，平均pose误差为28.66毫米。这种方法为我们提供了在不依赖于深度传感器的情况下在自洽三维手部姿势估计中使用距离信息的新可能性。

---

## Implicit Gaussian Splatting with Efficient Multi-Level Tri-Plane Representation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10041v1)

### Abstract (English)

Recent advancements in photo-realistic novel view synthesis have been
significantly driven by Gaussian Splatting (3DGS). Nevertheless, the explicit
nature of 3DGS data entails considerable storage requirements, highlighting a
pressing need for more efficient data representations. To address this, we
present Implicit Gaussian Splatting (IGS), an innovative hybrid model that
integrates explicit point clouds with implicit feature embeddings through a
multi-level tri-plane architecture. This architecture features 2D feature grids
at various resolutions across different levels, facilitating continuous spatial
domain representation and enhancing spatial correlations among Gaussian
primitives. Building upon this foundation, we introduce a level-based
progressive training scheme, which incorporates explicit spatial
regularization. This method capitalizes on spatial correlations to enhance both
the rendering quality and the compactness of the IGS representation.
Furthermore, we propose a novel compression pipeline tailored for both point
clouds and 2D feature grids, considering the entropy variations across
different levels. Extensive experimental evaluations demonstrate that our
algorithm can deliver high-quality rendering using only a few MBs, effectively
balancing storage efficiency and rendering fidelity, and yielding results that
are competitive with the state-of-the-art.

### 摘要 (中文)

最近在照片真实度小说视图合成方面取得的显著进展，主要得益于Gaussian Splatting（3DGS）。然而，3DGS数据的明确性导致了大量存储需求，从而提出了更有效的数据表示的需求。为此，我们提出了一种创新的混合模型——隐式高斯散点插值（IGS），该模型通过一个多层三角形架构结合了明确的点云和隐含特征嵌入。这个架构在不同级别上具有二维特征网格，提供了连续的空间域表示，并增强了不同级别的高斯实体之间的空间相关性。基于这一基础，我们引入了一个基于水平级的进步训练方案，其中包含了对明确定位的局部监督。这种方法利用空间相关性来增强IGS表示的质量和紧凑性。此外，我们还提出了一种专门针对点云和二维特征网格的新压缩管道，考虑到不同级别的熵变化。广泛的实验评估表明，我们的算法仅使用几兆字节即可实现高质量的渲染，有效地平衡了存储效率和渲染精度，并产生了与当前最先进的结果相竞争的结果。

---

## LNQ 2023 challenge_ Benchmark of weakly-supervised techniques for mediastinal lymph node quantification

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10069v1)

### Abstract (English)

Accurate assessment of lymph node size in 3D CT scans is crucial for cancer
staging, therapeutic management, and monitoring treatment response. Existing
state-of-the-art segmentation frameworks in medical imaging often rely on fully
annotated datasets. However, for lymph node segmentation, these datasets are
typically small due to the extensive time and expertise required to annotate
the numerous lymph nodes in 3D CT scans. Weakly-supervised learning, which
leverages incomplete or noisy annotations, has recently gained interest in the
medical imaging community as a potential solution. Despite the variety of
weakly-supervised techniques proposed, most have been validated only on private
datasets or small publicly available datasets. To address this limitation, the
Mediastinal Lymph Node Quantification (LNQ) challenge was organized in
conjunction with the 26th International Conference on Medical Image Computing
and Computer Assisted Intervention (MICCAI 2023). This challenge aimed to
advance weakly-supervised segmentation methods by providing a new, partially
annotated dataset and a robust evaluation framework. A total of 16 teams from 5
countries submitted predictions to the validation leaderboard, and 6 teams from
3 countries participated in the evaluation phase. The results highlighted both
the potential and the current limitations of weakly-supervised approaches. On
one hand, weakly-supervised approaches obtained relatively good performance
with a median Dice score of $61.0\%$. On the other hand, top-ranked teams, with
a median Dice score exceeding $70\%$, boosted their performance by leveraging
smaller but fully annotated datasets to combine weak supervision and full
supervision. This highlights both the promise of weakly-supervised methods and
the ongoing need for high-quality, fully annotated data to achieve higher
segmentation performance.

### 摘要 (中文)

在三维CT扫描中准确评估淋巴结大小对于癌症的分期、治疗管理以及监测治疗反应至关重要。现有的医学影像领域最先进的分割框架通常依赖于完全标注的数据集。然而，由于对3D CT扫描中的大量淋巴结进行标记所需的大量时间和专业知识，这些数据集通常是较小的。弱监督学习，即利用不完整或噪声标注的数据，近年来引起了医疗图像社区的兴趣，作为一种潜在解决方案。尽管提出了一系列各种类型的弱监督技术，但大多数都只验证了私人数据集或小公共可用数据集。为了应对这一限制，Mediastinal Lymph Node Quantification（LNQ）挑战在第26届国际医学成像计算与辅助干预会议（MICCAI 2023）联合组织。该挑战旨在通过提供一个新且部分标注的数据集和一套稳健的评价框架来推进弱监督分割方法。共有来自5个国家的16支队伍提交预测到验证排名榜，并有来自3个国家的6支队伍参加了评估阶段。结果展示了弱监督方法潜力的同时也揭示了当前局限性。一方面，弱监督方法获得了相对良好的性能，其中中位数Dice分数为$61.0\%$。另一方面，获得高排名的团队，其中位数Dice分数超过$70\%$，通过使用更少但已完全标注的数据集来结合弱监督和全监督，提高了他们的表现。这既显示了弱监督方法潜力同时也表明了需要高质量、完全标注数据以实现更高分割性能的持续需求。

---

## Modelling the Distribution of Human Motion for Sign Language Assessment

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10073v1)

### Abstract (English)

Sign Language Assessment (SLA) tools are useful to aid in language learning
and are underdeveloped. Previous work has focused on isolated signs or
comparison against a single reference video to assess Sign Languages (SL). This
paper introduces a novel SLA tool designed to evaluate the comprehensibility of
SL by modelling the natural distribution of human motion. We train our pipeline
on data from native signers and evaluate it using SL learners. We compare our
results to ratings from a human raters study and find strong correlation
between human ratings and our tool. We visually demonstrate our tools ability
to detect anomalous results spatio-temporally, providing actionable feedback to
aid in SL learning and assessment.

### 摘要 (中文)

手语评估（SLA）工具对语言学习有帮助，并且发展不足。以前的工作集中在孤立的手语或与单一参考视频进行比较来评估手语（SL）。本论文介绍了一种新的SLA工具，旨在通过模型自然运动分布来评估SL的可理解性。我们训练管道使用来自母语听者的数据并对其进行评估。我们比较我们的结果与人类评级者的研究中的评级，并发现人类评级与我们的工具之间的强相关性。我们视觉上展示了我们的工具能够检测异常结果的空间-时间差异，提供行动反馈以促进SL的学习和评估。

---

## Video Object Segmentation via SAM 2_ The 4th Solution for LSVOS Challenge VOS Track

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10125v1)

### Abstract (English)

Video Object Segmentation (VOS) task aims to segmenting a particular object
instance throughout the entire video sequence given only the object mask of the
first frame. Recently, Segment Anything Model 2 (SAM 2) is proposed, which is a
foundation model towards solving promptable visual segmentation in images and
videos. SAM 2 builds a data engine, which improves model and data via user
interaction, to collect the largest video segmentation dataset to date. SAM 2
is a simple transformer architecture with streaming memory for real-time video
processing, which trained on the date provides strong performance across a wide
range of tasks. In this work, we evaluate the zero-shot performance of SAM 2 on
the more challenging VOS datasets MOSE and LVOS. Without fine-tuning on the
training set, SAM 2 achieved 75.79 J&F on the test set and ranked 4th place for
6th LSVOS Challenge VOS Track.

### 摘要 (中文)

视频对象分割（VOS）任务的目标是根据第一帧的物体掩码，从整个视频序列中分割特定的对象实例。最近提出了Segment Anything Model 2（SAM 2），这是一个在图像和视频中解决可提示视觉分割的基础模型。SAM 2构建了一个数据引擎，通过用户交互改进模型和数据来提高性能，并收集了迄今为止最大的视频分割数据集。SAM 2是一个简单的Transformer架构，用于实时视频处理的流式内存，该架构使用日期训练，提供了广泛的任务上强大的性能。在本工作中，我们评估了SAM 2在更具有挑战性的VOS数据集MOSE和LVOS上的零样本性能。没有在训练集上微调，SAM 2在测试集上取得了75.79 J&F，在第六轮LSVOS挑战赛VOS赛道上排名第四。

---

## UNINEXT-Cutie_ The 1st Solution for LSVOS Challenge RVOS Track

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10129v1)

### Abstract (English)

Referring video object segmentation (RVOS) relies on natural language
expressions to segment target objects in video. In this year, LSVOS Challenge
RVOS Track replaced the origin YouTube-RVOS benchmark with MeViS. MeViS focuses
on referring the target object in a video through its motion descriptions
instead of static attributes, posing a greater challenge to RVOS task. In this
work, we integrate strengths of that leading RVOS and VOS models to build up a
simple and effective pipeline for RVOS. Firstly, We finetune the
state-of-the-art RVOS model to obtain mask sequences that are correlated with
language descriptions. Secondly, based on a reliable and high-quality key
frames, we leverage VOS model to enhance the quality and temporal consistency
of the mask results. Finally, we further improve the performance of the RVOS
model using semi-supervised learning. Our solution achieved 62.57 J&F on the
MeViS test set and ranked 1st place for 6th LSVOS Challenge RVOS Track.

### 摘要 (中文)

视频对象分割（Video Object Segmentation，简称RVOS）依赖自然语言表达来在视频中分割目标对象。本年，LVOS挑战赛的RVOS任务替换原YouTube-RVOS基准，转向了MeViS。MeViS专注于通过其运动描述来通过视频中的目标对象，这比静态属性更具挑战性。在此工作中，我们整合了领先于RVOS和VOSS模型的优势，构建了一个简单且有效的RVOS管道。首先，我们调整最先进的RVOS模型，以获得与语言描述相关的掩码序列。其次，基于可靠高质量的关键帧，我们将增强掩码结果的质量和时间一致性。最后，我们在半监督学习下进一步提高RVOS模型的表现。我们的解决方案在MeViS测试集上取得了62.57 J&F，并获得了第六届LSVOS挑战赛RVOS赛道的第一名。

---

## _R_2_-Mesh_ Reinforcement Learning Powered Mesh Reconstruction via Geometry and Appearance Refinement

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10135v1)

### Abstract (English)

Mesh reconstruction based on Neural Radiance Fields (NeRF) is popular in a
variety of applications such as computer graphics, virtual reality, and medical
imaging due to its efficiency in handling complex geometric structures and
facilitating real-time rendering. However, existing works often fail to capture
fine geometric details accurately and struggle with optimizing rendering
quality. To address these challenges, we propose a novel algorithm that
progressively generates and optimizes meshes from multi-view images. Our
approach initiates with the training of a NeRF model to establish an initial
Signed Distance Field (SDF) and a view-dependent appearance field.
Subsequently, we iteratively refine the SDF through a differentiable mesh
extraction method, continuously updating both the vertex positions and their
connectivity based on the loss from mesh differentiable rasterization, while
also optimizing the appearance representation. To further leverage
high-fidelity and detail-rich representations from NeRF, we propose an
online-learning strategy based on Upper Confidence Bound (UCB) to enhance
viewpoints by adaptively incorporating images rendered by the initial NeRF
model into the training dataset. Through extensive experiments, we demonstrate
that our method delivers highly competitive and robust performance in both mesh
rendering quality and geometric quality.

### 摘要 (中文)

基于神经光照场（NeRF）的网格重建在计算机图形学、虚拟现实和医学成像等领域因其高效处理复杂几何结构的能力而受到广泛欢迎。然而，现有的工作往往无法准确地捕捉精细的几何细节，并且在优化渲染质量方面遇到困难。针对这些挑战，我们提出了一种新颖的方法，该方法通过逐步生成并优化从多视图图像中提取的网格来逐个生成和优化网格。我们的方法首先训练了一个NeRF模型来建立初始的签名距离场（SDF）和一个视依赖的外观场。随后，我们在不同的网格提取方法的基础上迭代改进SDF，根据从可变分辨率栅格化损失更新顶点位置及其连接性，同时优化外观表示。为了进一步利用高保真和细节丰富的NeRF代表中的高级和详细信息，我们提出了基于上界置信边界（UCB）的在线学习策略，以适应性地结合由初始NeRF模型渲染的图像，以增强对原始NeRF模型所显示的视角的学习。通过广泛的实验，我们展示了我们的方法在网格渲染质量和几何质量方面的高度竞争力和鲁棒性。

---

## Multi-Scale Representation Learning for Image Restoration with State-Space Model

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10145v1)

### Abstract (English)

Image restoration endeavors to reconstruct a high-quality, detail-rich image
from a degraded counterpart, which is a pivotal process in photography and
various computer vision systems. In real-world scenarios, different types of
degradation can cause the loss of image details at various scales and degrade
image contrast. Existing methods predominantly rely on CNN and Transformer to
capture multi-scale representations. However, these methods are often limited
by the high computational complexity of Transformers and the constrained
receptive field of CNN, which hinder them from achieving superior performance
and efficiency in image restoration. To address these challenges, we propose a
novel Multi-Scale State-Space Model-based (MS-Mamba) for efficient image
restoration that enhances the capacity for multi-scale representation learning
through our proposed global and regional SSM modules. Additionally, an Adaptive
Gradient Block (AGB) and a Residual Fourier Block (RFB) are proposed to improve
the network's detail extraction capabilities by capturing gradients in various
directions and facilitating learning details in the frequency domain. Extensive
experiments on nine public benchmarks across four classic image restoration
tasks, image deraining, dehazing, denoising, and low-light enhancement,
demonstrate that our proposed method achieves new state-of-the-art performance
while maintaining low computational complexity. The source code will be
publicly available.

### 摘要 (中文)

图像修复旨在从降级的副本中重建高质量、细节丰富的图像，这是摄影和各种计算机视觉系统中的关键过程。在现实世界的各种场景中，不同的类型失真会导致不同尺度上的图像细节损失和图像对比度降低。现有方法主要依赖于CNN和Transformer来捕获多尺度表示。然而，这些方法往往受限于Transformer的高度计算复杂性和CNN的受限 receptive field，这限制了它们在图像修复方面取得优异性能和效率的能力。针对这些挑战，我们提出了一个多尺度状态空间模型（MS-Mamba）以高效地进行图像修复，通过我们的提出的全局和区域SSM模块增强多尺度表示学习能力。此外，还提出了一种适应性梯度块（AGB）和一个余弦频域阻塞块（RFB），以通过捕捉各种方向的梯度改进网络的细节提取能力，并促进在频率域内学习细节的学习。在来自四个经典图像修复任务的九个公共基准数据集上进行了广泛的实验，包括图像去雾、去模糊、降噪和低光增强，证明了我们提出的方案在保持较低计算复杂性的同时取得了新的领先水平。源代码将在公开可用。

---

## Structure-preserving Image Translation for Depth Estimation in Colonoscopy Video

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10153v1)

### Abstract (English)

Monocular depth estimation in colonoscopy video aims to overcome the unusual
lighting properties of the colonoscopic environment. One of the major
challenges in this area is the domain gap between annotated but unrealistic
synthetic data and unannotated but realistic clinical data. Previous attempts
to bridge this domain gap directly target the depth estimation task itself. We
propose a general pipeline of structure-preserving synthetic-to-real (sim2real)
image translation (producing a modified version of the input image) to retain
depth geometry through the translation process. This allows us to generate
large quantities of realistic-looking synthetic images for supervised depth
estimation with improved generalization to the clinical domain. We also propose
a dataset of hand-picked sequences from clinical colonoscopies to improve the
image translation process. We demonstrate the simultaneous realism of the
translated images and preservation of depth maps via the performance of
downstream depth estimation on various datasets.

### 摘要 (中文)

单目深度估计在结肠镜视频中旨在克服结肠内环境的不寻常光照特性。在这个领域的主要挑战之一是标注但不现实合成数据与未标注但真实临床数据之间的域差距。直接针对深度估计任务本身，以前的努力尝试在此鸿沟之间建立桥梁。我们提出了一种结构保留的合成到现实（sim2real）图像翻译通用管道（通过翻译过程保持深度几何），从而生成大量的具有改进的一般化临床领域的深度估计的真实看起来的合成图像。我们也提出了一个从临床结肠镜检查中手选序列的数据集来改善图像翻译过程。我们在各种数据集上同时展示了翻译图像的真实性和深度地图的保真度，通过下游深度估计性能的表现。

---

## NeuFlow v2_ High-Efficiency Optical Flow Estimation on Edge Devices

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10161v1)

### Abstract (English)

Real-time high-accuracy optical flow estimation is crucial for various
real-world applications. While recent learning-based optical flow methods have
achieved high accuracy, they often come with significant computational costs.
In this paper, we propose a highly efficient optical flow method that balances
high accuracy with reduced computational demands. Building upon NeuFlow v1, we
introduce new components including a much more light-weight backbone and a fast
refinement module. Both these modules help in keeping the computational demands
light while providing close to state of the art accuracy. Compares to other
state of the art methods, our model achieves a 10x-70x speedup while
maintaining comparable performance on both synthetic and real-world data. It is
capable of running at over 20 FPS on 512x384 resolution images on a Jetson Orin
Nano. The full training and evaluation code is available at
https://github.com/neufieldrobotics/NeuFlow_v2.

### 摘要 (中文)

实时高精度光学流估计对于各种实际应用至关重要。虽然近期基于学习的光学流方法已经取得了很高的准确性，但它们往往伴随着显著的计算成本。本论文提出了一种高效光学流方法，该方法在保持较高准确性和减少计算需求之间实现了平衡。基于NeuFlow v1，我们引入了新的组件，包括更轻量化的后端和快速细化模块。这两个模块帮助在保持计算需求较低的同时提供接近目前最先进的精确度。与其他现有技术相比，我们的模型在保持相同比例性能的同时，在合成数据和真实世界数据上都获得了10倍到70倍的速度提升。它可以在Jetson Orin Nano上以每秒超过20帧的速度运行512x384分辨率图像的训练和评估代码可在此处找到：https://github.com/neufieldrobotics/NeuFlow_v2

---

## Assessment of Spectral based Solutions for the Detection of Floating Marine Debris

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10187v1)

### Abstract (English)

Typically, the detection of marine debris relies on in-situ campaigns that
are characterized by huge human effort and limited spatial coverage. Following
the need of a rapid solution for the detection of floating plastic, methods
based on remote sensing data have been proposed recently. Their main limitation
is represented by the lack of a general reference for evaluating performance.
Recently, the Marine Debris Archive (MARIDA) has been released as a standard
dataset to develop and evaluate Machine Learning (ML) algorithms for detection
of Marine Plastic Debris. The MARIDA dataset has been created for simplifying
the comparison between detection solutions with the aim of stimulating the
research in the field of marine environment preservation. In this work, an
assessment of spectral based solutions is proposed by evaluating performance on
MARIDA dataset. The outcome highlights the need of precise reference for fair
evaluation.

### 摘要 (中文)

通常，海洋垃圾的检测依赖于现场调查，这些调查由大量的人力投入和有限的空间覆盖。随着漂浮塑料检测快速解决方案的需求日益迫切，基于遥感数据的方法最近被提出。其主要限制是缺乏对性能进行一般评价的通用参考。最近，已发布了一个标准数据集——海洋垃圾档案（MARIDA），用于开发和评估用于检测海洋塑料垃圾的机器学习（ML）算法。MARIDA数据集旨在简化比较不同检测解决方案的目标，以促进海洋环境保护研究领域的研究。在本文中，提出了通过评估MARIDA数据集上的光谱解决方案来评估性能的一种方法。结果表明，在公平评估方面需要精确的参考。

---

## SANER_ Annotation-free Societal Attribute Neutralizer for Debiasing CLIP

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10202v1)

### Abstract (English)

Large-scale vision-language models, such as CLIP, are known to contain
harmful societal bias regarding protected attributes (e.g., gender and age). In
this paper, we aim to address the problems of societal bias in CLIP. Although
previous studies have proposed to debias societal bias through adversarial
learning or test-time projecting, our comprehensive study of these works
identifies two critical limitations: 1) loss of attribute information when it
is explicitly disclosed in the input and 2) use of the attribute annotations
during debiasing process. To mitigate societal bias in CLIP and overcome these
limitations simultaneously, we introduce a simple-yet-effective debiasing
method called SANER (societal attribute neutralizer) that eliminates attribute
information from CLIP text features only of attribute-neutral descriptions.
Experimental results show that SANER, which does not require attribute
annotations and preserves original information for attribute-specific
descriptions, demonstrates superior debiasing ability than the existing
methods.

### 摘要 (中文)

大规模的视觉语言模型，如CLIP，已知包含有害的社会属性偏见（例如性别和年龄）。在本文中，我们旨在解决CLIP中的社会属性偏见问题。虽然之前的研究提出了通过对抗学习或测试时间投影来消除社会属性偏见的方法，但我们的全面研究发现了两个关键限制：1）当输入中明确披露时，它会丢失属性信息；2）在消除社会属性偏见的过程中使用属性注释。为了同时克服这些限制，并且有效地消除CLIP文本特征中的属性信息，我们引入了一个名为SANER（社会属性中立器）的有效去偏方法，该方法仅从属性中立描述的CLIP文本特征中删除属性信息。实验结果表明，在不依赖属性注释并保留属性特定描述原始信息的情况下，SANER比现有方法表现出更优的去偏能力。

---

## Ensemble Prediction via Covariate-dependent Stacking

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09755v1)

### Abstract (English)

This paper presents a novel approach to ensemble prediction called
"Covariate-dependent Stacking" (CDST). Unlike traditional stacking methods,
CDST allows model weights to vary flexibly as a function of covariates, thereby
enhancing predictive performance in complex scenarios. We formulate the
covariate-dependent weights through combinations of basis functions, estimate
them by optimizing cross-validation, and develop an Expectation-Maximization
algorithm, ensuring computational efficiency. To analyze the theoretical
properties, we establish an oracle inequality regarding the expected loss to be
minimized for estimating model weights. Through comprehensive simulation
studies and an application to large-scale land price prediction, we demonstrate
that CDST consistently outperforms conventional model averaging methods,
particularly on datasets where some models fail to capture the underlying
complexity. Our findings suggest that CDST is especially valuable for, but not
limited to, spatio-temporal prediction problems, offering a powerful tool for
researchers and practitioners in various fields of data analysis.

### 摘要 (中文)

这篇论文提出了一种新的组合预测方法叫做“基于变量依赖的堆叠”（CDST）。与传统的堆叠方法不同，CDST允许模型权重根据变量变化灵活地调整，从而在复杂场景中增强预测性能。我们通过基函数的组合来形式化变量依赖的权重，并通过交叉验证优化估计它们，在计算上保持高效性。为了分析理论属性，我们建立了关于最小损失估计模型权重的逻辑不等式。通过全面的模拟研究和应用到大规模土地价格预测中的综合实例，我们证明了CDST始终优于常规的模型平均方法，特别是在某些模型无法捕捉底层复杂性的数据集上。我们的发现表明，CDST特别适用于空间-时间预测问题，对数据分析领域内的研究人员和实践者提供了强大的工具。

---

## Predicting travel demand of a bike sharing system using graph convolutional neural networks

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09317v1)

### Abstract (English)

Public transportation systems play a crucial role in daily commutes, business
operations, and leisure activities, emphasizing the need for effective
management to meet public demands. One approach to achieve this goal is by
predicting demand at the station level. Bike-sharing systems, as a form of
transit service, contribute to the reduction of air and noise pollution, as
well as traffic congestion. This study focuses on predicting travel demand
within a bike-sharing system. A novel hybrid deep learning model called the
gate graph convolutional neural network is introduced. This model enables
prediction of the travel demand at station level. By integrating trajectory
data, weather data, access data, and leveraging gate graph convolution
networks, the accuracy of travel demand forecasting is significantly improved.
Chicago City bike-sharing system is chosen as the case study. In this
investigation, the proposed model is compared to the base models used in
previous literature to evaluate their performance, demonstrating that the main
model exhibits better performance than the base models. By utilizing this
framework, transportation planners can make informed decisions on resource
allocation and rebalancing management.

### 摘要 (中文)

公共交通系统在日常通勤、商业运营和休闲活动等方面发挥着至关重要的作用，强调了有效管理以满足公众需求的必要性。实现这一目标的一种方法是预测站级需求。自行车共享服务作为交通服务的一种形式，有助于减少空气和噪音污染，以及缓解交通拥堵。本研究侧重于预测自行车共享系统的旅行需求。引入了一种名为门网络卷积神经网络的新混合深度学习模型。该模型允许对站点级别进行旅行需求预测。通过集成轨迹数据、天气数据、访问数据，并利用门网络卷积神经网络，旅行需求预测的准确性显著提高。芝加哥市自行车共享系统的案例研究被选作研究对象。在此调查中，提出的模型与先前文献中使用的基准模型进行了比较，以评估其性能，证明主要模型的表现优于基准模型。利用此框架，运输规划者可以就资源分配和再平衡管理做出明智的决策。

---

## A Probabilistic Framework for Adapting to Changing and Recurring Concepts in Data Streams

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09324v1)

### Abstract (English)

The distribution of streaming data often changes over time as conditions
change, a phenomenon known as concept drift. Only a subset of previous
experience, collected in similar conditions, is relevant to learning an
accurate classifier for current data. Learning from irrelevant experience
describing a different concept can degrade performance. A system learning from
streaming data must identify which recent experience is irrelevant when
conditions change and which past experience is relevant when concepts reoccur,
\textit{e.g.,} when weather events or financial patterns repeat. Existing
streaming approaches either do not consider experience to change in relevance
over time and thus cannot handle concept drift, or only consider the recency of
experience and thus cannot handle recurring concepts, or only sparsely evaluate
relevance and thus fail when concept drift is missed. To enable learning in
changing conditions, we propose SELeCT, a probabilistic method for continuously
evaluating the relevance of past experience. SELeCT maintains a distinct
internal state for each concept, representing relevant experience with a unique
classifier. We propose a Bayesian algorithm for estimating state relevance,
combining the likelihood of drawing recent observations from a given state with
a transition pattern prior based on the system's current state.

### 摘要 (中文)

流数据的分布往往随着时间的变化而变化，这就是所谓的概念漂移现象。只有与当前数据收集相似条件下的少量以前经验是学习准确分类器相关的。从无关经验描述不同概念的学习性能会下降。一个在线学习系统必须识别当条件改变时最近的经验是否不相关以及在概念重出现时过去的经验是否相关，例如，当天气事件或金融模式重复时。现有的流式处理方法要么不考虑时间上的相关性随时间而变化，因此无法处理概念漂移，要么只考虑经历的近期性和概念的重现性，因此无法处理循环的概念，或者仅稀疏地评估相关性，因此当概念漂移被忽略时就会失败。为了能够在不断变化的条件下进行学习，我们提出了SELeCT，这是一种连续评估过去经验的相关性的概率方法。SELeCT维护每个概念的独特内部状态，代表有关经验使用独特的分类器。我们提出了一种基于系统的当前状态的过渡模式前的概率算法来估计状态的相关性，结合了从给定状态下抽取最近观察的似然性和基于系统当前状态的转换模式前的概率。

---

## Improvement of Bayesian PINN Training Convergence in Solving Multi-scale PDEs with Noise

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09340v1)

### Abstract (English)

Bayesian Physics Informed Neural Networks (BPINN) have received considerable
attention for inferring differential equations' system states and physical
parameters according to noisy observations. However, in practice, Hamiltonian
Monte Carlo (HMC) used to estimate the internal parameters of BPINN often
encounters troubles, including poor performance and awful convergence for a
given step size used to adjust the momentum of those parameters. To improve the
efficacy of HMC convergence for the BPINN method and extend its application
scope to multi-scale partial differential equations (PDE), we developed a
robust multi-scale Bayesian PINN (dubbed MBPINN) method by integrating
multi-scale deep neural networks (MscaleDNN) and Bayesian inference. In this
newly proposed MBPINN method, we reframe HMC with Stochastic Gradient Descent
(SGD) to ensure the most ``likely'' estimation is always provided, and we
configure its solver as a Fourier feature mapping-induced MscaleDNN. The MBPINN
method offers several key advantages: (1) it is more robust than HMC, (2) it
incurs less computational cost than HMC, and (3) it is more flexible for
complex problems. We demonstrate the applicability and performance of the
proposed method through general Poisson and multi-scale elliptic problems in
one- to three-dimensional spaces. Our findings indicate that the proposed
method can avoid HMC failures and provide valid results. Additionally, our
method can handle complex PDE and produce comparable results for general PDE.
These findings suggest that our proposed approach has excellent potential for
physics-informed machine learning for parameter estimation and solution
recovery in the case of ill-posed problems.

### 摘要 (中文)

贝叶斯物理启发神经网络（BPINN）因其根据噪声观测推断差分方程系统状态和物理参数而受到广泛关注。然而，在实践中，用于估计BPINN内部参数的哈密顿马尔可夫随机过程（HMC）经常遇到困难，包括在调整那些参数的动量时性能不佳以及收敛很差的问题。为了改善BPINN方法HMC收敛的有效性，并将其应用范围扩展到多尺度偏微分方程（PDE），我们通过集成多尺度深度神经网络（MScaleDNN）和贝叶斯推理开发了一种稳健的多尺度贝叶斯PINN（MBPINN）方法。在本文提出的新MBPINN方法中，我们将HMC重新定义为Stochastic Gradient Descent（SGD）以确保总是提供最“可能”的估计，并配置其求解器为四维特征映射诱导的MScaleDNN。新提出的MBPINN方法具有几个关键优势：（1）它比HMC更健壮；（2）它的计算成本低于HMC；（3）它更适合复杂问题。我们在一至三维空间的一般泊松和多尺度椭圆问题上展示了该方法的适用性和性能。我们的发现表明，所提出的方法可以避免HMC失败并提供有效结果。此外，我们的方法还可以处理复杂的PDE，并且对于一般PDE的结果相当一致。这些发现表明，我们提出的这种方法对解决不完全问题中的物理启发机器学习有很好的潜在潜力来估计参数和恢复解。

---

## Clustering and Alignment_ Understanding the Training Dynamics in Modular Addition

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09414v1)

### Abstract (English)

Recent studies have revealed that neural networks learn interpretable
algorithms for many simple problems. However, little is known about how these
algorithms emerge during training. In this article, we study the training
dynamics of a simplified transformer with 2-dimensional embeddings on the
problem of modular addition. We observe that embedding vectors tend to organize
into two types of structures: grids and circles. We study these structures and
explain their emergence as a result of two simple tendencies exhibited by pairs
of embeddings: clustering and alignment. We propose explicit formulae for these
tendencies as interaction forces between different pairs of embeddings. To show
that our formulae can fully account for the emergence of these structures, we
construct an equivalent particle simulation where we find that identical
structures emerge. We use our insights to discuss the role of weight decay and
reveal a new mechanism that links regularization and training dynamics. We also
release an interactive demo to support our findings:
https://modular-addition.vercel.app/.

### 摘要 (中文)

最近的研究揭示了神经网络在许多简单问题上学习可解释的算法。然而，关于这些算法是如何在训练过程中出现的还知之甚少。本文研究了一种简化Transformer的问题——模数加法，观察到嵌入向量倾向于组织成两种结构：网格和圆圈。我们研究了这些结构，并解释了它们的出现是由于两个简单的趋势由两对嵌入表示所表现出来的集群和匹配。我们提出了这些趋势之间的明确方程作为不同对嵌入之间的相互作用力。为了证明我们的方程式能够完全描述这些结构的出现，我们构建了一个等价粒子模拟，在其中发现相同结构出现了。利用我们的见解来讨论权重衰减的作用，并揭示了新的机制，即正则化与训练动力学之间的联系。我们也发布了交互式演示支持我们的发现：https://modular-addition.vercel.app/.

---

## Reefknot_ A Comprehensive Benchmark for Relation Hallucination Evaluation_ Analysis and Mitigation in Multimodal Large Language Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09429v1)

### Abstract (English)

Hallucination issues persistently plagued current multimodal large language
models (MLLMs). While existing research primarily focuses on object-level or
attribute-level hallucinations, sidelining the more sophisticated relation
hallucinations that necessitate advanced reasoning abilities from MLLMs.
Besides, recent benchmarks regarding relation hallucinations lack in-depth
evaluation and effective mitigation. Moreover, their datasets are typically
derived from a systematic annotation process, which could introduce inherent
biases due to the predefined process. To handle the aforementioned challenges,
we introduce Reefknot, a comprehensive benchmark specifically targeting
relation hallucinations, consisting of over 20,000 samples derived from
real-world scenarios. Specifically, we first provide a systematic definition of
relation hallucinations, integrating perspectives from perceptive and cognitive
domains. Furthermore, we construct the relation-based corpus utilizing the
representative scene graph dataset Visual Genome (VG), from which semantic
triplets follow real-world distributions. Our comparative evaluation across
three distinct tasks revealed a substantial shortcoming in the capabilities of
current MLLMs to mitigate relation hallucinations. Finally, we advance a novel
confidence-based mitigation strategy tailored to tackle the relation
hallucinations problem. Across three datasets, including Reefknot, we observed
an average reduction of 9.75% in the hallucination rate. We believe our paper
sheds valuable insights into achieving trustworthy multimodal intelligence. Our
dataset and code will be released upon paper acceptance.

### 摘要 (中文)

当前的多模态大型语言模型（MLMM）一直存在幻觉问题。虽然现有的研究主要集中在对象级或属性级幻觉，而忽视了更高级别的关系幻觉，即需要从MMLMM中获得高级推理能力。此外，关于关系幻觉的相关评估缺乏深入的评估和有效的缓解措施。而且他们的数据通常来源于系统性标注过程，这可能会由于预先定义的过程引入内在偏见。为了应对上述挑战，我们引入了Reefknot，这是一个专门针对关系幻觉的全面基准，包括超过20,000个来自现实世界场景的真实样本。具体来说，我们将首先提供一种系统的幻觉定义，整合感知和认知领域的视角。其次，我们将利用代表性的图象生成网络（VG）的数据集来构建基于关系的语料库，其中实境分布的语义三元组遵循真实世界的分布。我们在三个不同的任务上进行了比较评估，发现目前的MMLMM在抑制关系幻觉方面的能力存在显著不足。最后，我们提出了一个全新的基于信心的缓解策略，该策略旨在解决关系幻觉问题。在三个包含Reefknot在内的数据集中，我们观察到平均降低了9.75％的幻觉率。我们认为我们的论文提供了实现可信多模态智能的重要见解。我们的数据集和代码将在论文接受后发布。

---

## Reparameterized Multi-Resolution Convolutions for Long Sequence Modelling

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09453v1)

### Abstract (English)

Global convolutions have shown increasing promise as powerful general-purpose
sequence models. However, training long convolutions is challenging, and kernel
parameterizations must be able to learn long-range dependencies without
overfitting. This work introduces reparameterized multi-resolution convolutions
($\texttt{MRConv}$), a novel approach to parameterizing global convolutional
kernels for long-sequence modelling. By leveraging multi-resolution
convolutions, incorporating structural reparameterization and introducing
learnable kernel decay, $\texttt{MRConv}$ learns expressive long-range kernels
that perform well across various data modalities. Our experiments demonstrate
state-of-the-art performance on the Long Range Arena, Sequential CIFAR, and
Speech Commands tasks among convolution models and linear-time transformers.
Moreover, we report improved performance on ImageNet classification by
replacing 2D convolutions with 1D $\texttt{MRConv}$ layers.

### 摘要 (中文)

全球卷积展现出强大的通用目的序列模型的潜力，但训练长卷积是具有挑战性的，并且需要能够学习长期依赖关系而不会过拟合的卷积参数化。本工作引入了多分辨率卷积（$\texttt{MRConv}$）作为全局卷积核的参数化方法的一种新型方法，它利用多分辨率卷积、结构再参数化和引入可学习的卷积衰减，以学习表达性强的长期依赖关系的长序列模型。我们的实验展示了在各种数据模态中卷积模型和线性时间变换器上取得最佳性能。此外，我们通过替换二维卷积层中的2D卷积层使用1D $\texttt{MRConv}$ 层来改进ImageNet分类性能。

---

## Ancestral Reinforcement Learning_ Unifying Zeroth-Order Optimization and Genetic Algorithms for Reinforcement Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09493v1)

### Abstract (English)

Reinforcement Learning (RL) offers a fundamental framework for discovering
optimal action strategies through interactions within unknown environments.
Recent advancement have shown that the performance and applicability of RL can
significantly be enhanced by exploiting a population of agents in various ways.
Zeroth-Order Optimization (ZOO) leverages an agent population to estimate the
gradient of the objective function, enabling robust policy refinement even in
non-differentiable scenarios. As another application, Genetic Algorithms (GA)
boosts the exploration of policy landscapes by mutational generation of policy
diversity in an agent population and its refinement by selection. A natural
question is whether we can have the best of two worlds that the agent
population can have. In this work, we propose Ancestral Reinforcement Learning
(ARL), which synergistically combines the robust gradient estimation of ZOO
with the exploratory power of GA. The key idea in ARL is that each agent within
a population infers gradient by exploiting the history of its ancestors, i.e.,
the ancestor population in the past, while maintaining the diversity of
policies in the current population as in GA. We also theoretically reveal that
the populational search in ARL implicitly induces the KL-regularization of the
objective function, resulting in the enhanced exploration. Our results extend
the applicability of populational algorithms for RL.

### 摘要 (中文)

强化学习（RL）提供了一种发现通过在未知环境中与之交互来发现最优行动策略的根本框架。最近的进步已经表明，通过利用代理人口中的各种方式，RL的表现和适用性可以显著增强。零阶优化（ZOO）利用代理人口估计目标函数的梯度，即使是在不可导的情况下也能进行稳健的政策完善。作为另一个应用，遗传算法（GA）通过基因变异产生政策多样性，并通过选择对其进行改进。一个自然的问题是，我们是否能拥有两个世界，即代理人口能够拥有。在这项工作中，我们提出了一种协同结合了ZOO中稳健梯度估计能力与GA探索力的Ancestral Reinforcement Learning（ARL）。ARL的关键思想是，在一个人口内每个代理都利用祖先的历史推断梯度，即祖先人口的历史，同时保持当前人口中政策的多样性。我们还理论上揭示了ARL中的人口搜索隐含地诱导了目标函数的KL正则化，导致了更佳的探索。我们的结果扩展了对RL的普及型算法的应用范围。

---

## Directed Exploration in Reinforcement Learning from Linear Temporal Logic

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09495v1)

### Abstract (English)

Linear temporal logic (LTL) is a powerful language for task specification in
reinforcement learning, as it allows describing objectives beyond the
expressivity of conventional discounted return formulations. Nonetheless,
recent works have shown that LTL formulas can be translated into a variable
rewarding and discounting scheme, whose optimization produces a policy
maximizing a lower bound on the probability of formula satisfaction. However,
the synthesized reward signal remains fundamentally sparse, making exploration
challenging. We aim to overcome this limitation, which can prevent current
algorithms from scaling beyond low-dimensional, short-horizon problems. We show
how better exploration can be achieved by further leveraging the LTL
specification and casting its corresponding Limit Deterministic B\"uchi
Automaton (LDBA) as a Markov reward process, thus enabling a form of high-level
value estimation. By taking a Bayesian perspective over LDBA dynamics and
proposing a suitable prior distribution, we show that the values estimated
through this procedure can be treated as a shaping potential and mapped to
informative intrinsic rewards. Empirically, we demonstrate applications of our
method from tabular settings to high-dimensional continuous systems, which have
so far represented a significant challenge for LTL-based reinforcement learning
algorithms.

### 摘要 (中文)

线性时序逻辑(LTL)是强化学习中任务指定的强大语言，因为它允许描述超越常规折扣回报形式的更广泛的客观目标。然而，最近的工作表明，LTL公式可以转换为一个变量奖励和折扣机制，其优化产生了一个政策，它最大化了满足条件的概率下界。然而，合成的奖励信号仍然是基本稀疏的，这使得探索变得具有挑战性。我们的目标是克服这一限制，以防止当前算法无法在低维度、短期问题上扩展到更大的维度或较长的时间间隔。我们展示了如何通过进一步利用LTL说明并将其对应的有限决定式伯努利自动机（LDBA）作为马尔可夫奖励过程，从而实现高水平的价值估计。通过对LDBA动力学的Bayesian视角进行研究，并提出合适的先验分布，我们显示，通过这种方法估算得到的价值可以通过塑造潜力来映射成有用的内在奖励。实证研究表明，在表征LTL基础强化学习算法面临的巨大挑战方面，从表格设置到高维连续系统的应用已经得到了演示。

---

## Fine-gained air quality inference based on low-quality sensing data using self-supervised learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09526v1)

### Abstract (English)

Fine-grained air quality (AQ) mapping is made possible by the proliferation
of cheap AQ micro-stations (MSs). However, their measurements are often
inaccurate and sensitive to local disturbances, in contrast to standardized
stations (SSs) that provide accurate readings but fall short in number. To
simultaneously address the issues of low data quality (MSs) and high label
sparsity (SSs), a multi-task spatio-temporal network (MTSTN) is proposed, which
employs self-supervised learning to utilize massive unlabeled data, aided by
seasonal and trend decomposition of MS data offering reliable information as
features. The MTSTN is applied to infer NO$_2$, O$_3$ and PM$_{2.5}$
concentrations in a 250 km$^2$ area in Chengdu, China, at a resolution of
500m$\times$500m$\times$1hr. Data from 55 SSs and 323 MSs were used, along with
meteorological, traffic, geographic and timestamp data as features. The MTSTN
excels in accuracy compared to several benchmarks, and its performance is
greatly enhanced by utilizing low-quality MS data. A series of ablation and
pressure tests demonstrate the results' robustness and interpretability,
showcasing the MTSTN's practical value for accurate and affordable AQ
inference.

### 摘要 (中文)

通过大量廉价的空气质量微站（MS）的普及，实现了精细空气质量（AQ）地图的可能。然而，他们的测量往往不准确且对当地干扰敏感，而标准化站点（SS）提供了准确的结果但数量有限。因此，同时解决低数据质量和高标签稀疏性的问题，提出了一种多任务空间-时间网络（MTSTN），它利用大量的无标签数据进行自我监督学习，并借助季节性和趋势分解提供的可靠信息作为特征。该MTSTN被应用于中国成都250平方公里区域的55个标准站点和323个微站，在分辨率下为500米×500米×1小时。使用了气象、交通、地理和时间戳等数据作为特征。MTSTN在与几个基准相比具有更高的精度，并通过使用低质量的MS数据显著提高了性能。一系列去除和压力测试证明了结果的稳健性和可解释性，展示了MTSTN用于准确和实惠的AQ推断的实际价值。

---

## Seamless Integration_ Sampling Strategies in Federated Learning Systems

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09545v1)

### Abstract (English)

Federated Learning (FL) represents a paradigm shift in the field of machine
learning, offering an approach for a decentralized training of models across a
multitude of devices while maintaining the privacy of local data. However, the
dynamic nature of FL systems, characterized by the ongoing incorporation of new
clients with potentially diverse data distributions and computational
capabilities, poses a significant challenge to the stability and efficiency of
these distributed learning networks. The seamless integration of new clients is
imperative to sustain and enhance the performance and robustness of FL systems.
This paper looks into the complexities of integrating new clients into existing
FL systems and explores how data heterogeneity and varying data distribution
(not independent and identically distributed) among them can affect model
training, system efficiency, scalability and stability. Despite these
challenges, the integration of new clients into FL systems presents
opportunities to enhance data diversity, improve learning performance, and
leverage distributed computational power. In contrast to other fields of
application such as the distributed optimization of word predictions on Gboard
(where federated learning once originated), there are usually only a few
clients in the production environment, which is why information from each new
client becomes all the more valuable. This paper outlines strategies for
effective client selection strategies and solutions for ensuring system
scalability and stability. Using the example of images from optical quality
inspection, it offers insights into practical approaches. In conclusion, this
paper proposes that addressing the challenges presented by new client
integration is crucial to the advancement and efficiency of distributed
learning networks, thus paving the way for the adoption of Federated Learning
in production environments.

### 摘要 (中文)

联邦学习（Federated Learning，简称FL）是机器学习领域的一个范式转变，它提供了一种在多台设备上分布式训练模型的方法，并且保持了本地数据的隐私。然而，联邦学习系统的动态特性，即不断接纳具有可能不同数据分布和计算能力的新客户端，对这些分布式学习网络的稳定性和效率构成了一个巨大的挑战。

无缝地集成新客户是维持和提升FL系统性能的关键。本文探讨了如何处理数据异质性以及它们中的不同数据分布（非独立同分布）如何影响模型训练、系统效率、可扩展性和稳定性。尽管存在这些挑战，但新客户的集成到FL系统中提供了改善数据多样性的机会，提高学习性能的机会，以及利用分布式计算能力的机会。与其他如Gboard分布式优化预测词的领域相比，生产环境中通常只有少数客户端，因此每新的客户端信息都变得更加宝贵。本文概述了有效选择策略和确保系统可扩展性和稳定性的解决方案。以光学质量检查图像为例，提出了实用方法的见解。总的来说，本论文提出，解决新客户集成面临的挑战对于推动分布式学习网络的发展和效率至关重要，从而为生产环境铺平道路。

---

## sTransformer_ A Modular Approach for Extracting Inter-Sequential and Temporal Information for Time-Series Forecasting

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09723v1)

### Abstract (English)

In recent years, numerous Transformer-based models have been applied to
long-term time-series forecasting (LTSF) tasks. However, recent studies with
linear models have questioned their effectiveness, demonstrating that simple
linear layers can outperform sophisticated Transformer-based models. In this
work, we review and categorize existing Transformer-based models into two main
types: (1) modifications to the model structure and (2) modifications to the
input data. The former offers scalability but falls short in capturing
inter-sequential information, while the latter preprocesses time-series data
but is challenging to use as a scalable module. We propose
$\textbf{sTransformer}$, which introduces the Sequence and Temporal
Convolutional Network (STCN) to fully capture both sequential and temporal
information. Additionally, we introduce a Sequence-guided Mask Attention
mechanism to capture global feature information. Our approach ensures the
capture of inter-sequential information while maintaining module scalability.
We compare our model with linear models and existing forecasting models on
long-term time-series forecasting, achieving new state-of-the-art results. We
also conducted experiments on other time-series tasks, achieving strong
performance. These demonstrate that Transformer-based structures remain
effective and our model can serve as a viable baseline for time-series tasks.

### 摘要 (中文)

近年来，许多基于Transformer的模型被应用于长期时间序列预测（LTSF）任务。然而，最近的研究质疑了线性模型的有效性，展示了简单的线性层可以超越复杂度更高的Transformer基于模型。在本文中，我们回顾并分类现有的Transformer基于模型分为两大类：(1)对模型结构的修改和(2)对输入数据的修改。前者提供了可扩展性但无法捕捉相互序列信息，而后者预处理时间序列数据但难以作为可扩展模块使用。我们提出了sTransformer，它引入了Sequence和Temporal Convolutional Network(STCN)来完全捕获两者之间的序列和时间序列信息。此外，我们还提出了一种Sequence-guided Mask Attention机制以捕捉全局特征信息。我们的方法确保了序列信息的捕获同时保持模块的可扩展性。我们将模型与线性模型以及现有预测模型进行了比较，在长期时间序列预测任务上取得了新的顶峰状态。我们也对其他时间序列任务进行了实验，取得了强大的表现。这些表明Transformer架构仍然有效，我们的模型可以成为时间序列任务的可行基准。

---

## Sequential Federated Learning in Hierarchical Architecture on Non-IID Datasets

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09762v1)

### Abstract (English)

In a real federated learning (FL) system, communication overhead for passing
model parameters between the clients and the parameter server (PS) is often a
bottleneck. Hierarchical federated learning (HFL) that poses multiple edge
servers (ESs) between clients and the PS can partially alleviate communication
pressure but still needs the aggregation of model parameters from multiple ESs
at the PS. To further reduce communication overhead, we bring sequential FL
(SFL) into HFL for the first time, which removes the central PS and enables the
model training to be completed only through passing the global model between
two adjacent ESs for each iteration, and propose a novel algorithm adaptive to
such a combinational framework, referred to as Fed-CHS. Convergence results are
derived for strongly convex and non-convex loss functions under various data
heterogeneity setups, which show comparable convergence performance with the
algorithms for HFL or SFL solely. Experimental results provide evidence of the
superiority of our proposed Fed-CHS on both communication overhead saving and
test accuracy over baseline methods.

### 摘要 (中文)

在真正的联邦学习（FL）系统中，通过客户端和参数服务器（PS）之间的模型参数传递的通信开销往往是一个瓶颈。采用多边缘服务器（ESs）的层次联邦学习（HFL）可以部分缓解通信压力，但仍需要对多个ES的模型参数进行聚合在PS上。为了进一步降低通信开销，我们首次引入了顺序联邦学习（SFL），它移除了中央PS，使得每个迭代只需通过在相邻ES之间传输全局模型来完成模型训练，并提出了一种适应这种组合框架的新算法，称为Fed-CHS。在各种数据异质性设置下，对于强凸函数和非凸函数的弱凸函数损失函数，导出了收敛结果，显示了与仅使用HFL或SFL的算法相比，在节省通信开销和测试精度方面的相同性能。实验结果提供了在通信开销节省和测试准确性上的优势优于基线方法的证据。

---

## Structure-enhanced Contrastive Learning for Graph Clustering

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09790v1)

### Abstract (English)

Graph clustering is a crucial task in network analysis with widespread
applications, focusing on partitioning nodes into distinct groups with stronger
intra-group connections than inter-group ones. Recently, contrastive learning
has achieved significant progress in graph clustering. However, most methods
suffer from the following issues: 1) an over-reliance on meticulously designed
data augmentation strategies, which can undermine the potential of contrastive
learning. 2) overlooking cluster-oriented structural information, particularly
the higher-order cluster(community) structure information, which could unveil
the mesoscopic cluster structure information of the network. In this study,
Structure-enhanced Contrastive Learning (SECL) is introduced to addresses these
issues by leveraging inherent network structures. SECL utilizes a cross-view
contrastive learning mechanism to enhance node embeddings without elaborate
data augmentations, a structural contrastive learning module for ensuring
structural consistency, and a modularity maximization strategy for harnessing
clustering-oriented information. This comprehensive approach results in robust
node representations that greatly enhance clustering performance. Extensive
experiments on six datasets confirm SECL's superiority over current
state-of-the-art methods, indicating a substantial improvement in the domain of
graph clustering.

### 摘要 (中文)



---

## Enhance Modality Robustness in Text-Centric Multimodal Alignment with Adversarial Prompting

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09798v1)

### Abstract (English)

Converting different modalities into generalized text, which then serves as
input prompts for large language models (LLMs), is a common approach for
aligning multimodal models, particularly when pairwise data is limited.
Text-centric alignment method leverages the unique properties of text as a
modality space, transforming diverse inputs into a unified textual
representation, thereby enabling downstream models to effectively interpret
various modal inputs. This study evaluates the quality and robustness of
multimodal representations in the face of noise imperfections, dynamic input
order permutations, and missing modalities, revealing that current text-centric
alignment methods can compromise downstream robustness. To address this issue,
we propose a new text-centric adversarial training approach that significantly
enhances robustness compared to traditional robust training methods and
pre-trained multimodal foundation models. Our findings underscore the potential
of this approach to improve the robustness and adaptability of multimodal
representations, offering a promising solution for dynamic and real-world
applications.

### 摘要 (中文)

将不同的模态转换成通用文本，然后作为大型语言模型（LLM）的输入提示，是多模态模型对齐的常见方法。特别是当有限的数据仅限于一对数据时，这种技术就显得尤为重要。

文本为中心的方法利用了文本这一模态空间的独特属性，将多样化的输入转化为统一的文本表示，从而使得下游模型能够有效地解释各种模态输入。本研究评估了在面对噪声缺陷、动态输入顺序以及缺失模态的情况下，多模态表示的质量和鲁棒性，并揭示出当前基于文本中心的对齐方法可能会降低下游的鲁棒性。为了应对这个问题，我们提出了一个新的基于文本中心对抗训练的方法，其鲁棒性显著优于传统的鲁棒训练方法和预训练的多模态基础模型。我们的发现强调了这种方法改进多模态表示质量和适应性的潜力，为动态和现实世界应用提供了一个有希望的解决方案。

---

## GINO-Q_ Learning an Asymptotically Optimal Index Policy for Restless Multi-armed Bandits

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09882v1)

### Abstract (English)

The restless multi-armed bandit (RMAB) framework is a popular model with
applications across a wide variety of fields. However, its solution is hindered
by the exponentially growing state space (with respect to the number of arms)
and the combinatorial action space, making traditional reinforcement learning
methods infeasible for large-scale instances. In this paper, we propose GINO-Q,
a three-timescale stochastic approximation algorithm designed to learn an
asymptotically optimal index policy for RMABs. GINO-Q mitigates the curse of
dimensionality by decomposing the RMAB into a series of subproblems, each with
the same dimension as a single arm, ensuring that complexity increases linearly
with the number of arms. Unlike recently developed Whittle-index-based
algorithms, GINO-Q does not require RMABs to be indexable, enhancing its
flexibility and applicability. Our experimental results demonstrate that GINO-Q
consistently learns near-optimal policies, even for non-indexable RMABs where
Whittle-index-based algorithms perform poorly, and it converges significantly
faster than existing baselines.

### 摘要 (中文)

多臂赌徒（Multi-Armed Bandit，简称MAB）模型因其在多个领域的广泛应用而备受推崇。然而，它的解决方案受到状态空间（随着arms数量的增加呈指数增长）和动作空间（组合行动空间）的巨大增长的限制，使得传统强化学习方法无法应用于大规模实例。本文提出了一种新的算法GINO-Q，旨在学习RMAB中的亚最优策略，通过分解RMAB成一系列子问题，每个子问题是单个arms大小的一致维度，确保随着arms数量的增长，复杂度线性增长。与最近开发的基于Whittle指数的算法不同，GINO-Q不需要RMAB是可索引的，这增强了其灵活性和适用性。我们的实验结果表明，GINO-Q能够连续地学习近最优策略，即使对于非可索引RMAB，在Whittle指数算法性能不佳的情况下，它也能显著更快地收敛。

---

## _p_SVM_ Soft-margin SVMs with _p_-norm Hinge Loss

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09908v1)

### Abstract (English)

Support Vector Machines (SVMs) based on hinge loss have been extensively
discussed and applied to various binary classification tasks. These SVMs
achieve a balance between margin maximization and the minimization of slack due
to outliers. Although many efforts have been dedicated to enhancing the
performance of SVMs with hinge loss, studies on $p$SVMs, soft-margin SVMs with
$p$-norm hinge loss, remain relatively scarce. In this paper, we explore the
properties, performance, and training algorithms of $p$SVMs. We first derive
the generalization bound of $p$SVMs, then formulate the dual optimization
problem, comparing it with the traditional approach. Furthermore, we discuss a
generalized version of the Sequential Minimal Optimization (SMO) algorithm,
$p$SMO, to train our $p$SVM model. Comparative experiments on various datasets,
including binary and multi-class classification tasks, demonstrate the
effectiveness and advantages of our $p$SVM model and the $p$SMO method.

### 摘要 (中文)

基于对数损失的支撑向量机（SVM）在各种二分类任务中进行了广泛讨论和应用。这些SVM能够兼顾最大间隔最大化与最小化异常点带来的松弛度。尽管有许多研究致力于改进使用对数损失的SVM性能，但软边缘支持向量机（SVMs）的对数损失平滑的支持向量机的研究相对较少。本文探索了$p$SVM的性质、性能以及训练算法。首先，我们导出了$p$SVM的一般化约束，然后提出了一种比较传统的方法的双优化问题，并讨论了一个关于$p$SVM模型的全局最优解法$p$SVM。对于各种数据集，包括二分类和多分类任务，比较实验展示了我们的$p$SVM模型及其$p$SVM算法的有效性及优势。

---

## Expressive Power of Temporal Message Passing

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09918v1)

### Abstract (English)

Graph neural networks (GNNs) have recently been adapted to temporal settings,
often employing temporal versions of the message-passing mechanism known from
GNNs. We divide temporal message passing mechanisms from literature into two
main types: global and local, and establish Weisfeiler-Leman characterisations
for both. This allows us to formally analyse expressive power of temporal
message-passing models. We show that global and local temporal message-passing
mechanisms have incomparable expressive power when applied to arbitrary
temporal graphs. However, the local mechanism is strictly more expressive than
the global mechanism when applied to colour-persistent temporal graphs, whose
node colours are initially the same in all time points. Our theoretical
findings are supported by experimental evidence, underlining practical
implications of our analysis.

### 摘要 (中文)

最近，图神经网络（GNNs）被适应到时序设置中，
经常使用从GNN中知道的时空消息传递机制。我们根据文献中的时序消息传递机制分为两种主要类型：全局和局部，并为两者建立Weisfeiler-Lehman性质分析。这允许我们在理论上分析时序消息传递模型的表达能力。我们表明，在应用到任意时序图时，全球和局部时序消息传递机制具有不可比拟的表达能力。然而，当应用于保持节点颜色相同的持久性时序图时，局部机制严格比全局机制更富有表现力。我们的理论发现得到了实验证据的支持，强调了我们分析结果的实际意义。

---

## Mask in the Mirror_ Implicit Sparsification

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09966v1)

### Abstract (English)

Sparsifying deep neural networks to reduce their inference cost is an NP-hard
problem and difficult to optimize due to its mixed discrete and continuous
nature. Yet, as we prove, continuous sparsification has already an implicit
bias towards sparsity that would not require common projections of relaxed mask
variables. While implicit rather than explicit regularization induces benefits,
it usually does not provide enough flexibility in practice, as only a specific
target sparsity is obtainable. To exploit its potential for continuous
sparsification, we propose a way to control the strength of the implicit bias.
Based on the mirror flow framework, we derive resulting convergence and
optimality guarantees in the context of underdetermined linear regression and
demonstrate the utility of our insights in more general neural network
sparsification experiments, achieving significant performance gains,
particularly in the high-sparsity regime. Our theoretical contribution might be
of independent interest, as we highlight a way to enter the rich regime and
show that implicit bias is controllable by a time-dependent Bregman potential.

### 摘要 (中文)

将深层神经网络进行稀疏化以降低其推理成本是一个NP-hard问题，且由于其混合的离散和连续性，因此很难优化。然而，我们证明，连续稀疏化已经具有对稀疏性的隐含偏见，这不需要松弛掩蔽变量的一般投影。尽管隐式而不是显式的正则化带来好处，但在实践中通常不足以提供足够的灵活性，因为只能获得特定的目标稀疏度。为了利用其在连续稀疏化中的潜力，我们提出了一种控制隐式偏见强度的方法。基于镜像流动框架，我们在线性回归等欠定线性回归中导出了收敛性和最优保证，并展示了我们的见解在更一般的人工神经网络稀疏化实验中的实用价值，在高稀疏度域中取得了显著性能提升。我们的理论贡献可能具有独立的兴趣，因为我们强调了进入丰富区域的方式，并显示了隐式偏见可以通过时间依赖的Bregman势可控。

---

## The Exploration-Exploitation Dilemma Revisited_ An Entropy Perspective

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09974v1)

### Abstract (English)

The imbalance of exploration and exploitation has long been a significant
challenge in reinforcement learning. In policy optimization, excessive reliance
on exploration reduces learning efficiency, while over-dependence on
exploitation might trap agents in local optima. This paper revisits the
exploration-exploitation dilemma from the perspective of entropy by revealing
the relationship between entropy and the dynamic adaptive process of
exploration and exploitation. Based on this theoretical insight, we establish
an end-to-end adaptive framework called AdaZero, which automatically determines
whether to explore or to exploit as well as their balance of strength.
Experiments show that AdaZero significantly outperforms baseline models across
various Atari and MuJoCo environments with only a single setting. Especially in
the challenging environment of Montezuma, AdaZero boosts the final returns by
up to fifteen times. Moreover, we conduct a series of visualization analyses to
reveal the dynamics of our self-adaptive mechanism, demonstrating how entropy
reflects and changes with respect to the agent's performance and adaptive
process.

### 摘要 (中文)

探索和开采的失衡一直是强化学习中的一个重大挑战。在政策优化中，过度依赖探索会导致学习效率低下，而过分依赖于探索可能会使代理陷入局部最优解。本文从熵的角度重新审视了探索-利用困境，揭示了探索和利用动态适应过程之间的关系。基于这一理论洞察，我们建立了名为AdaZero的端到端自适应框架，该框架自动确定是否探索或利用以及两者的力量平衡。实验结果显示，在各种Atari和MuJoCo环境中仅设置一次的情况下，AdaZero在性能方面远超基准模型。尤其是在Montezuma这样的困难环境中，AdaZero最终回报提高了15倍以上。此外，我们还进行了系列可视化分析来揭示我们的自我适应机制的动力学，展示了熵如何反映并随着代理的表现和适应进程的变化而变化。

---

## Uniting contrastive and generative learning for event sequences models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09995v1)

### Abstract (English)

High-quality representation of transactional sequences is vital for modern
banking applications, including risk management, churn prediction, and
personalized customer offers. Different tasks require distinct representation
properties: local tasks benefit from capturing the client's current state,
while global tasks rely on general behavioral patterns. Previous research has
demonstrated that various self-supervised approaches yield representations that
better capture either global or local qualities.
  This study investigates the integration of two self-supervised learning
techniques - instance-wise contrastive learning and a generative approach based
on restoring masked events in latent space. The combined approach creates
representations that balance local and global transactional data
characteristics. Experiments conducted on several public datasets, focusing on
sequence classification and next-event type prediction, show that the
integrated method achieves superior performance compared to individual
approaches and demonstrates synergistic effects. These findings suggest that
the proposed approach offers a robust framework for advancing event sequences
representation learning in the financial sector.

### 摘要 (中文)

高质量的交易序列表示对于现代银行应用至关重要，包括风险管理、流失预测和个性化客户优惠。不同的任务需要不同的表示属性：本地任务受益于捕获客户端当前状态，而全局任务依赖于一般行为模式。先前的研究已经证明，各种自监督方法产生的表示能够更好地捕捉全球或局部性质。
本研究调查了两种自监督学习技术——实例级对比学习和基于恢复隐藏空间中的遮蔽事件的生成方法之间的结合。这种综合方法创建平衡当地和全局交易数据特性的代表。在几个公共数据集上进行的实验，专注于序列分类和后续事件类型的预测中，显示集成方法在与单独方法相比获得优于性能，并展示了协同效应。这些发现表明，提出的框架提供了一个强大的框架来推进金融部门事件序列表示学习的进步。

---

## Unlocking the Power of LSTM for Long Term Time Series Forecasting

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10006v1)

### Abstract (English)

Traditional recurrent neural network architectures, such as long short-term
memory neural networks (LSTM), have historically held a prominent role in time
series forecasting (TSF) tasks. While the recently introduced sLSTM for Natural
Language Processing (NLP) introduces exponential gating and memory mixing that
are beneficial for long term sequential learning, its potential short memory
issue is a barrier to applying sLSTM directly in TSF. To address this, we
propose a simple yet efficient algorithm named P-sLSTM, which is built upon
sLSTM by incorporating patching and channel independence. These modifications
substantially enhance sLSTM's performance in TSF, achieving state-of-the-art
results. Furthermore, we provide theoretical justifications for our design, and
conduct extensive comparative and analytical experiments to fully validate the
efficiency and superior performance of our model.

### 摘要 (中文)

传统的循环神经网络架构，如长短期记忆神经网络（LSTM），在时间序列预测（TSF）任务中一直占据着重要地位。尽管最近引入了适用于自然语言处理（NLP）的sLSTM，它引入了指数门控和内存混合，这对长期顺序学习非常有帮助，但在直接应用于TSF时，其潜在短记忆问题是一个障碍。为了应对这一挑战，我们提出了一个名为P-sLSTM的简单而高效的算法，该算法基于sLSTM通过集成拼接和通道独立性进行构建。这些修改极大地提高了sLSTM在TSF中的性能，实现了最先进的结果。此外，我们还提供了对设计的理论证明，并进行了广泛的比较和分析实验来完全验证模型的有效性和优异表现。

---

## PinnDE_ Physics-Informed Neural Networks for Solving Differential Equations

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10011v1)

### Abstract (English)

In recent years the study of deep learning for solving differential equations
has grown substantially. The use of physics-informed neural networks (PINNs)
and deep operator networks (DeepONets) have emerged as two of the most useful
approaches in approximating differential equation solutions using machine
learning. Here, we propose PinnDE, an open-source python library for solving
differential equations with both PINNs and DeepONets. We give a brief review of
both PINNs and DeepONets, introduce PinnDE along with the structure and usage
of the package, and present worked examples to show PinnDE's effectiveness in
approximating solutions with both PINNs and DeepONets.

### 摘要 (中文)

近年来，使用机器学习来解决微分方程的问题研究有了显著增长。使用物理引导神经网络（PINNs）和深度操作网络（DeepONets）是两种在使用机器学习近似解微分方程时非常有用的策略。这里，我们提出PinnDE，这是一个开源的Python库，用于同时使用PINNs和DeepONets来解决微分方程。我们简要介绍了PINNs和DeepONets，介绍并展示了PinnDE包的结构和使用方法，并通过工作示例展示PinnDE在使用机器学习近似解决方案的能力。

---

## Efficient Exploration in Deep Reinforcement Learning_ A Novel Bayesian Actor-Critic Algorithm

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10055v1)

### Abstract (English)

Reinforcement learning (RL) and Deep Reinforcement Learning (DRL), in
particular, have the potential to disrupt and are already changing the way we
interact with the world. One of the key indicators of their applicability is
their ability to scale and work in real-world scenarios, that is in large-scale
problems. This scale can be achieved via a combination of factors, the
algorithm's ability to make use of large amounts of data and computational
resources and the efficient exploration of the environment for viable solutions
(i.e. policies).
  In this work, we investigate and motivate some theoretical foundations for
deep reinforcement learning. We start with exact dynamic programming and work
our way up to stochastic approximations and stochastic approximations for a
model-free scenario, which forms the theoretical basis of modern reinforcement
learning. We present an overview of this highly varied and rapidly changing
field from the perspective of Approximate Dynamic Programming. We then focus
our study on the short-comings with respect to exploration of the cornerstone
approaches (i.e. DQN, DDQN, A2C) in deep reinforcement learning. On the theory
side, our main contribution is the proposal of a novel Bayesian actor-critic
algorithm. On the empirical side, we evaluate Bayesian exploration as well as
actor-critic algorithms on standard benchmarks as well as state-of-the-art
evaluation suites and show the benefits of both of these approaches over
current state-of-the-art deep RL methods. We release all the implementations
and provide a full python library that is easy to install and hopefully will
serve the reinforcement learning community in a meaningful way, and provide a
strong foundation for future work.

### 摘要 (中文)

强化学习（RL）和深度强化学习（DRL），特别是，它们有可能颠覆并正在改变我们与世界互动的方式。其中的关键指标之一是它们的可扩展性和在实际场景中的适用性，即大规模问题中。可以通过多种因素来实现这一点，算法利用大量数据和计算资源的有效探索以及可行解决方案的高效搜索（即策略）。在本文中，我们将调查和发展一些关于深度强化学习的基础理论。首先，从精确动态规划开始，然后工作到随机近似和模型无关情景下的随机近似，这构成了现代强化学习的理论基础。我们从一个视角对这一广泛变化且迅速发展的领域进行概述，即近似动态规划。然后，我们专注于针对深层强化学习中探索不足的问题。在理论上，我们的主要贡献是提出了一种新的Bayesian Actor-Critic算法。在实证方面，我们评估了基于贝叶斯探索以及Actor-Critic算法的标准基准和最先进的评价套件，并显示了这两种方法相对于当前最先进的深度RL方法的优点。我们在理论上的主要贡献是提出了一个新的Bayesian Actor-Critic算法。在实证上，我们使用标准基准和最先进的评价套件对基于贝叶斯探索以及Actor-Critic算法进行了评估，并显示了这两种方法优于当前最先进的深度RL方法。我们提供了所有实现，并提供了一个易于安装的Python库，希望这个库能以有意义的方式服务于强化学习社区，并为未来的工作奠定坚实的基础。

---

## TANGO_ Clustering with Typicality-Aware Nonlocal Mode-Seeking and Graph-Cut Optimization

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10084v1)

### Abstract (English)

Density-based clustering methods by mode-seeking usually achieve clustering
by using local density estimation to mine structural information, such as local
dependencies from lower density points to higher neighbors. However, they often
rely too heavily on \emph{local} structures and neglect \emph{global}
characteristics, which can lead to significant errors in peak selection and
dependency establishment. Although introducing more hyperparameters that revise
dependencies can help mitigate this issue, tuning them is challenging and even
impossible on real-world datasets. In this paper, we propose a new algorithm
(TANGO) to establish local dependencies by exploiting a global-view
\emph{typicality} of points, which is obtained by mining further the density
distributions and initial dependencies. TANGO then obtains sub-clusters with
the help of the adjusted dependencies, and characterizes the similarity between
sub-clusters by incorporating path-based connectivity. It achieves final
clustering by employing graph-cut on sub-clusters, thus avoiding the
challenging selection of cluster centers. Moreover, this paper provides
theoretical analysis and an efficient method for the calculation of typicality.
Experimental results on several synthetic and $16$ real-world datasets
demonstrate the effectiveness and superiority of TANGO.

### 摘要 (中文)

基于模式寻找的密度聚类方法通常通过利用局部密度估计挖掘结构信息，如低密度点到较高邻近点之间的局部依赖关系。然而，它们往往过于依赖本地结构，并忽视全局特性，这可能导致峰值选择和依赖关系建立上的重大错误。尽管引入更多调整依赖度的超参数可以帮助缓解这一问题，但对实际世界数据集进行调优是具有挑战性的甚至不可能的。在本文中，我们提出了一种新的算法（TANGO），以通过利用点的全局视图“典型性”来建立局部依赖关系，这是通过对密度分布和初始依赖关系进行进一步挖掘获得的。然后，TANGO通过调整依赖度帮助获取子群，通过结合路径基连接度来刻画子群间的相似性。最后，它使用于子群上进行图分割的方法避免了选择簇中心的挑战。此外，本文还提供了理论分析和计算典型性的高效方法。在几个合成和16个真实世界数据集上，TANGO的有效性和优越性得到了证明。

---

## MASALA_ Model-Agnostic Surrogate Explanations by Locality Adaptation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10085v1)

### Abstract (English)

Existing local Explainable AI (XAI) methods, such as LIME, select a region of
the input space in the vicinity of a given input instance, for which they
approximate the behaviour of a model using a simpler and more interpretable
surrogate model. The size of this region is often controlled by a user-defined
locality hyperparameter. In this paper, we demonstrate the difficulties
associated with defining a suitable locality size to capture impactful model
behaviour, as well as the inadequacy of using a single locality size to explain
all predictions. We propose a novel method, MASALA, for generating
explanations, which automatically determines the appropriate local region of
impactful model behaviour for each individual instance being explained. MASALA
approximates the local behaviour used by a complex model to make a prediction
by fitting a linear surrogate model to a set of points which experience similar
model behaviour. These points are found by clustering the input space into
regions of linear behavioural trends exhibited by the model. We compare the
fidelity and consistency of explanations generated by our method with existing
local XAI methods, namely LIME and CHILLI. Experiments on the PHM08 and MIDAS
datasets show that our method produces more faithful and consistent
explanations than existing methods, without the need to define any sensitive
locality hyperparameters.

### 摘要 (中文)

现有本地可解释人工智能（XAI）方法，如LIME，选择位于给定输入实例附近的输入空间中的一个区域，使用一个更简单、更具解释性的拟合模型对其进行近似行为。这个区域的大小通常由用户定义的局部性超参数控制。本论文展示了在捕获有影响力模型行为时定义适当局部尺寸所面临的困难，以及仅使用单一局部尺寸来解释所有预测的不足。我们提出了一种新的方法MASALA，用于生成解释，该方法自动确定每个被解释的实例中影响模型行为的有效区域。通过设置线性拟合模型到具有相似模型行为的点集，MASALA估计了复杂模型做出预测时使用的当地行为。这些点是通过将输入空间聚类成表现出类似模型行为的线性行为趋势的模式而找到的。我们比较了我们的方法与现有本地XAI方法，即LIME和CHILLI的 fidelity 和一致性。在PHM08和MIDAS数据集上的实验表明，我们的方法产生的解释比现有的方法更加忠实且一致，无需定义任何敏感的局部性超参数。

---

## Concept Distillation from Strong to Weak Models via Hypotheses-to-Theories Prompting

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09365v1)

### Abstract (English)

Hand-crafting high quality prompts to optimize the performance of language
models is a complicated and labor-intensive process. Furthermore, when
migrating to newer, smaller, or weaker models (possibly due to latency or cost
gains), prompts need to be updated to re-optimize the task performance. We
propose Concept Distillation (CD), an automatic prompt optimization technique
for enhancing weaker models on complex tasks. CD involves: (1) collecting
mistakes made by weak models with a base prompt (initialization), (2) using a
strong model to generate reasons for these mistakes and create rules/concepts
for weak models (induction), and (3) filtering these rules based on validation
set performance and integrating them into the base prompt
(deduction/verification). We evaluated CD on NL2Code and mathematical reasoning
tasks, observing significant performance boosts for small and weaker language
models. Notably, Mistral-7B's accuracy on Multi-Arith increased by 20%, and
Phi-3-mini-3.8B's accuracy on HumanEval rose by 34%. Compared to other
automated methods, CD offers an effective, cost-efficient strategy for
improving weak models' performance on complex tasks and enables seamless
workload migration across different language models without compromising
performance.

### 摘要 (中文)

手工优化高质量提示以优化语言模型的性能是一个复杂且耗时的过程。此外，当迁移到新的、更小或较弱的模型（可能由于延迟或成本收益）时，需要更新提示以重新优化任务表现。我们提出了概念分层（CD），这是一种自动提示优化技术，用于增强较弱模型在复杂任务上的性能。CD涉及：(1)收集弱模型中错误的初始基提示；(2)使用强模型生成这些错误的原因，并创建规则/概念供弱模型（引证）；(3)根据验证集性能过滤这些规则，并将其集成到基础提示中（归纳/验证）。我们在NL2Code和数学推理任务上进行了评估，观察到了对小型和较弱的语言模型的小幅度性能提升。值得注意的是，Mistral-7B在Multi-Arith上的准确率提高了20%，而Phi-3-mini-3.8B在HumanEval上的准确率提高了34%。与其他自动方法相比，CD提供了一种有效且经济高效的策略来提高复杂任务中的弱模型的表现，并使不同语言模型之间的负载迁移无缝进行，而不影响性能。

---

## ELASTIC_ Efficient Linear Attention for Sequential Interest Compression

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09380v1)

### Abstract (English)

State-of-the-art sequential recommendation models heavily rely on
transformer's attention mechanism. However, the quadratic computational and
memory complexities of self attention have limited its scalability for modeling
users' long range behaviour sequences. To address this problem, we propose
ELASTIC, an Efficient Linear Attention for SequenTial Interest Compression,
requiring only linear time complexity and decoupling model capacity from
computational cost. Specifically, ELASTIC introduces a fixed length interest
experts with linear dispatcher attention mechanism which compresses the
long-term behaviour sequences to a significantly more compact representation
which reduces up to 90% GPU memory usage with x2.7 inference speed up. The
proposed linear dispatcher attention mechanism significantly reduces the
quadratic complexity and makes the model feasible for adequately modeling
extremely long sequences. Moreover, in order to retain the capacity for
modeling various user interests, ELASTIC initializes a vast learnable interest
memory bank and sparsely retrieves compressed user's interests from the memory
with a negligible computational overhead. The proposed interest memory
retrieval technique significantly expands the cardinality of available interest
space while keeping the same computational cost, thereby striking a trade-off
between recommendation accuracy and efficiency. To validate the effectiveness
of our proposed ELASTIC, we conduct extensive experiments on various public
datasets and compare it with several strong sequential recommenders.
Experimental results demonstrate that ELASTIC consistently outperforms
baselines by a significant margin and also highlight the computational
efficiency of ELASTIC when modeling long sequences. We will make our
implementation code publicly available.

### 摘要 (中文)

最先进的顺序推荐模型主要依赖于Transformer的注意力机制。然而，自注意力的线性计算和内存复杂度限制了其用于表示用户长期行为序列的能力。为了解决这个问题，我们提出了Elastic，一个高效的线性注意力序列兴趣压缩器，只需要线性的运行时间，并且从计算成本中解耦了模型容量。具体来说，Elastic引入了一个固定长度的兴趣专家，使用线性调度注意力机制，以压缩长时间的行为序列到一个显著更紧凑的表示，这在x2.7的推理速度上可以减少高达90%的GPU内存使用量。提出的线性调度注意力机制大大减少了线性和提高了模型对足够长序列建模的可行性。此外，为了保留模型对各种用户兴趣进行建模的能力，Elastic初始化了一个巨大的可学习的兴趣记忆库，并用很少的计算开销从内存中稀疏地检索压缩后的用户兴趣。提出的兴趣记忆检索技术极大地扩展了可用兴趣空间的卡诺图，同时保持相同的计算成本，从而在推荐精度和效率之间取得了折衷。为了验证我们提出的Elastic的有效性，我们在各种公共数据集上进行了广泛的实验，并与几个强大的顺序推荐系统进行了比较。实验结果表明，在显著的差距下，Elastic一直优于基准。当我们尝试建模非常长的序列时，Elastic在保持相同计算成本的情况下具有更高的效率。我们将公开我们的实现代码。

---

## Offline RLHF Methods Need More Accurate Supervision Signals

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09385v1)

### Abstract (English)

With the rapid advances in Large Language Models (LLMs), aligning LLMs with
human preferences become increasingly important. Although Reinforcement
Learning with Human Feedback (RLHF) proves effective, it is complicated and
highly resource-intensive. As such, offline RLHF has been introduced as an
alternative solution, which directly optimizes LLMs with ranking losses on a
fixed preference dataset. Current offline RLHF only captures the ``ordinal
relationship'' between responses, overlooking the crucial aspect of ``how
much'' one is preferred over the others. To address this issue, we propose a
simple yet effective solution called \textbf{R}eward \textbf{D}ifference
\textbf{O}ptimization, shorted as \textbf{RDO}. Specifically, we introduce {\it
reward difference coefficients} to reweigh sample pairs in offline RLHF. We
then develop a {\it difference model} involving rich interactions between a
pair of responses for predicting these difference coefficients. Experiments
with 7B LLMs on the HH and TL;DR datasets substantiate the effectiveness of our
method in both automatic metrics and human evaluation, thereby highlighting its
potential for aligning LLMs with human intent and values.

### 摘要 (中文)

随着大型语言模型（LLM）技术的快速发展，使模型与人类偏好相匹配变得越来越重要。虽然强化学习结合人机反馈（RLHF）方法有效，但其复杂性和资源密集度较高。因此，引入了离线RLHF作为替代解决方案，直接优化LLMs在固定偏好的数据集上进行排名损失。当前的离线RLHF仅捕捉到“响应之间的顺序关系”，忽略了“哪一种更受欢迎”的关键方面。为了解决这个问题，我们提出了一种简单而有效的解决方案称为奖励差异优化，简称为RDO。具体而言，我们将引入“奖励差异系数”来重新分配样本对在离线RLHF中的权重。然后，我们发展了一个涉及两个响应之间丰富互动的差值模型来预测这些差值系数。通过7B LLM在HH和TL；DR数据集上的实验验证，证明了我们的方法在自动指标和人类评估方面的有效性，从而凸显出它在使模型与人类意图和价值观相匹配方面的潜力。

---

## Comparison between the Structures of Word Co-occurrence and Word Similarity Networks for Ill-formed and Well-formed Texts in Taiwan Mandarin

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09404v1)

### Abstract (English)

The study of word co-occurrence networks has attracted the attention of
researchers due to their potential significance as well as applications.
Understanding the structure of word co-occurrence networks is therefore
important to fully realize their significance and usages. In past studies, word
co-occurrence networks built on well-formed texts have been found to possess
certain characteristics, including being small-world, following a two-regime
power law distribution, and being generally disassortative. On the flip side,
past studies have found that word co-occurrence networks built from ill-formed
texts such as microblog posts may behave differently from those built from
well-formed documents. While both kinds of word co-occurrence networks are
small-world and disassortative, word co-occurrence networks built from
ill-formed texts are scale-free and follow the power law distribution instead
of the two-regime power law distribution. However, since past studies on the
behavior of word co-occurrence networks built from ill-formed texts only
investigated English, the universality of such characteristics remains to be
seen among different languages. In addition, it is yet to be investigated
whether there could be possible similitude/differences between word
co-occurrence networks and other potentially comparable networks. This study
therefore investigates and compares the structure of word co-occurrence
networks and word similarity networks based on Taiwan Mandarin ill-formed
internet forum posts and compare them with those built with well-formed
judicial judgments, and seeks to find out whether the three aforementioned
properties (scale-free, small-world, and disassortative) for ill-formed and
well-formed texts are universal among different languages and between word
co-occurrence and word similarity networks.

### 摘要 (中文)

由于它们的潜在重要性和应用，对词共现网络的研究吸引了研究人员的关注。因此，理解词共现网络结构的重要性是实现其重要性与应用的关键。在以往的研究中，建立在良好文本上的词共现网络已被发现具有某些特性，包括小世界、遵循两阶幂律分布和一般非配对；相反，在不好的文本（如微博帖子）上建立的词共现网络可能表现出与良好文档上建立的词共现网络不同的行为。虽然两种类型的词共现网络都是小世界和非配对，但是来自不良文本的词共现网络却是无界且遵循幂律分布而不是两阶幂律分布。然而，关于不良文本上建立的词共现网络的行为只有研究了英语，这些特征是否适用于不同语言中的其他可能相等的网络仍然是未知的。此外，尚未调查是否有可能的相似性/差异存在词共现网络和可能的可比较网络。因此，本研究根据台湾普通话不良互联网论坛帖子及其与良好判例的对比，并将其与良好文本上建立的词相似性网络进行比较，以确定不同语言和词共现网络和词相似性网络中上述三种属性（无界、小世界和非配对）的普遍性。

---

## Challenges and Responses in the Practice of Large Language Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09416v1)

### Abstract (English)

This paper carefully summarizes extensive and profound questions from all
walks of life, focusing on the current high-profile AI field, covering multiple
dimensions such as industry trends, academic research, technological innovation
and business applications. This paper meticulously curates questions that are
both thought-provoking and practically relevant, providing nuanced and
insightful answers to each. To facilitate readers' understanding and reference,
this paper specifically classifies and organizes these questions systematically
and meticulously from the five core dimensions of computing power
infrastructure, software architecture, data resources, application scenarios,
and brain science. This work aims to provide readers with a comprehensive,
in-depth and cutting-edge AI knowledge framework to help people from all walks
of life grasp the pulse of AI development, stimulate innovative thinking, and
promote industrial progress.

### 摘要 (中文)

这篇论文精心总结了来自各个领域的深刻而广泛的问题，重点放在当前的高调人工智能领域上，涵盖了行业趋势、学术研究、技术创新和商业应用等多个维度。这篇论文细致地收集了既具有启发性又实际相关的深入问题，并提供了对每个问题的细腻且有洞察力的回答。为了方便读者理解和参考，这篇论文特别系统地从计算能力基础设施、软件架构、数据资源、应用场景和脑科学五个核心维度进行了分类和组织，以便提供给所有层次的人都一个全面、深入和前沿的人工智能知识框架，帮助各行各业的人们把握人工智能发展的脉搏，激发创新思维，促进工业进步。

---

## Distinguish Confusion in Legal Judgment Prediction via Revised Relation Knowledge

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09422v1)

### Abstract (English)

Legal Judgment Prediction (LJP) aims to automatically predict a law case's
judgment results based on the text description of its facts. In practice, the
confusing law articles (or charges) problem frequently occurs, reflecting that
the law cases applicable to similar articles (or charges) tend to be misjudged.
Although some recent works based on prior knowledge solve this issue well, they
ignore that confusion also occurs between law articles with a high posterior
semantic similarity due to the data imbalance problem instead of only between
the prior highly similar ones, which is this work's further finding. This paper
proposes an end-to-end model named \textit{D-LADAN} to solve the above
challenges. On the one hand, D-LADAN constructs a graph among law articles
based on their text definition and proposes a graph distillation operation
(GDO) to distinguish the ones with a high prior semantic similarity. On the
other hand, D-LADAN presents a novel momentum-updated memory mechanism to
dynamically sense the posterior similarity between law articles (or charges)
and a weighted GDO to adaptively capture the distinctions for revising the
inductive bias caused by the data imbalance problem. We perform extensive
experiments to demonstrate that D-LADAN significantly outperforms
state-of-the-art methods in accuracy and robustness.

### 摘要 (中文)

法律判决预测（LJP）旨在根据事实描述自动预测案件的判决结果。在实践中，混淆的法律文章（或指控）问题经常出现，这反映出适用类似文章（或指控）的案件往往被误判。虽然一些基于先验知识的工作很好地解决了这个问题，但它们忽略了由于数据不平衡问题而发生的混淆，并且只关注于前文高度相似的文章之间，这是我们工作的进一步发现。这篇论文提出了一种名为D-LADAN的端到端模型来解决上述挑战。一方面，D-LADAN根据文本定义构建了一个关于法律文章之间的图，并提出了一个图蒸馏操作（GDO）以区分那些具有高先验语义相似性的文章。另一方面，D-LADAN提出了一种新颖的记忆更新机制来动态地感知法律文章（或指控）与权重GDO之间的后验相似性，从而适应性地捕获由数据不平衡问题引起的归纳偏差差异。我们进行了广泛的实验来演示，在准确性上D-LADAN显着优于最先进的方法。

---

## FASST_ Fast LLM-based Simultaneous Speech Translation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09430v1)

### Abstract (English)

Simultaneous speech translation (SST) takes streaming speech input and
generates text translation on the fly. Existing methods either have high
latency due to recomputation of input representations, or fall behind of
offline ST in translation quality. In this paper, we propose FASST, a fast
large language model based method for streaming speech translation. We propose
blockwise-causal speech encoding and consistency mask, so that streaming speech
input can be encoded incrementally without recomputation. Furthermore, we
develop a two-stage training strategy to optimize FASST for simultaneous
inference. We evaluate FASST and multiple strong prior models on MuST-C
dataset. Experiment results show that FASST achieves the best quality-latency
trade-off. It outperforms the previous best model by an average of 1.5 BLEU
under the same latency for English to Spanish translation.

### 摘要 (中文)

实时语音翻译（SST）利用流式语音输入生成即时的文本翻译。现有的方法由于需要重新计算输入表示而导致延迟高，或者在翻译质量上落后于离线ST。在这篇论文中，我们提出了一种快速大规模语言模型基方法FASST，用于实时语音翻译。我们提出了块级因果性语音编码和一致性掩码，这样可以无须重新计算就可以逐块地对流式语音输入进行编码。此外，我们开发了一个两阶段训练策略优化FASST以同时推理。我们在MuST-C数据集上评估了FASST及其多个强先验模型。实验结果显示，FASST在相同延迟下实现了最佳的质量-延迟权衡。它比之前最好的模型平均高出1.5个BLEU。

---

## HySem_ A context length optimized LLM pipeline for unstructured tabular extraction

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09434v1)

### Abstract (English)

Regulatory compliance reporting in the pharmaceutical industry relies on
detailed tables, but these are often under-utilized beyond compliance due to
their unstructured format and arbitrary content. Extracting and semantically
representing tabular data is challenging due to diverse table presentations.
Large Language Models (LLMs) demonstrate substantial potential for semantic
representation, yet they encounter challenges related to accuracy and context
size limitations, which are crucial considerations for the industry
applications. We introduce HySem, a pipeline that employs a novel context
length optimization technique to generate accurate semantic JSON
representations from HTML tables. This approach utilizes a custom fine-tuned
model specifically designed for cost- and privacy-sensitive small and medium
pharmaceutical enterprises. Running on commodity hardware and leveraging
open-source models, our auto-correcting agents rectify both syntax and semantic
errors in LLM-generated content. HySem surpasses its peer open-source models in
accuracy and provides competitive performance when benchmarked against OpenAI
GPT-4o and effectively addresses context length limitations, which is a crucial
factor for supporting larger tables.

### 摘要 (中文)

在制药行业，法规合规报告依赖于详细的表格，但这些通常无法充分利用其潜力，因为它们的格式不规范且内容任意。从多维度提取和语义表示表格数据是一个挑战，因为不同的表展示方式导致了数据的不同性。大型语言模型（LLMs）在语义表示方面具有巨大的潜力，但在准确性、上下文大小限制等方面遇到了挑战，这对行业的应用至关重要。我们引入HySem，这是一种管道，采用一种新颖的上下文长度优化技术，可以从HTML表格中生成准确的语义JSON表示。这种方法利用了一个专门设计的小型到中型企业成本和隐私敏感的自定义微调模型。它运行在通用硬件上，并利用开源模型，我们的自动修正器可以纠正LLM生成的内容中的语法和语义错误。HySem在其同行开源模型中超过了OpenAI GPT-4，在与OpenAI GPT-4进行对比时表现良好，有效地解决了上下文长度限制的问题，这是支持更大表格的关键因素。

---

## Towards Boosting LLMs-driven Relevance Modeling with Progressive Retrieved Behavior-augmented Prompting

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09439v1)

### Abstract (English)

Relevance modeling is a critical component for enhancing user experience in
search engines, with the primary objective of identifying items that align with
users' queries. Traditional models only rely on the semantic congruence between
queries and items to ascertain relevance. However, this approach represents
merely one aspect of the relevance judgement, and is insufficient in isolation.
Even powerful Large Language Models (LLMs) still cannot accurately judge the
relevance of a query and an item from a semantic perspective. To augment
LLMs-driven relevance modeling, this study proposes leveraging user
interactions recorded in search logs to yield insights into users' implicit
search intentions. The challenge lies in the effective prompting of LLMs to
capture dynamic search intentions, which poses several obstacles in real-world
relevance scenarios, i.e., the absence of domain-specific knowledge, the
inadequacy of an isolated prompt, and the prohibitive costs associated with
deploying LLMs. In response, we propose ProRBP, a novel Progressive Retrieved
Behavior-augmented Prompting framework for integrating search scenario-oriented
knowledge with LLMs effectively. Specifically, we perform the user-driven
behavior neighbors retrieval from the daily search logs to obtain
domain-specific knowledge in time, retrieving candidates that users consider to
meet their expectations. Then, we guide LLMs for relevance modeling by
employing advanced prompting techniques that progressively improve the outputs
of the LLMs, followed by a progressive aggregation with comprehensive
consideration of diverse aspects. For online serving, we have developed an
industrial application framework tailored for the deployment of LLMs in
relevance modeling. Experiments on real-world industry data and online A/B
testing demonstrate our proposal achieves promising performance.

### 摘要 (中文)

相关性建模是搜索引擎用户体验增强的关键组成部分，主要目标是识别与用户查询相匹配的项目。传统的模型仅依赖于查询和项目之间的语义一致性来确定相关性。然而，这种方法只是相关判断的一个方面，并且在孤立的情况下是不够的。即使强大的语言模型（LLMs）也无法准确地从语义角度判断查询和项目的相关性。为了补充基于LLMs的相关性建模，本研究提出利用搜索日志中记录的用户交互来获取用户隐含搜索意图的信息。挑战在于有效地引导LLMs捕捉动态搜索意图，这在实际关联场景中存在几个障碍，即缺乏特定知识、孤立提示的不足以及部署LLMs的成本过高。为此，我们提出了一种名为ProRBP的新进展检索行为增强提示框架，用于有效整合搜索场景相关的知识与LLMs。具体来说，我们将从每日的搜索日志中驱动用户的行为邻居检索以获取时间内的领域专有知识，检索满足用户期望的候选者。然后，我们指导LLMs进行相关性建模，采用先进的提示技术逐步改进LLMs的输出，最后通过考虑各种方面的综合考虑进行逐步聚合。对于在线服务，我们已经开发了一个适用于部署LLMs进行相关性建模工业应用框架。对现实世界行业的数据进行实验和在线A/B测试表明，我们的提议取得了令人满意的表现。

---

## PanoSent_ A Panoptic Sextuple Extraction Benchmark for Multimodal Conversational Aspect-based Sentiment Analysis

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09481v1)

### Abstract (English)

While existing Aspect-based Sentiment Analysis (ABSA) has received extensive
effort and advancement, there are still gaps in defining a more holistic
research target seamlessly integrating multimodality, conversation context,
fine-granularity, and also covering the changing sentiment dynamics as well as
cognitive causal rationales. This paper bridges the gaps by introducing a
multimodal conversational ABSA, where two novel subtasks are proposed: 1)
Panoptic Sentiment Sextuple Extraction, panoramically recognizing holder,
target, aspect, opinion, sentiment, rationale from multi-turn multi-party
multimodal dialogue. 2) Sentiment Flipping Analysis, detecting the dynamic
sentiment transformation throughout the conversation with the causal reasons.
To benchmark the tasks, we construct PanoSent, a dataset annotated both
manually and automatically, featuring high quality, large scale, multimodality,
multilingualism, multi-scenarios, and covering both implicit and explicit
sentiment elements. To effectively address the tasks, we devise a novel
Chain-of-Sentiment reasoning framework, together with a novel multimodal large
language model (namely Sentica) and a paraphrase-based verification mechanism.
Extensive evaluations demonstrate the superiority of our methods over strong
baselines, validating the efficacy of all our proposed methods. The work is
expected to open up a new era for the ABSA community, and thus all our codes
and data are open at https://PanoSent.github.io/

### 摘要 (中文)

虽然现有的基于方面（Aspect-Based Sentiment Analysis）已经获得了广泛的努力和进步，但仍然存在定义一个更全面的研究目标无缝集成多模态、对话上下文、精确度以及也涵盖情感动态变化和认知因果理由的缺失。本论文通过引入一个多模态会话ABS A，提出两个新子任务：1）全景情绪六元体提取，全景识别持有人、目标、方面、意见、情绪、原因从多轮多方多模态对话中；2）情感翻转分析，在对话中检测动态的情感转变，并追溯其原因。为了基准任务，我们构建了一个名为PanoSent的数据集，该数据集手动标注并自动标注，具有高质量、大规模、多模态、跨语种、多场景的特点，并覆盖了隐含和显性的情绪元素。为了有效地解决这些任务，我们设计了一种新的链式情感推理框架，以及一种新的多模态大型语言模型（即Sentica），以及一种基于引证的验证机制。广泛的评估展示了我们的方法优于强基线的有效性，证明了所有我们提出的策略的有效性。这项工作预计将开启ABS A社区的新时代，因此所有的代码和数据都开放在https://PanoSent.github.io/。

---

## REFINE-LM_ Mitigating Language Model Stereotypes via Reinforcement Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09489v1)

### Abstract (English)

With the introduction of (large) language models, there has been significant
concern about the unintended bias such models may inherit from their training
data. A number of studies have shown that such models propagate gender
stereotypes, as well as geographical and racial bias, among other biases. While
existing works tackle this issue by preprocessing data and debiasing
embeddings, the proposed methods require a lot of computational resources and
annotation effort while being limited to certain types of biases. To address
these issues, we introduce REFINE-LM, a debiasing method that uses
reinforcement learning to handle different types of biases without any
fine-tuning. By training a simple model on top of the word probability
distribution of a LM, our bias agnostic reinforcement learning method enables
model debiasing without human annotations or significant computational
resources. Experiments conducted on a wide range of models, including several
LMs, show that our method (i) significantly reduces stereotypical biases while
preserving LMs performance; (ii) is applicable to different types of biases,
generalizing across contexts such as gender, ethnicity, religion, and
nationality-based biases; and (iii) it is not expensive to train.

### 摘要 (中文)

随着大型语言模型的引入，人们担心这些模型可能会从训练数据中继承不希望出现的偏见。许多研究表明，这种模型在传播性别刻板印象、地理和种族偏见等其他偏见方面具有潜在影响。虽然现有工作通过预处理数据并消除嵌入来解决这一问题，但我们的方法需要大量的计算资源和标注努力，并且只能处理某些类型的偏见。针对这些问题，我们提出了一种名为REFINE-LM的方法，该方法使用强化学习来处理不同类型偏见而不进行微调。通过对LM词概率分布上的简单模型进行培训，我们的无监督强化学习方法使模型能够自动解蔽而无需人类注释或大量计算资源。对各种模型（包括几个语言模型）进行的实验显示，我们的方法（i）显著减少了 stereotypes的偏见，同时保持了LMS的性能；（ii）适用于不同类型的偏见，可以在诸如性别、种族、宗教和国籍偏见等上下文中通用；（iii）训练成本并不昂贵。

---

## Beyond Local Views_ Global State Inference with Diffusion Models for Cooperative Multi-Agent Reinforcement Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09501v1)

### Abstract (English)

In partially observable multi-agent systems, agents typically only have
access to local observations. This severely hinders their ability to make
precise decisions, particularly during decentralized execution. To alleviate
this problem and inspired by image outpainting, we propose State Inference with
Diffusion Models (SIDIFF), which uses diffusion models to reconstruct the
original global state based solely on local observations. SIDIFF consists of a
state generator and a state extractor, which allow agents to choose suitable
actions by considering both the reconstructed global state and local
observations. In addition, SIDIFF can be effortlessly incorporated into current
multi-agent reinforcement learning algorithms to improve their performance.
Finally, we evaluated SIDIFF on different experimental platforms, including
Multi-Agent Battle City (MABC), a novel and flexible multi-agent reinforcement
learning environment we developed. SIDIFF achieved desirable results and
outperformed other popular algorithms.

### 摘要 (中文)

在部分观测的多代理系统中，通常只有局部观察。这极大地限制了它们做出精确决策的能力，尤其是在分布式执行时尤其如此。为了缓解这个问题，并受到图像修复的启发，我们提出了基于扩散模型的状态推理（SIDIFF），该模型仅根据局部观察来重建原始全局状态。SIDIFF由一个状态生成器和一个状态提取器组成，允许代理人通过考虑重构后的全球状态和局部观察来选择合适的行动。此外，SIDIFF可以轻松地集成到当前的多代理强化学习算法中以提高其性能。最后，我们在不同的实验平台上评估了SIDIFF，包括我们开发的新型灵活多代理强化学习环境Multi-Agent Battle City（MABC）。SIDIFF取得了令人满意的结果，并且超过了其他流行的算法。

---

## A Logic for Policy Based Resource Exchanges in Multiagent Systems

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09516v1)

### Abstract (English)

In multiagent systems autonomous agents interact with each other to achieve
individual and collective goals. Typical interactions concern negotiation and
agreement on resource exchanges. Modeling and formalizing these agreements pose
significant challenges, particularly in capturing the dynamic behaviour of
agents, while ensuring that resources are correctly handled. Here, we propose
exchange environments as a formal setting where agents specify and obey
exchange policies, which are declarative statements about what resources they
offer and what they require in return. Furthermore, we introduce a decidable
extension of the computational fragment of linear logic as a fundamental tool
for representing exchange environments and studying their dynamics in terms of
provability.

### 摘要 (中文)

在多代理系统中，自主代理与其他代理互动以实现个人和集体目标。典型交互涉及谈判和资源交换协议的协商。这些协议的设计和规范性是巨大的挑战，尤其是在捕捉代理动态行为的同时，确保资源正确处理。在这里，我们提出环境作为正式设置，在其中，代理指定并遵守交易政策，后者是对他们提供什么资源以及他们需要什么回报的声明性的陈述。此外，我们引入了一种可验证的扩展线性逻辑计算片段作为代表交易环境和将其动态表示为证明的一种基本工具。

---

## Revisiting the Graph Reasoning Ability of Large Language Models_ Case Studies in Translation_ Connectivity and Shortest Path

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09529v1)

### Abstract (English)

Large Language Models (LLMs) have achieved great success in various reasoning
tasks. In this work, we focus on the graph reasoning ability of LLMs. Although
theoretical studies proved that LLMs are capable of handling graph reasoning
tasks, empirical evaluations reveal numerous failures. To deepen our
understanding on this discrepancy, we revisit the ability of LLMs on three
fundamental graph tasks: graph description translation, graph connectivity, and
the shortest-path problem. Our findings suggest that LLMs can fail to
understand graph structures through text descriptions and exhibit varying
performance for all these three fundamental tasks. Meanwhile, we perform a
real-world investigation on knowledge graphs and make consistent observations
with our findings. The codes and datasets are available.

### 摘要 (中文)

大型语言模型（LLM）在各种推理任务中取得了巨大成功。在这项工作中，我们专注于LLM的图推理能力。虽然理论研究证明了LMM能够处理图推理任务，但实证评估揭示了诸多失败。为了加深对这一差异的理解，我们将重新评估LLM在三个基本图任务上的能力：图形描述翻译、图形连通性和最短路径问题。我们的发现表明，尽管文本描述可以理解图结构，但LLM在这些三个基本任务上表现各异。同时，我们在知识图谱上进行了一次现实世界的调查，并与我们的发现保持一致。代码和数据集可供使用。

---

## Using ChatGPT to Score Essays and Short-Form Constructed Responses

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09540v1)

### Abstract (English)

This study aimed to determine if ChatGPT's large language models could match
the scoring accuracy of human and machine scores from the ASAP competition. The
investigation focused on various prediction models, including linear
regression, random forest, gradient boost, and boost. ChatGPT's performance was
evaluated against human raters using quadratic weighted kappa (QWK) metrics.
Results indicated that while ChatGPT's gradient boost model achieved QWKs close
to human raters for some data sets, its overall performance was inconsistent
and often lower than human scores. The study highlighted the need for further
refinement, particularly in handling biases and ensuring scoring fairness.
Despite these challenges, ChatGPT demonstrated potential for scoring
efficiency, especially with domain-specific fine-tuning. The study concludes
that ChatGPT can complement human scoring but requires additional development
to be reliable for high-stakes assessments. Future research should improve
model accuracy, address ethical considerations, and explore hybrid models
combining ChatGPT with empirical methods.

### 摘要 (中文)

这项研究旨在确定ChatGPT的大语言模型是否能与ASAP竞赛的人工智能和机器人的评分相匹配。调查集中在各种预测模型上，包括线性回归、随机森林、梯度提升、和boost。通过使用平方权重kappa（QWK）指标评估ChatGPT的表现，对人类评级者进行评估。结果显示，尽管在某些数据集上，ChatGPT的梯度提升模型的QWK分数接近于人类评级者，但其整体表现是不一致的，并且往往低于人类得分。该研究强调了进一步改进的需求，尤其是在处理偏见并确保评分公平方面。虽然存在这些挑战，但ChatGPT显示了提高效率的可能性，特别是在针对特定领域进行微调时。研究表明，ChatGPT可以补充人类评分，但它需要额外的发展才能在高风险评估中可靠。未来的研究应该改善模型准确性，解决伦理考虑，并探索结合ChatGPT和实证方法的混合模型。

---

## Grammatical Error Feedback_ An Implicit Evaluation Approach

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09565v1)

### Abstract (English)

Grammatical feedback is crucial for consolidating second language (L2)
learning. Most research in computer-assisted language learning has focused on
feedback through grammatical error correction (GEC) systems, rather than
examining more holistic feedback that may be more useful for learners. This
holistic feedback will be referred to as grammatical error feedback (GEF). In
this paper, we present a novel implicit evaluation approach to GEF that
eliminates the need for manual feedback annotations. Our method adopts a
grammatical lineup approach where the task is to pair feedback and essay
representations from a set of possible alternatives. This matching process can
be performed by appropriately prompting a large language model (LLM). An
important aspect of this process, explored here, is the form of the lineup,
i.e., the selection of foils. This paper exploits this framework to examine the
quality and need for GEC to generate feedback, as well as the system used to
generate feedback, using essays from the Cambridge Learner Corpus.

### 摘要 (中文)

语法反馈对于巩固第二语言（L2）学习至关重要。在计算机辅助语言学习的研究中，大多数研究集中在通过语法错误修正系统提供反馈上，而忽略了可能对学习者更有用的更全面反馈。这种更全面的反馈将被称作语法错误反馈（GEF）。在此文中，我们提出了一种新的隐式评价方法来生成EF，它消除了手动标注反馈注释的需求。我们的方法采用语法排列的方法，其中任务是匹配从一组可能替代品中提取出的反馈和论文表示。这个匹配过程可以通过适当地提示大型语言模型（LLM）来执行。在这个过程中，探索的重要方面之一是队列的形式，即选择靶子的选择，这是本文的重点。这一框架利用此框架来审查生成反馈的质量以及用于生成反馈的系统，并使用来自剑桥学习者库的论文进行评估。

---

## MergeRepair_ An Exploratory Study on Merging Task-Specific Adapters in Code LLMs for Automated Program Repair

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09568v1)

### Abstract (English)

[Context] Large Language Models (LLMs) have shown good performance in several
software development-related tasks such as program repair, documentation, code
refactoring, debugging, and testing. Adapters are specialized, small modules
designed for parameter efficient fine-tuning of LLMs for specific tasks,
domains, or applications without requiring extensive retraining of the entire
model. These adapters offer a more efficient way to customize LLMs for
particular needs, leveraging the pre-existing capabilities of the large model.
Merging LLMs and adapters has shown promising results for various natural
language domains and tasks, enabling the use of the learned models and adapters
without additional training for a new task. [Objective] This research proposes
continual merging and empirically studies the capabilities of merged adapters
in Code LLMs, specially for the Automated Program Repair (APR) task. The goal
is to gain insights into whether and how merging task-specific adapters can
affect the performance of APR. [Method] In our framework, MergeRepair, we plan
to merge multiple task-specific adapters using three different merging methods
and evaluate the performance of the merged adapter for the APR task.
Particularly, we will employ two main merging scenarios for all three
techniques, (i) merging using equal-weight averaging applied on parameters of
different adapters, where all adapters are of equal importance; and (ii) our
proposed approach, continual merging, in which we sequentially merge the
task-specific adapters and the order and weight of merged adapters matter. By
exploratory study of merging techniques, we will investigate the improvement
and generalizability of merged adapters for APR. Through continual merging, we
will explore the capability of merged adapters and the effect of task order, as
it occurs in real-world software projects.

### 摘要 (中文)

【背景】大型语言模型（Large Language Models，LLMs）在程序修复、文档编写、代码重构、调试和测试等软件开发相关任务中表现出良好的性能。专用的、小型模块专门设计用于参数高效微调特定任务、领域或应用程序的LLMs适配器，无需重新训练整个模型即可实现这一点。这些适配器提供了更有效的方式来定制针对特定需求的LLMs，利用大型模型预存能力的优势。

融合LLMs和适配器已经展示了在自然语言领域的各种任务上的良好效果，使得使用学习到的模型和适配器而无需对新的任务进行额外训练。【目标】本研究提出持续融合的概念，并通过实验性研究确定了合并适配器在代码LLMs中的能力和如何影响自动程序修复（Automated Program Repair，APR）任务的表现。目标是了解是否以及如何结合特定任务的适配器可以影响APR任务的性能。【方法】我们的框架MergeRepair计划采用三种不同的融合技术来合并多个特定任务的适配器，并评估由MergeRepair生成的适配器在APR任务中的表现。特别是，在所有三个技术中，我们将采用两个主要的融合场景来合并所有适配器，包括：（i）根据不同适配器参数的不同权重均匀加权，其中所有适配器的重要性相同；（ii）我们提出的连续融合法，即我们在序列上依次合并任务特定的适配器，并且合并后的适配器顺序和权重至关重要。通过探索合并技术的研究，我们将调查合并适配器改进及其通用性的能力。通过连续融合，我们将探索合并适配器的能力以及任务顺序在实际世界软件项目中发生的效应。

---

## Antidote_ Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09600v1)

### Abstract (English)

Safety aligned Large Language Models (LLMs) are vulnerable to harmful
fine-tuning attacks \cite{qi2023fine}-- a few harmful data mixed in the
fine-tuning dataset can break the LLMs's safety alignment. Existing mitigation
strategies include alignment stage solutions \cite{huang2024vaccine,
rosati2024representation} and fine-tuning stage solutions
\cite{huang2024lazy,mukhoti2023fine}. However, our evaluation shows that both
categories of defenses fail \textit{when some specific training
hyper-parameters are chosen} -- a large learning rate or a large number of
training epochs in the fine-tuning stage can easily invalidate the defense,
which however, is necessary to guarantee finetune performance. To this end, we
propose Antidote, a post-fine-tuning stage solution, which remains
\textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning
stage}}. Antidote relies on the philosophy that by removing the harmful
parameters, the harmful model can be recovered from the harmful behaviors,
regardless of how those harmful parameters are formed in the fine-tuning stage.
With this philosophy, we introduce a one-shot pruning stage after harmful
fine-tuning to remove the harmful weights that are responsible for the
generation of harmful content. Despite its embarrassing simplicity, empirical
results show that Antidote can reduce harmful score while maintaining accuracy
on downstream tasks.

### 摘要 (中文)

安全对齐大型语言模型（LLM）（以下简称“大模型”）在有害微调攻击中是脆弱的。有害数据混入微调集可以破坏大模型的安全对齐。现有防御策略包括阶段解决方案，如《黄浩博等：疫苗式对抗与微调阶段方案》和《墨霍蒂等：懒惰式对抗与微调阶段方案》，以及阶段解决方案，《胡光等：疫苗式对抗与微调阶段方案》。然而，我们的评估表明，这两种类型的防御策略在某些特定的训练超参数选择时都会失败——在微调阶段的大学习率或大量训练周期很容易破坏防御，但这是必要的以保证微调性能。为此，我们提出了一种后微调阶段解决方案，即Antidote。它对阶段中的有害微调不敏感。AntiDote基于一个哲学，即通过移除有害参数，可以从有害行为中恢复有害模型，无论这些有害参数是在微调阶段如何形成的。凭借这个哲学，我们在有害微调后引入了一个一次性剪枝阶段，在有害微调后移除导致有害内容生成的责任权重。尽管它的简单性令人尴尬，但实验证明，AntiDote可以在保持下游任务准确率的同时降低有害分数。

---

## How to Make the Most of LLMs_ Grammatical Knowledge for Acceptability Judgments

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09639v1)

### Abstract (English)

The grammatical knowledge of language models (LMs) is often measured using a
benchmark of linguistic minimal pairs, where LMs are presented with a pair of
acceptable and unacceptable sentences and required to judge which is
acceptable. The existing dominant approach, however, naively calculates and
compares the probabilities of paired sentences using LMs. Additionally, large
language models (LLMs) have yet to be thoroughly examined in this field. We
thus investigate how to make the most of LLMs' grammatical knowledge to
comprehensively evaluate it. Through extensive experiments of nine judgment
methods in English and Chinese, we demonstrate that a probability readout
method, in-template LP, and a prompting-based method, Yes/No probability
computing, achieve particularly high performance, surpassing the conventional
approach. Our analysis reveals their different strengths, e.g., Yes/No
probability computing is robust against token-length bias, suggesting that they
harness different aspects of LLMs' grammatical knowledge. Consequently, we
recommend using diverse judgment methods to evaluate LLMs comprehensively.

### 摘要 (中文)

语言模型（LM）的语法知识通常使用语义最小对（SLD）来衡量，其中LM被提供一组可接受和不可接受的句子，并要求判断哪一个是可接受的。然而，现有的主导方法仅粗略地计算并比较LM之间的配对句的概率。此外，在这个领域中，大型语言模型（LLM）尚未得到充分研究。因此，我们调查如何充分利用LLMs的语法知识以全面评估它。通过在英语和汉语中的九种判别方法的大量实验，我们展示了概率读取方法、内模板LP以及基于提问的计算方法——是或否概率计算——取得了特别高的性能，超越了传统的方法。我们的分析揭示了它们的不同优势，例如是或否概率计算可以抵抗词长偏见，表明它们利用了不同方面的LLMs的语法知识。因此，我们建议采用多种判别方法来全面评估LMs。

---

## Debiased Contrastive Representation Learning for Mitigating Dual Biases in Recommender Systems

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09646v1)

### Abstract (English)

In recommender systems, popularity and conformity biases undermine
recommender effectiveness by disproportionately favouring popular items,
leading to their over-representation in recommendation lists and causing an
unbalanced distribution of user-item historical data. We construct a causal
graph to address both biases and describe the abstract data generation
mechanism. Then, we use it as a guide to develop a novel Debiased Contrastive
Learning framework for Mitigating Dual Biases, called DCLMDB. In DCLMDB, both
popularity bias and conformity bias are handled in the model training process
by contrastive learning to ensure that user choices and recommended items are
not unduly influenced by conformity and popularity. Extensive experiments on
two real-world datasets, Movielens-10M and Netflix, show that DCLMDB can
effectively reduce the dual biases, as well as significantly enhance the
accuracy and diversity of recommendations.

### 摘要 (中文)

在推荐系统中，流行偏好和趋同偏见削弱了推荐系统的有效性，它们倾向于优先推荐热门内容，导致他们在推荐列表中的过量出现，并且用户-物品的历史数据的不平衡分布。我们构建了一个因果图来应对这两种偏差，并描述了抽象数据生成机制。然后，它作为指导用于开发一种名为Debiased Contrastive Learning框架的新方法，以抵消双偏见，称为DCLMDB。在DCLMDB中，通过对比学习处理模型训练过程中的流行偏好和趋同偏好，确保用户的决策和推荐的项目不受趋同和流行的影响过大。对两个真实世界数据集Movielens-10M和Netflix进行的广泛实验表明，DCLMDB有效地减少了双偏见，并显著提高了推荐准确性和多样性。

---

## Deep Learning-based Machine Condition Diagnosis using Short-time Fourier Transformation Variants

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09649v1)

### Abstract (English)

In motor condition diagnosis, electrical current signature serves as an
alternative feature to vibration-based sensor data, which is a more expensive
and invasive method. Machine learning (ML) techniques have been emerging in
diagnosing motor conditions using only motor phase current signals. This study
converts time-series motor current signals to time-frequency 2D plots using
Short-time Fourier Transform (STFT) methods. The motor current signal dataset
consists of 3,750 sample points with five classes - one healthy and four
synthetically-applied motor fault conditions, and with five loading conditions:
0, 25, 50, 75, and 100%. Five transformation methods are used on the dataset:
non-overlap and overlap STFTs, non-overlap and overlap realigned STFTs, and
synchrosqueezed STFT. Then, deep learning (DL) models based on the previous
Convolutional Neural Network (CNN) architecture are trained and validated from
generated plots of each method. The DL models of overlap-STFT, overlap R-STFT,
non-overlap STFT, non-overlap R-STFT, and synchrosqueezed-STFT performed
exceptionally with an average accuracy of 97.65, 96.03, 96.08, 96.32, and
88.27%, respectively. Four methods outperformed the previous best ML method
with 93.20% accuracy, while all five outperformed previous 2D-plot-based
methods with accuracy of 80.25, 74.80, and 82.80%, respectively, using the same
dataset, same DL architecture, and validation steps.

### 摘要 (中文)

在电机状态诊断中，电电流签名是振动传感器数据的替代特征，这是一种更昂贵和侵入性的方法。机器学习（ML）技术已经涌现出来，仅使用电机相电流信号来诊断电机状态。本研究通过短时间傅里叶变换（STFT）方法将时间序列电机电流信号转换成二维时频图。电机电流信号数据集由3750个采样点组成，分为五类：一种健康状况和四种合成应用的电机故障条件，以及五个负载条件：0、25、50、75和100%。对数据集进行了5种变换方法：非重叠和重叠STFTs，非重叠和重叠重新排列STFTs，同步压缩STFT。然后，根据生成的每个方法的每种方法的深度学习（DL）模型进行训练和验证。所有重叠STFT、重叠R-STFT、非重叠STFT、非重叠R-STFT和同步压缩STFT的表现异常良好，平均准确率分别为97.65%，96.03%，96.08%，96.32%和88.27%。四个方法优于之前最佳的ML方法，其准确率为93.20%，而所有五个都优于基于相同数据集、相同DL架构和验证步骤的2D-图方法，其准确率分别为80.25%，74.80%和82.80%。

---

## Data-driven Conditional Instrumental Variables for Debiasing Recommender Systems

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09651v1)

### Abstract (English)

In recommender systems, latent variables can cause user-item interaction data
to deviate from true user preferences. This biased data is then used to train
recommendation models, further amplifying the bias and ultimately compromising
both recommendation accuracy and user satisfaction. Instrumental Variable (IV)
methods are effective tools for addressing the confounding bias introduced by
latent variables; however, identifying a valid IV is often challenging. To
overcome this issue, we propose a novel data-driven conditional IV (CIV)
debiasing method for recommender systems, called CIV4Rec. CIV4Rec automatically
generates valid CIVs and their corresponding conditioning sets directly from
interaction data, significantly reducing the complexity of IV selection while
effectively mitigating the confounding bias caused by latent variables in
recommender systems. Specifically, CIV4Rec leverages a variational autoencoder
(VAE) to generate the representations of the CIV and its conditional set from
interaction data, followed by the application of least squares to derive causal
representations for click prediction. Extensive experiments on two real-world
datasets, Movielens-10M and Douban-Movie, demonstrate that our CIV4Rec
successfully identifies valid CIVs, effectively reduces bias, and consequently
improves recommendation accuracy.

### 摘要 (中文)

在推荐系统中，隐变量会导致用户-物品交互数据偏离真实用户偏好。这些偏倚的数据随后被用于训练推荐模型，进一步放大偏见，并最终损害推荐准确性和用户体验。无偏估计（IV）方法是解决由隐变量引入的混淆性偏见的有效工具；然而，识别有效的IV通常是具有挑战性的。为了解决这一问题，我们提出了一种新的基于条件无偏估计（CIV）去污方法，称为CIV4Rec。CIV4Rec自动从交互数据直接生成有效的CIV及其对应条件集，显著减少了IV选择的复杂性，同时有效地缓解了由隐变量引起的推荐系统的混淆性偏见。具体而言，CIV4Rec利用变分自编码器（VAE）来从交互数据生成CIV和其条件集的表示，然后应用最小二乘法来获得点击预测的因果表示。对两个实际世界数据集Movielens-10M和Douban-Movie进行广泛的实验表明，我们的CIV4Rec成功地检测到有效的CIV，有效减少偏见，并且因此提高推荐准确性。

---

## Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09698v1)

### Abstract (English)

Recent advances in Large Language Models (LLMs) have demonstrated significant
potential in the field of Recommendation Systems (RSs). Most existing studies
have focused on converting user behavior logs into textual prompts and
leveraging techniques such as prompt tuning to enable LLMs for recommendation
tasks. Meanwhile, research interest has recently grown in multimodal
recommendation systems that integrate data from images, text, and other sources
using modality fusion techniques. This introduces new challenges to the
existing LLM-based recommendation paradigm which relies solely on text modality
information. Moreover, although Multimodal Large Language Models (MLLMs)
capable of processing multi-modal inputs have emerged, how to equip MLLMs with
multi-modal recommendation capabilities remains largely unexplored. To this
end, in this paper, we propose the Multimodal Large Language Model-enhanced
Sequential Multimodal Recommendation (MLLM-MSR) model. To capture the dynamic
user preference, we design a two-stage user preference summarization method.
Specifically, we first utilize an MLLM-based item-summarizer to extract image
feature given an item and convert the image into text. Then, we employ a
recurrent user preference summarization generation paradigm to capture the
dynamic changes in user preferences based on an LLM-based user-summarizer.
Finally, to enable the MLLM for multi-modal recommendation task, we propose to
fine-tune a MLLM-based recommender using Supervised Fine-Tuning (SFT)
techniques. Extensive evaluations across various datasets validate the
effectiveness of MLLM-MSR, showcasing its superior ability to capture and adapt
to the evolving dynamics of user preferences.

### 摘要 (中文)

最近，大型语言模型（LLMs）在推荐系统（RS）领域的进步展示了巨大的潜力。目前的研究主要集中在将用户行为日志转换成文本提示，并利用诸如调优提示的技术来使LLMs适用于推荐任务。与此同时，研究兴趣最近对融合图像、文本和其他来源的多模态推荐系统产生了浓厚的兴趣。这引入了现有基于文本模式的信息依赖型推荐范式的新的挑战。此外，虽然具有处理多种模态输入能力的多模态大型语言模型（MLLMs）已经出现，如何为MLLM配备多模态推荐功能仍然是未被探索的。为此，在本文中，我们提出了增强顺序多模态推荐（MLLM-MSR）模型。为了捕捉动态的用户偏好，我们设计了一个两阶段的用户偏好总结方法。具体来说，首先，我们使用一个基于MLLM的项目摘要器提取给定项目的图像特征并将图像转换为文本。然后，我们采用基于LLM的用户摘要生成范式捕获根据LMM的用户摘要的变化的动态变化。最后，为了使MLLM用于多模态推荐任务，我们提出使用监督微调（SFT）技术对MLLM进行再训练。广泛的评估数据集验证了MLLM-MSR的有效性，显示了它优于捕捉和适应用户偏好的不断演变的动力学的能力。

---

## Revisiting Reciprocal Recommender Systems_ Metrics_ Formulation_ and Method

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09748v1)

### Abstract (English)

Reciprocal recommender systems~(RRS), conducting bilateral recommendations
between two involved parties, have gained increasing attention for enhancing
matching efficiency. However, the majority of existing methods in the
literature still reuse conventional ranking metrics to separately assess the
performance on each side of the recommendation process. These methods overlook
the fact that the ranking outcomes of both sides collectively influence the
effectiveness of the RRS, neglecting the necessity of a more holistic
evaluation and a capable systemic solution.
  In this paper, we systemically revisit the task of reciprocal recommendation,
by introducing the new metrics, formulation, and method. Firstly, we propose
five new evaluation metrics that comprehensively and accurately assess the
performance of RRS from three distinct perspectives: overall coverage,
bilateral stability, and balanced ranking. These metrics provide a more
holistic understanding of the system's effectiveness and enable a comprehensive
evaluation. Furthermore, we formulate the RRS from a causal perspective,
formulating recommendations as bilateral interventions, which can better model
the decoupled effects of potential influencing factors. By utilizing the
potential outcome framework, we further develop a model-agnostic causal
reciprocal recommendation method that considers the causal effects of
recommendations. Additionally, we introduce a reranking strategy to maximize
matching outcomes, as measured by the proposed metrics. Extensive experiments
on two real-world datasets from recruitment and dating scenarios demonstrate
the effectiveness of our proposed metrics and approach. The code and dataset
are available at: https://github.com/RUCAIBox/CRRS.

### 摘要 (中文)

互惠推荐系统（RRS）~（RRS），在参与双方之间进行双边推荐，已引起越来越多的关注以提高匹配效率。然而，在现有文献中，大多数方法仍然仅从每个方面单独评估推荐过程的性能。这些方法忽略了两个方面之间的排名结果共同影响RSS效果的事实，忽视了构建一个更全面的评价和有能力的整体解决方案的必要性。
在此文中，我们系统地重新审视互惠推荐任务，通过引入新的指标、框架和方法来提出新见解。首先，我们提出了五个全新的评估指标，全面而准确地评估RSS从三个不同视角的效果：总体覆盖、双边稳定性以及平衡的排名。这些指标提供了对系统的有效性更为全面的理解，并使可以进行全面的评估。此外，我们将RSS从因果角度来考虑，将建议视为双边干预，这更好地模型可能影响因素的分离效应。通过潜在结果框架，我们进一步发展了一个无需模型特定因果互惠推荐方法，该方法考虑了推荐的影响。此外，我们还介绍了一种重排序策略，以最大化根据所提出的指标测量的结果的匹配度。对于招聘和约会场景的两个真实世界数据集的实验展示了我们提出指标和方法的有效性。代码和数据集可在https://github.com/RUCAIBox/CRRS找到。

---

## AutoML-guided Fusion of Entity and LLM-based representations

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09794v1)

### Abstract (English)

Large semantic knowledge bases are grounded in factual knowledge. However,
recent approaches to dense text representations (embeddings) do not efficiently
exploit these resources. Dense and robust representations of documents are
essential for effectively solving downstream classification and retrieval
tasks. This work demonstrates that injecting embedded information from
knowledge bases can augment the performance of contemporary Large Language
Model (LLM)-based representations for the task of text classification. Further,
by considering automated machine learning (AutoML) with the fused
representation space, we demonstrate it is possible to improve classification
accuracy even if we use low-dimensional projections of the original
representation space obtained via efficient matrix factorization. This result
shows that significantly faster classifiers can be achieved with minimal or no
loss in predictive performance, as demonstrated using five strong LLM baselines
on six diverse real-life datasets.

### 摘要 (中文)

大型语义知识库依赖于事实性知识。然而，最近对密集文本表示（嵌入）方法并未有效利用这些资源。对于有效的下游分类和检索任务，文档的密集且强大的表示是必要的。本工作展示了从知识库中注入嵌入信息可以增强当前基于语言模型（LLM）的表示在文本分类任务上的性能。此外，通过融合表示空间考虑自动机器学习（AutoML），我们演示了即使使用由高效矩阵分解获取的原始表示空间低维投影也可以改善分类准确性。这一结果表明，在不损失预测性能的情况下，可以通过使用五种强的语言模型基线在六个多样化的实际数据集上实现显著更快的分类器。

---

## Contextual Dual Learning Algorithm with Listwise Distillation for Unbiased Learning to Rank

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09817v1)

### Abstract (English)

Unbiased Learning to Rank (ULTR) aims to leverage biased implicit user
feedback (e.g., click) to optimize an unbiased ranking model. The effectiveness
of the existing ULTR methods has primarily been validated on synthetic
datasets. However, their performance on real-world click data remains unclear.
Recently, Baidu released a large publicly available dataset of their web search
logs. Subsequently, the NTCIR-17 ULTRE-2 task released a subset dataset
extracted from it. We conduct experiments on commonly used or effective ULTR
methods on this subset to determine whether they maintain their effectiveness.
In this paper, we propose a Contextual Dual Learning Algorithm with Listwise
Distillation (CDLA-LD) to simultaneously address both position bias and
contextual bias. We utilize a listwise-input ranking model to obtain
reconstructed feature vectors incorporating local contextual information and
employ the Dual Learning Algorithm (DLA) method to jointly train this ranking
model and a propensity model to address position bias. As this ranking model
learns the interaction information within the documents list of the training
set, to enhance the ranking model's generalization ability, we additionally
train a pointwise-input ranking model to learn the listwise-input ranking
model's capability for relevance judgment in a listwise manner. Extensive
experiments and analysis confirm the effectiveness of our approach.

### 摘要 (中文)

无偏学习到排序（ULTR）的目标是利用有偏的隐式用户反馈（例如点击），以优化一个无偏的排名模型。现有ULTR方法的有效性主要通过合成数据集进行验证。然而，它们在真实世界中点击数据上的表现仍然不清楚。最近，百度发布了一个大公开可用的网页搜索日志的数据集。随后，NTCIR-17 ULTRE-2任务从该数据集中提取了一部分数据集。我们将在这一小部分上对常用或有效的方法进行实验，以确定它们是否保持其有效性。

在这篇论文中，我们提出了一种上下文双学习算法列表分层蒸馏（CDLA-LD）来同时解决位置偏见和上下文偏见。我们将使用列表输入的排名模型获得包含本地语境信息的重构特征向量，并使用双学习算法（DLA）方法联合训练这个排名模型和一个倾向模型来共同解决位置偏见。由于这种排名模型学习了训练集中的文档列表内的交互信息，为了增强排名模型的一般化能力，我们还进一步训练一个点输入的排名模型，在列表方式下学习列表输入排名模型的相关判断能力。广泛的实验和分析证实了我们的方法的有效性。

---

## CMoralEval_ A Moral Evaluation Benchmark for Chinese Large Language Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09819v1)

### Abstract (English)

What a large language model (LLM) would respond in ethically relevant
context? In this paper, we curate a large benchmark CMoralEval for morality
evaluation of Chinese LLMs. The data sources of CMoralEval are two-fold: 1) a
Chinese TV program discussing Chinese moral norms with stories from the society
and 2) a collection of Chinese moral anomies from various newspapers and
academic papers on morality. With these sources, we aim to create a moral
evaluation dataset characterized by diversity and authenticity. We develop a
morality taxonomy and a set of fundamental moral principles that are not only
rooted in traditional Chinese culture but also consistent with contemporary
societal norms. To facilitate efficient construction and annotation of
instances in CMoralEval, we establish a platform with AI-assisted instance
generation to streamline the annotation process. These help us curate
CMoralEval that encompasses both explicit moral scenarios (14,964 instances)
and moral dilemma scenarios (15,424 instances), each with instances from
different data sources. We conduct extensive experiments with CMoralEval to
examine a variety of Chinese LLMs. Experiment results demonstrate that
CMoralEval is a challenging benchmark for Chinese LLMs. The dataset is publicly
available at \url{https://github.com/tjunlp-lab/CMoralEval}.

### 摘要 (中文)

一个大型语言模型（LLM）在伦理相关语境下会如何回答？在这篇论文中，我们收集了中国大型基准的CMoralEval数据集用于道德评价评估。数据源有两个方面：一是讨论社会故事与传统道德规范的中国电视节目；二是来自各种报纸和学术论文的中国道德谬误集合。通过这些来源，我们的目标是创建一个具有多样性和真实性的道德评价数据集。我们开发了一个道德分类体系和一些基本道德原则，这些不仅植根于中国传统文化，而且与当代社会规范保持一致。为了促进CMoralEval实例高效构建和标注过程，我们建立了一个有AI辅助生成实例平台来简化注释过程。这些帮助我们筛选出涵盖明确道德场景（14,964个实例）和道德困境场景（15,424个实例），每个数据源都有不同的实例。我们在CMoralEval上进行了广泛的实验来检验多种中国LMM。实验结果表明，CMoralEval是一个对中国LMM极具挑战性的基准测试。该数据集已公开发布在GitHub上的地址为https://github.com/tjunlp-lab/CMoralEval。

---

## Importance Weighting Can Help Large Language Models Self-Improve

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09849v1)

### Abstract (English)

Large language models (LLMs) have shown remarkable capability in numerous
tasks and applications. However, fine-tuning LLMs using high-quality datasets
under external supervision remains prohibitively expensive. In response, LLM
self-improvement approaches have been vibrantly developed recently. The typical
paradigm of LLM self-improvement involves training LLM on self-generated data,
part of which may be detrimental and should be filtered out due to the unstable
data quality. While current works primarily employs filtering strategies based
on answer correctness, in this paper, we demonstrate that filtering out correct
but with high distribution shift extent (DSE) samples could also benefit the
results of self-improvement. Given that the actual sample distribution is
usually inaccessible, we propose a new metric called DS weight to approximate
DSE, inspired by the Importance Weighting methods. Consequently, we integrate
DS weight with self-consistency to comprehensively filter the self-generated
samples and fine-tune the language model. Experiments show that with only a
tiny valid set (up to 5\% size of the training set) to compute DS weight, our
approach can notably promote the reasoning ability of current LLM
self-improvement methods. The resulting performance is on par with methods that
rely on external supervision from pre-trained reward models.

### 摘要 (中文)

大型语言模型（LLM）在众多任务和应用中表现出非凡的能力。然而，使用高质量数据集进行外源监督下的自优化仍然非常昂贵。因此，最近发展了一系列改进方法来提高LLM的自我提升能力。典型的方法是训练LLM使用自动生成的数据的一部分，其中一部分可能是有害的，并且应该通过不稳定的数据质量过滤出来。尽管当前的工作主要基于答案正确性进行过滤策略，本文展示了通过计算分布转移程度（DSE），即高度不稳定的样本也可以对自我提升的结果产生积极影响。由于实际样本分布通常不可访问，我们提出了一个新的指标称为DS权重，该指标由重要性加权方法启发而来。因此，我们将DS权重与一致性相结合，全面地过滤生成的样本并微调语言模型。实验结果显示，在仅计算DS权重所需的最小有效集（不超过训练集大小的5％）的情况下，我们的方法可以显著促进当前自优化方法的推理能力。结果性能与依赖于预训练奖励模型外部监督的方法相同。

---

## Self-Directed Turing Test for Large Language Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09853v1)

### Abstract (English)

The Turing test examines whether AIs can exhibit human-like behaviour in
natural language conversations. Traditional Turing tests adopt a rigid dialogue
format where each participant sends only one message each time and require
continuous human involvement to direct the entire interaction with the test
subject. This fails to reflect a natural conversational style and hinders the
evaluation of Large Language Models (LLMs) in complex and prolonged dialogues.
This paper proposes the Self-Directed Turing Test, which extends the original
test with a burst dialogue format, allowing more dynamic exchanges by multiple
consecutive messages. It further efficiently reduces human workload by having
the LLM self-direct the majority of the test process, iteratively generating
dialogues that simulate its interaction with humans. With the pseudo-dialogue
history, the model then engages in a shorter dialogue with a human, which is
paired with a human-human conversation on the same topic to be judged using
questionnaires. We introduce the X-Turn Pass-Rate metric to assess the human
likeness of LLMs across varying durations. While LLMs like GPT-4 initially
perform well, achieving pass rates of 51.9% and 38.9% during 3 turns and 10
turns of dialogues respectively, their performance drops as the dialogue
progresses, which underscores the difficulty in maintaining consistency in the
long term.

### 摘要 (中文)

图灵测试旨在评估AI在自然语言对话中是否能表现出人类般的行为。传统的图灵测试采用固定对话格式，其中每次参与者发送的信息都是一样的，并需要持续的人类介入来引导整个交互与测试对象。这不符合自然的会话风格，也无法有效地评估大规模语言模型（LLM）在复杂和长时间的对话中的表现。

本文提出自导图灵测试，它扩展了原始测试，增加了突发对话格式，允许更多的动态交流通过多次连续消息。此外，它还可以有效减少人类的工作量，让大型语言模型自动主导大部分测试过程，不断生成模拟其与人类互动的对话。通过伪对话历史，该模型然后与一个人类进行短对话，配对在同一主题上的人类-人对话，用于使用问卷进行评判。我们引入X转过率指标来评估不同长度下LLMs的人性相似度。虽然GPT-4等初始模型在3轮和10轮对话中分别达到51.9%和38.9%的通过率，但随着对话的进展，他们的性能开始下降，这说明很难在长期保持一致性。

---

## TeamLoRA_ Boosting Low-Rank Adaptation with Expert Collaboration and Competition

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09856v1)

### Abstract (English)

While Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA have
effectively addressed GPU memory constraints during fine-tuning, their
performance often falls short, especially in multidimensional task scenarios.
To address this issue, one straightforward solution is to introduce
task-specific LoRA modules as domain experts, leveraging the modeling of
multiple experts' capabilities and thus enhancing the general capability of
multi-task learning. Despite promising, these additional components often add
complexity to the training and inference process, contravening the efficient
characterization of PEFT designed for. Considering this, we introduce an
innovative PEFT method, TeamLoRA, consisting of a collaboration and competition
module for experts, and thus achieving the right balance of effectiveness and
efficiency: (i) For collaboration, a novel knowledge-sharing and -organizing
mechanism is devised to appropriately reduce the scale of matrix operations,
thereby boosting the training and inference speed. (ii) For competition, we
propose leveraging a game-theoretic interaction mechanism for experts,
encouraging experts to transfer their domain-specific knowledge while facing
diverse downstream tasks, and thus enhancing the performance. By doing so,
TeamLoRA elegantly connects the experts as a "Team" with internal collaboration
and competition, enabling a faster and more accurate PEFT paradigm for
multi-task learning. To validate the superiority of TeamLoRA, we curate a
comprehensive multi-task evaluation(CME) benchmark to thoroughly assess the
capability of multi-task learning. Experiments conducted on our CME and other
benchmarks indicate the effectiveness and efficiency of TeamLoRA. Our project
is available at https://github.com/Lin-Tianwei/TeamLoRA.

### 摘要 (中文)

虽然像LoRA这样的参数效率微调（PEFT）方法在微调时有效地解决了GPU内存约束问题，但它们的性能往往无法满足多维任务场景。为了应对这一问题，一种直接的方法是引入任务特定的LoRA模块作为专家，利用模型中的多个专家能力进行建模，从而增强多任务学习的一般能力。尽管有希望，但这些附加组件通常会增加训练和推理过程的复杂性，违背了设计用于高效处理PEFT的目标。考虑到这一点，我们提出了一个创新的PEFT方法，即团队LoRA，它由合作与竞争模块组成，因此实现了有效性和效率的最佳平衡：(i)对于合作，我们提出了一种新型的知识共享和组织机制，以适当减少矩阵操作的规模，从而提高训练和推理速度。(ii)对于竞争，我们提出了利用博弈论交互机制来鼓励专家之间在面对多样化的下游任务时相互转移自己的专长知识，并且以此来增强表现。通过这样做，团队LoRA优雅地将专家们作为一个“团队”连接起来，内部协作和竞争，使多任务学习的PEFT模式变得更快更准确。为了验证TeamLoRA的优势，我们创建了一个全面的多任务评估(CME)基准来全面评估多任务学习的能力。在我们的CME和其他基准上进行的实验表明，TeamLoRA的有效性和效率。我们的项目可用https://github.com/Lin-Tianwei/TeamLoRA。

---

## Benchmarking LLMs for Translating Classical Chinese Poetry_Evaluating Adequacy_ Fluency_ and Elegance

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09945v1)

### Abstract (English)

Large language models (LLMs) have shown remarkable performance in general
translation tasks. However, the increasing demand for high-quality translations
that are not only adequate but also fluent and elegant. To assess the extent to
which current LLMs can meet these demands, we introduce a suitable benchmark
for translating classical Chinese poetry into English. This task requires not
only adequacy in translating culturally and historically significant content
but also a strict adherence to linguistic fluency and poetic elegance. Our
study reveals that existing LLMs fall short of this task. To address these
issues, we propose RAT, a \textbf{R}etrieval-\textbf{A}ugmented machine
\textbf{T}ranslation method that enhances the translation process by
incorporating knowledge related to classical poetry. Additionally, we propose
an automatic evaluation metric based on GPT-4, which better assesses
translation quality in terms of adequacy, fluency, and elegance, overcoming the
limitations of traditional metrics. Our dataset and code will be made
available.

### 摘要 (中文)

大型语言模型（LLM）在一般翻译任务中表现出色。然而，随着对高质量翻译需求的增加，这些翻译不仅要足够好，而且还要流畅优美。为了评估现有LLM在满足这些要求方面的能力，我们引入了一个适合古典中国诗歌到英语的翻译基准。这项任务不仅要求能够准确地翻译具有文化历史意义的内容，而且还必须严格遵守语义流利和诗意之美。我们的研究发现，现有的LLM无法满足这一要求。为了解决这些问题，我们提出了一种新的方法——RAT，一种检索增强机器翻译方法，它通过结合与古典诗歌相关的知识来增强翻译过程。此外，我们还提出了一个基于GPT-4的自动评价指标，该指标更全面地评估了翻译的质量，包括准确性、流畅性和诗意美，克服了传统指标的局限性。我们的数据集和代码将会公开。

---

## Microscopic Analysis on LLM players via Social Deduction Game

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09946v1)

### Abstract (English)

Recent studies have begun developing autonomous game players for social
deduction games using large language models (LLMs). When building LLM players,
fine-grained evaluations are crucial for addressing weaknesses in game-playing
abilities. However, existing studies have often overlooked such assessments.
Specifically, we point out two issues with the evaluation methods employed.
First, game-playing abilities have typically been assessed through game-level
outcomes rather than specific event-level skills; Second, error analyses have
lacked structured methodologies. To address these issues, we propose an
approach utilizing a variant of the SpyFall game, named SpyGame. We conducted
an experiment with four LLMs, analyzing their gameplay behavior in SpyGame both
quantitatively and qualitatively. For the quantitative analysis, we introduced
eight metrics to resolve the first issue, revealing that these metrics are more
effective than existing ones for evaluating the two critical skills: intent
identification and camouflage. In the qualitative analysis, we performed
thematic analysis to resolve the second issue. This analysis identifies four
major categories that affect gameplay of LLMs. Additionally, we demonstrate how
these categories complement and support the findings from the quantitative
analysis.

### 摘要 (中文)

最近的研究已经开始使用大型语言模型（LLM）开发自主游戏玩家，用于社会推理游戏。在构建LMM玩家时，对游戏能力的细粒度评估对于识别弱点至关重要。然而，现有的研究往往忽视了这样的评估。具体来说，我们指出现有方法中两个问题。

首先，通常通过游戏级别的结果来评估游戏能力，而不是特定事件级技能；其次，错误分析缺乏结构化的方法。为了应对这些问题，我们提出了一个利用变体SpyFall游戏名称为SpyGame的方法。我们在四个LMM上进行了实验，并对其在SpyGame中的行为进行量化和定性分析。对于定量分析，我们引入了八个指标来解决第一个问题，发现这些指标比现有方法更有效，以评价两个关键技能：意图识别和伪装。在定性分析中，我们执行主题分析来解决第二个问题。这项分析确定了影响LLM游戏的关键类别。此外，我们演示了如何这些类别与定量分析的结果相结合和支持。

总的来说，我们的研究表明，尽管当前的研究已经取得了一些进展，但仍然存在一些不足之处。因此，我们需要进一步研究并提出有效的解决方案来解决这些问题。

---

## Fiber Transmission Model with Parameterized Inputs based on GPT-PINN Neural Network

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09947v1)

### Abstract (English)

In this manuscript, a novelty principle driven fiber transmission model for
short-distance transmission with parameterized inputs is put forward. By taking
into the account of the previously proposed principle driven fiber model, the
reduced basis expansion method and transforming the parameterized inputs into
parameterized coefficients of the Nonlinear Schrodinger Equations, universal
solutions with respect to inputs corresponding to different bit rates can all
be obtained without the need of re-training the whole model. This model, once
adopted, can have prominent advantages in both computation efficiency and
physical background. Besides, this model can still be effectively trained
without the needs of transmitted signals collected in advance. Tasks of on-off
keying signals with bit rates ranging from 2Gbps to 50Gbps are adopted to
demonstrate the fidelity of the model.

### 摘要 (中文)

在这一手稿中，提出了一种驱动原则的光纤传输模型，该模型考虑了先前提出的驱动原理，通过将参数化输入转换为非线性方程中的参数化系数来减少基扩张方法，可以无须重新训练整个模型地获得针对不同比特率对应的通用解。一旦采用此模型，它可以在计算效率和物理背景方面具有显著优势。此外，即使不预先收集发送信号，也可以有效地对该模型进行训练。使用从2Gbps到50Gbps的不同比特率的开关键信号任务来展示模型的准确性。

---

## Principle Driven Parameterized Fiber Model based on GPT-PINN Neural Network

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09951v1)

### Abstract (English)

In cater the need of Beyond 5G communications, large numbers of data driven
artificial intelligence based fiber models has been put forward as to utilize
artificial intelligence's regression ability to predict pulse evolution in
fiber transmission at a much faster speed compared with the traditional split
step Fourier method. In order to increase the physical interpretabiliy,
principle driven fiber models have been proposed which inserts the Nonlinear
Schodinger Equation into their loss functions. However, regardless of either
principle driven or data driven models, they need to be re-trained the whole
model under different transmission conditions. Unfortunately, this situation
can be unavoidable when conducting the fiber communication optimization work.
If the scale of different transmission conditions is large, then the whole
model needs to be retrained large numbers of time with relatively large scale
of parameters which may consume higher time costs. Computing efficiency will be
dragged down as well. In order to address this problem, we propose the
principle driven parameterized fiber model in this manuscript. This model
breaks down the predicted NLSE solution with respect to one set of transmission
condition into the linear combination of several eigen solutions which were
outputted by each pre-trained principle driven fiber model via the reduced
basis method. Therefore, the model can greatly alleviate the heavy burden of
re-training since only the linear combination coefficients need to be found
when changing the transmission condition. Not only strong physical
interpretability can the model posses, but also higher computing efficiency can
be obtained. Under the demonstration, the model's computational complexity is
0.0113% of split step Fourier method and 1% of the previously proposed
principle driven fiber model.

### 摘要 (中文)

为了满足Beyond 5G通信的需求，提出了大量的基于人工智能的光纤模型。这些模型利用人工智能的回归能力预测光纤传输脉冲演化的速度远快于传统的分步四元法。为了提高物理可解释性，提出了一种原则驱动光纤模型，在其损失函数中插入非线性方程。然而，无论哪种类型的模型都需要在不同传输条件下重新训练整个模型。不幸的是，在进行光纤通信优化工作时，这种情况是不可避免的。如果不同的传输条件规模较大，则需要大量时间重新训练整个模型，这可能会消耗较高的计算成本。此外，计算效率也会受到影响。为了解决这个问题，本文提出了基于原则驱动参数化光纤模型。该模型通过减少基方法输出每个预训练的原则驱动光纤模型中的几个线性解组合来分解关于特定传输条件的NLSE预测解决方案。因此，当改变传输条件时，只需要找到线性组合系数即可，而不需要重新训练整个模型。不仅强物理可解释性，而且计算效率也可以得到大大提高。在演示中，该模型的计算复杂度仅为分步四元法的0.0113%，以及之前提出的原则驱动光纤模型的1%。

---

## Edge-Cloud Collaborative Motion Planning for Autonomous Driving with Large Language Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09972v1)

### Abstract (English)

Integrating large language models (LLMs) into autonomous driving enhances
personalization and adaptability in open-world scenarios. However, traditional
edge computing models still face significant challenges in processing complex
driving data, particularly regarding real-time performance and system
efficiency. To address these challenges, this study introduces EC-Drive, a
novel edge-cloud collaborative autonomous driving system with data drift
detection capabilities. EC-Drive utilizes drift detection algorithms to
selectively upload critical data, including new obstacles and traffic pattern
changes, to the cloud for processing by GPT-4, while routine data is
efficiently managed by smaller LLMs on edge devices. This approach not only
reduces inference latency but also improves system efficiency by optimizing
communication resource use. Experimental validation confirms the system's
robust processing capabilities and practical applicability in real-world
driving conditions, demonstrating the effectiveness of this edge-cloud
collaboration framework. Our data and system demonstration will be released at
https://sites.google.com/view/ec-drive.

### 摘要 (中文)

将大型语言模型（LLM）集成到自动驾驶中，可以增强开放世界场景中的个性化和适应性。然而，传统的边缘计算模型在处理复杂驾驶数据时仍面临重大挑战，特别是关于实时性能和系统效率的问题。为了应对这些挑战，本研究引入了EC-Drive，这是一种具有数据漂移检测能力的新型边缘-云协作自动驾驶系统。EC-Drive利用漂移检测算法选择性地上传关键数据，包括新障碍物和交通模式变化，将它们发送到云端进行GPT-4处理，而常规数据则由边缘设备上的较小的LLM高效管理。这种方法不仅减少了推理延迟，而且通过优化通信资源使用提高了系统的效率。实验证明了该系统的强大处理能力和实际应用在真实世界驾驶条件下的有效性，展示了这一边缘-云合作框架的有效性。我们的数据和系统演示将在以下链接中发布：https://sites.google.com/view/ec-drive。

---

## Towards a Knowledge Graph for Models and Algorithms in Applied Mathematics

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10003v1)

### Abstract (English)

Mathematical models and algorithms are an essential part of mathematical
research data, as they are epistemically grounding numerical data. In order to
represent models and algorithms as well as their relationship semantically to
make this research data FAIR, two previously distinct ontologies were merged
and extended, becoming a living knowledge graph. The link between the two
ontologies is established by introducing computational tasks, as they occur in
modeling, corresponding to algorithmic tasks. Moreover, controlled vocabularies
are incorporated and a new class, distinguishing base quantities from specific
use case quantities, was introduced. Also, both models and algorithms can now
be enriched with metadata. Subject-specific metadata is particularly relevant
here, such as the symmetry of a matrix or the linearity of a mathematical
model. This is the only way to express specific workflows with concrete models
and algorithms, as the feasible solution algorithm can only be determined if
the mathematical properties of a model are known. We demonstrate this using two
examples from different application areas of applied mathematics. In addition,
we have already integrated over 250 research assets from applied mathematics
into our knowledge graph.

### 摘要 (中文)

数学模型和算法是数学研究数据中不可或缺的一部分，因为它们提供了数理数据的逻辑基础。为了使这些研究数据符合FAIR原则，将模型和算法及其与之相关的语义关系表示出来，需要融合之前分离的两个元知识图谱。通过引入计算任务来建立这两个元知识图谱之间的链接，计算任务对应于在建模过程中出现的任务。此外，还包含控制词汇表，并新增了一个新类别的概念，区分了基本量与特定用例中的量。也使得模型和算法能够被填充上元数据。对于特定的主题，特别相关的是矩阵对称性或数学模型的线性性。这是唯一可以表达具体工作流的方式，因为可行解算法只能确定如果模型具有某种数学性质。我们使用来自应用数学的250多个研究资产已经整合到我们的知识图谱中。

---

## Deterministic Policy Gradient Primal-Dual Methods for Continuous-Space Constrained MDPs

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10015v1)

### Abstract (English)

We study the problem of computing deterministic optimal policies for
constrained Markov decision processes (MDPs) with continuous state and action
spaces, which are widely encountered in constrained dynamical systems.
Designing deterministic policy gradient methods in continuous state and action
spaces is particularly challenging due to the lack of enumerable state-action
pairs and the adoption of deterministic policies, hindering the application of
existing policy gradient methods for constrained MDPs. To this end, we develop
a deterministic policy gradient primal-dual method to find an optimal
deterministic policy with non-asymptotic convergence. Specifically, we leverage
regularization of the Lagrangian of the constrained MDP to propose a
deterministic policy gradient primal-dual (D-PGPD) algorithm that updates the
deterministic policy via a quadratic-regularized gradient ascent step and the
dual variable via a quadratic-regularized gradient descent step. We prove that
the primal-dual iterates of D-PGPD converge at a sub-linear rate to an optimal
regularized primal-dual pair. We instantiate D-PGPD with function approximation
and prove that the primal-dual iterates of D-PGPD converge at a sub-linear rate
to an optimal regularized primal-dual pair, up to a function approximation
error. Furthermore, we demonstrate the effectiveness of our method in two
continuous control problems: robot navigation and fluid control. To the best of
our knowledge, this appears to be the first work that proposes a deterministic
policy search method for continuous-space constrained MDPs.

### 摘要 (中文)

我们研究了计算约束Markov决策过程（MDP）中连续状态和动作空间的确定最优策略的问题，这是在约束动态系统中广泛遇到的问题。由于无法枚举状态-动作对以及采用确定性政策，使得使用现有政策梯度方法来解决约束MDP变得特别具有挑战性。为此，我们开发了一种定性政策梯度主副算法，以找到一个非渐近收敛的确定性政策。具体来说，我们利用约束MDP拉格朗日算子中的正则化来提出一种定性政策梯度主副（D-PGPD）算法，该算法通过更新定性政策的二次正则化梯度上升步长和双变量的二次正则化梯度下降步长来更新定性政策。我们证明了D-PGPD主副迭代的速率小于线性的收敛到优化的正规化主副对。我们用函数逼近实例化D-PGPD，并证明了D-PGPD主副迭代的速率小于线性的收敛到优化的正规化主副对，直到函数逼近误差。此外，我们还演示了我们的方法的有效性两个连续控制问题：机器人导航和流体控制。迄今为止，这似乎是我们提出的第一个用于连续空间约束MDP的定性政策搜索方法。

---

## Envisioning Possibilities and Challenges of AI for Personalized Cancer Care

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10108v1)

### Abstract (English)

The use of Artificial Intelligence (AI) in healthcare, including in caring
for cancer survivors, has gained significant interest. However, gaps remain in
our understanding of how such AI systems can provide care, especially for
ethnic and racial minority groups who continue to face care disparities.
Through interviews with six cancer survivors, we identify critical gaps in
current healthcare systems such as a lack of personalized care and insufficient
cultural and linguistic accommodation. AI, when applied to care, was seen as a
way to address these issues by enabling real-time, culturally aligned, and
linguistically appropriate interactions. We also uncovered concerns about the
implications of AI-driven personalization, such as data privacy, loss of human
touch in caregiving, and the risk of echo chambers that limit exposure to
diverse information. We conclude by discussing the trade-offs between
AI-enhanced personalization and the need for structural changes in healthcare
that go beyond technological solutions, leading us to argue that we should
begin by asking, ``Why personalization?''

### 摘要 (中文)

人工智能在医疗领域的应用，尤其是对癌症康复患者的护理，引起了极大的关注。然而，在如何利用这些AI系统提供护理方面，我们仍然存在理解的缺口，特别是在继续面临医疗保健不平等的少数族裔和种族群体中。

通过与六位癌症康复者进行访谈，我们识别出了当前医疗体系中的关键差距，例如缺乏个性化护理、不足的文化和语言适应性。当AI应用于护理时，人们将其视为解决这些问题的方法，以实现即时、文化一致和语义适当的交互。此外，我们也发现了一些关于AI驱动个人化潜在影响的担忧，如数据隐私问题、照顾过程中失去的人际接触以及可能导致信息偏见限制暴露于多样化信息的风险。最后，我们讨论了通过技术解决方案之外的结构变革来权衡AI增强个性化的利弊，并得出结论，我们应该首先询问“为什么个性化？”。

---

## Rhyme-aware Chinese lyric generator based on GPT

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10130v1)

### Abstract (English)

Neural language representation models such as GPT, pre-trained on large-scale
corpora, can effectively capture rich semantic patterns from plain text and be
fine-tuned to consistently improve natural language generation performance.
However, existing pre-trained language models used to generate lyrics rarely
consider rhyme information, which is crucial in lyrics. Using a pre-trained
model directly results in poor performance. To enhance the rhyming quality of
generated lyrics, we incorporate integrated rhyme information into our model,
thereby improving lyric generation performance.

### 摘要 (中文)

像GPT这样的神经语言表示模型，通过在大型语料库上预训练，可以从纯文本中有效地捕捉到丰富的语义模式，并且可以被调优以持续提高自然语言生成的表现。然而，用于生成歌词的现有预训练语言模型很少考虑押韵信息，这是歌词中至关重要的因素。直接使用预训练模型会得到较差的表现。为了改善生成歌词的押韵质量，我们将集成的押韵信息整合到我们的模型中，从而提高歌词生成的表现。

---

## Customizing Language Models with Instance-wise LoRA for Sequential Recommendation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10159v1)

### Abstract (English)

Sequential recommendation systems predict a user's next item of interest by
analyzing past interactions, aligning recommendations with individual
preferences. Leveraging the strengths of Large Language Models (LLMs) in
knowledge comprehension and reasoning, recent approaches have applied LLMs to
sequential recommendation through language generation paradigms. These methods
convert user behavior sequences into prompts for LLM fine-tuning, utilizing
Low-Rank Adaptation (LoRA) modules to refine recommendations. However, the
uniform application of LoRA across diverse user behaviors sometimes fails to
capture individual variability, leading to suboptimal performance and negative
transfer between disparate sequences. To address these challenges, we propose
Instance-wise LoRA (iLoRA), integrating LoRA with the Mixture of Experts (MoE)
framework. iLoRA creates a diverse array of experts, each capturing specific
aspects of user preferences, and introduces a sequence representation guided
gate function. This gate function processes historical interaction sequences to
generate enriched representations, guiding the gating network to output
customized expert participation weights. This tailored approach mitigates
negative transfer and dynamically adjusts to diverse behavior patterns.
Extensive experiments on three benchmark datasets demonstrate the effectiveness
of iLoRA, highlighting its superior performance compared to existing methods in
capturing user-specific preferences and improving recommendation accuracy.

### 摘要 (中文)

顺序推荐系统通过分析用户的过去交互，以及与个人偏好的匹配来预测用户下一次感兴趣的内容。利用大型语言模型（LLM）在知识理解和推理方面的优势，最近的方法已经将LSTM应用于序列推荐的语言生成范式中。这些方法将用户行为序列转换为LLM微调的提示，利用低秩适应（LoRA）模块进行改进推荐。然而，LoRA在不同用户行为中的统一应用有时会失败，无法捕捉个体差异，导致性能不佳和不同序列之间的负转移。为了应对这些问题，我们提出实例级别的LoRA（iLoRA），将LoRA与混合专家（MoE）框架相结合。iLoRA创建了一组多样化的专家，每个专家都捕获了特定方面的人们偏好，并引入了一个基于序列表示的门控功能。该门控函数处理历史交互序列以生成丰富表示，引导注意力网络输出定制的专家参与权重。这种个性化的方法有效地减少了负转移，并动态地调整到多样化的行为模式。在三个基准数据集上对iLoRA进行了广泛实验，展示了其在捕获用户特定偏好的有效性以及提高推荐精度方面的优越性。

---

## Demystifying the Communication Characteristics for Distributed Transformer Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10197v1)

### Abstract (English)

Deep learning (DL) models based on the transformer architecture have
revolutionized many DL applications such as large language models (LLMs),
vision transformers, audio generation, and time series prediction. Much of this
progress has been fueled by distributed training, yet distributed communication
remains a substantial bottleneck to training progress. This paper examines the
communication behavior of transformer models - that is, how different
parallelism schemes used in multi-node/multi-GPU DL Training communicate data
in the context of transformers. We use GPT-based language models as a case
study of the transformer architecture due to their ubiquity. We validate the
empirical results obtained from our communication logs using analytical models.
At a high level, our analysis reveals a need to optimize small message
point-to-point communication further, correlations between sequence length,
per-GPU throughput, model size, and optimizations used, and where to
potentially guide further optimizations in framework and HPC middleware design
and optimization.

### 摘要 (中文)

基于Transformer架构的深度学习模型已经极大地改变了诸如大型语言模型（LLMs）、视觉变换器、音频生成和时间序列预测等许多深度学习应用。其中大部分进步都得益于分布式训练，但分布式通信仍然是训练进度的一个重要瓶颈。本论文研究了Transformer模型的通信行为——即在多节点/多GPU深度学习训练中，不同并行方案如何在transformer架构下进行数据通信。我们以GPT为基础的语言模型作为案例研究，因为它们很普遍。我们使用我们的通信日志验证从这些分析模型获得的实证结果。从整体上说，我们的分析揭示了一个需要进一步优化小消息点到点通信的需求，序列长度、每块显卡吞吐量、模型大小以及使用的优化之间的相关性，以及框架和HPC中间件设计和优化中的可能方向。

---

## A Likelihood-Free Approach to Goal-Oriented Bayesian Optimal Experimental Design

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09582v1)

### Abstract (English)

Conventional Bayesian optimal experimental design seeks to maximize the
expected information gain (EIG) on model parameters. However, the end goal of
the experiment often is not to learn the model parameters, but to predict
downstream quantities of interest (QoIs) that depend on the learned parameters.
And designs that offer high EIG for parameters may not translate to high EIG
for QoIs. Goal-oriented optimal experimental design (GO-OED) thus directly
targets to maximize the EIG of QoIs.
  We introduce LF-GO-OED (likelihood-free goal-oriented optimal experimental
design), a computational method for conducting GO-OED with nonlinear
observation and prediction models. LF-GO-OED is specifically designed to
accommodate implicit models, where the likelihood is intractable. In
particular, it builds a density ratio estimator from samples generated from
approximate Bayesian computation (ABC), thereby sidestepping the need for
likelihood evaluations or density estimations. The overall method is validated
on benchmark problems with existing methods, and demonstrated on scientific
applications of epidemiology and neural science.

### 摘要 (中文)

常规贝叶斯优化实验设计寻求最大化模型参数的预期信息增益（EIG）。然而，实验的目的并不总是学习模型参数，而是预测对所学参数感兴趣的下游量（QoIs）。而那些提供高EIG参数的设计可能不会转而具有高的EIG QoIs。因此，有目的的优化实验设计（GO-OED）直接目标是最大化QoIs的EIG。

我们引入LF-GO-OED（无概率估计的目标导向优化实验设计），一种用于通过非线性观测和预测模型进行GO-OED的计算方法。LF-GO-OED专门设计用来容纳隐式模型，其中似然不可计算。具体来说，它从来自近似贝叶斯计算（ABC）生成的数据样本中构建密度比估计器，从而绕过需要拟合或密度估计的需求。该总体方法在现有方法上进行了验证，并展示了流行病学和神经科学方面的科学应用。

---

## Predicting path-dependent processes by deep learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09941v1)

### Abstract (English)

In this paper, we investigate a deep learning method for predicting
path-dependent processes based on discretely observed historical information.
This method is implemented by considering the prediction as a nonparametric
regression and obtaining the regression function through simulated samples and
deep neural networks. When applying this method to fractional Brownian motion
and the solutions of some stochastic differential equations driven by it, we
theoretically proved that the $L_2$ errors converge to 0, and we further
discussed the scope of the method. With the frequency of discrete observations
tending to infinity, the predictions based on discrete observations converge to
the predictions based on continuous observations, which implies that we can
make approximations by the method. We apply the method to the fractional
Brownian motion and the fractional Ornstein-Uhlenbeck process as examples.
Comparing the results with the theoretical optimal predictions and taking the
mean square error as a measure, the numerical simulations demonstrate that the
method can generate accurate results. We also analyze the impact of factors
such as prediction period, Hurst index, etc. on the accuracy.

### 摘要 (中文)

在本文中，我们研究了一种基于离散观察历史信息的深度学习方法来预测根据路径依赖过程。这种方法通过考虑预测是一个非参数回归，并通过模拟样本和深度神经网络获得回归函数。当我们将其应用于布朗运动分位数、它驱动的一些随机微分方程的解，理论上证明了$L_2$误差收敛到0，进一步讨论了该方法的范围。随着观测间隔频率趋向无穷大，基于离散观测的结果趋向于基于连续观测的结果，这意味着我们可以利用这个方法进行近似。我们以布朗运动分位数和分形奥本海默过程为例应用该方法。与理论最优预测结果进行比较，采用均方误差作为衡量标准，数值模拟显示，该方法可以生成准确的结果。我们也分析了预测期、Hurst指数等因素对精度的影响。

---

## Unpaired Volumetric Harmonization of Brain MRI with Conditional Latent Diffusion

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09315v1)

### Abstract (English)

Multi-site structural MRI is increasingly used in neuroimaging studies to
diversify subject cohorts. However, combining MR images acquired from various
sites/centers may introduce site-related non-biological variations.
Retrospective image harmonization helps address this issue, but current methods
usually perform harmonization on pre-extracted hand-crafted radiomic features,
limiting downstream applicability. Several image-level approaches focus on 2D
slices, disregarding inherent volumetric information, leading to suboptimal
outcomes. To this end, we propose a novel 3D MRI Harmonization framework
through Conditional Latent Diffusion (HCLD) by explicitly considering image
style and brain anatomy. It comprises a generalizable 3D autoencoder that
encodes and decodes MRIs through a 4D latent space, and a conditional latent
diffusion model that learns the latent distribution and generates harmonized
MRIs with anatomical information from source MRIs while conditioned on target
image style. This enables efficient volume-level MRI harmonization through
latent style translation, without requiring paired images from target and
source domains during training. The HCLD is trained and evaluated on 4,158
T1-weighted brain MRIs from three datasets in three tasks, assessing its
ability to remove site-related variations while retaining essential biological
features. Qualitative and quantitative experiments suggest the effectiveness of
HCLD over several state-of-the-arts

### 摘要 (中文)

多站点结构磁共振成像（MRI）在神经影像学研究中越来越被用于多样化样本组。然而，从多个站点/中心获取的MRI图像结合可能会引入来自不同站点/中心的非生物变异。回顾性图像校准有助于解决这一问题，但当前方法通常仅在提取的手工创建放射性特征上进行校准，限制了下游应用的有效性。一些基于水平的方法专注于二维切片，忽略了内在的体积信息，导致不理想的后果。为此，我们提出了通过条件隐式扩散（HCLD）的新三维MRI校准框架，通过明确考虑图像风格和大脑解剖来考虑图像样式。它包含一个通用化的三维自动编码器，通过4D隐空间编码和解码MRI，以及一种条件隐式扩散模型，该模型学习隐藏分布并生成根据源MRI条件化地带有解剖信息的和谐MRI。这使得可以在无须训练时通过对齐方式通过潜在样式翻译实现高效的体积级MRI校准，而无需在目标和源域之间收集对齐的图像。HCLD在三个任务、三个数据集上的4,158个T1加权脑部MRI上进行了训练和评估，以评估其去除站点相关变异的同时保留基本生物学特征的能力。定量和定性实验表明，HCLD在几个最先进的基础上优于其他方法。

---

## Improving Lung Cancer Diagnosis and Survival Prediction with Deep Learning and CT Imaging

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09367v1)

### Abstract (English)

Lung cancer is a major cause of cancer-related deaths, and early diagnosis
and treatment are crucial for improving patients' survival outcomes. In this
paper, we propose to employ convolutional neural networks to model the
non-linear relationship between the risk of lung cancer and the lungs'
morphology revealed in the CT images. We apply a mini-batched loss that extends
the Cox proportional hazards model to handle the non-convexity induced by
neural networks, which also enables the training of large data sets.
Additionally, we propose to combine mini-batched loss and binary cross-entropy
to predict both lung cancer occurrence and the risk of mortality. Simulation
results demonstrate the effectiveness of both the mini-batched loss with and
without the censoring mechanism, as well as its combination with binary
cross-entropy. We evaluate our approach on the National Lung Screening Trial
data set with several 3D convolutional neural network architectures, achieving
high AUC and C-index scores for lung cancer classification and survival
prediction. These results, obtained from simulations and real data experiments,
highlight the potential of our approach to improving the diagnosis and
treatment of lung cancer.

### 摘要 (中文)

肺癌是导致癌症相关死亡的主要原因之一，早期诊断和治疗对于提高患者生存率至关重要。在本文中，我们提出使用卷积神经网络来模型肺部形态在CT图像中暴露的非线性关系，以处理由深度学习引起的非凸性。此外，我们提出结合mini-batch损失和二元交叉熵来预测肺部癌的发生以及死亡风险。模拟结果表明，mini-batch损失与无截断机制的效果，以及其与二元交叉熵的组合的有效性。我们在几个三维卷积神经网络架构上评估我们的方法，并取得了肺部癌分类和生存预测的高AUC和C指数。这些从模拟和真实数据实验中获得的结果，强调了我们的方法改善肺癌诊断和治疗潜力的可能性。

---

## Flemme_ A Flexible and Modular Learning Platform for Medical Images

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09369v1)

### Abstract (English)

As the rapid development of computer vision and the emergence of powerful
network backbones and architectures, the application of deep learning in
medical imaging has become increasingly significant. Unlike natural images,
medical images lack huge volumes of data but feature more modalities, making it
difficult to train a general model that has satisfactory performance across
various datasets. In practice, practitioners often suffer from manually
creating and testing models combining independent backbones and architectures,
which is a laborious and time-consuming process. We propose Flemme, a FLExible
and Modular learning platform for MEdical images. Our platform separates
encoders from the model architectures so that different models can be
constructed via various combinations of supported encoders and architectures.
We construct encoders using building blocks based on convolution, transformer,
and state-space model (SSM) to process both 2D and 3D image patches. A base
architecture is implemented following an encoder-decoder style, with several
derived architectures for image segmentation, reconstruction, and generation
tasks. In addition, we propose a general hierarchical architecture
incorporating a pyramid loss to optimize and fuse vertical features.
Experiments demonstrate that this simple design leads to an average improvement
of 5.60% in Dice score and 7.81% in mean interaction of units (mIoU) for
segmentation models, as well as an enhancement of 5.57% in peak signal-to-noise
ratio (PSNR) and 8.22% in structural similarity (SSIM) for reconstruction
models. We further utilize Flemme as an analytical tool to assess the
effectiveness and efficiency of various encoders across different tasks. Code
is available at https://github.com/wlsdzyzl/flemme.

### 摘要 (中文)

随着计算机视觉的快速发展和强大网络骨架和架构的出现，医学影像中的深度学习应用越来越重要。与自然图像不同，医学图像缺乏大量的数据量，但具有更多的模态，这使得难以训练一个能够满足各种数据集上良好性能的一般模型变得困难。在实践中，实践者通常会遭受手动构建并测试结合独立后端和架构的模型的过程，这是一个耗时费力的过程。我们提出Flemme，一种灵活模块化的医学图像学习平台。我们的平台分离了编码器和模型架构，使可以通过支持的编码器和架构的各种组合来构造不同的模型。我们使用卷积、Transformer和状态空间模型（SSM）等构建块处理二维和三维图像片断。基于编码器-解码器风格的一个基础架构被实现，并且提出了几个用于分割、重建和生成任务的衍生架构。此外，我们还提出了一个包含金字塔损失的通用高层级架构，以优化和融合垂直特征。实验表明，这种简单的设计导致分割模型中Dice分数平均提高了5.6%，单位交互值(mIoU)提高了7.81%，而重建模型中峰值信号噪声比(Peak Signal-to-Noise Ratio，PSNR)提高5.57%，结构相似性(SSIM)提高了8.22%。我们进一步利用Flemme作为分析工具，评估不同任务下不同编码器的有效性和效率。代码可在https://github.com/wlsdzyzl/flemme获取。

---

## FD2Talk_ Towards Generalized Talking Head Generation with Facial Decoupled Diffusion Model

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09384v1)

### Abstract (English)

Talking head generation is a significant research topic that still faces
numerous challenges. Previous works often adopt generative adversarial networks
or regression models, which are plagued by generation quality and average
facial shape problem. Although diffusion models show impressive generative
ability, their exploration in talking head generation remains unsatisfactory.
This is because they either solely use the diffusion model to obtain an
intermediate representation and then employ another pre-trained renderer, or
they overlook the feature decoupling of complex facial details, such as
expressions, head poses and appearance textures. Therefore, we propose a Facial
Decoupled Diffusion model for Talking head generation called FD2Talk, which
fully leverages the advantages of diffusion models and decouples the complex
facial details through multi-stages. Specifically, we separate facial details
into motion and appearance. In the initial phase, we design the Diffusion
Transformer to accurately predict motion coefficients from raw audio. These
motions are highly decoupled from appearance, making them easier for the
network to learn compared to high-dimensional RGB images. Subsequently, in the
second phase, we encode the reference image to capture appearance textures. The
predicted facial and head motions and encoded appearance then serve as the
conditions for the Diffusion UNet, guiding the frame generation. Benefiting
from decoupling facial details and fully leveraging diffusion models, extensive
experiments substantiate that our approach excels in enhancing image quality
and generating more accurate and diverse results compared to previous
state-of-the-art methods.

### 摘要 (中文)

合成头像是一个重要的研究课题，仍然面临着众多挑战。以往的工作经常采用生成对抗网络或回归模型，这些问题导致生成质量不佳和平均面部形状问题。尽管扩散模型显示了令人印象深刻的生成能力，但在合成头像的探索中仍存在不足。这是因为它们要么仅使用扩散模型获取中间表示，并然后采用另一个预训练渲染器，要么忽视了复杂面部细节特征解耦的问题，如表情、头部姿势和外观纹理。因此，我们提出了一种名为FD2Talk的面部解耦扩散模型来生成合成头像，该模型充分利用了扩散模型的优势，通过多阶段解耦复杂的面部细节。具体来说，我们在初始阶段设计Diffusion Transformer准确预测原始音频中的运动系数，这些运动与外观高度解耦，使得网络更容易学习，相比之下，高维度RGB图像更难被网络学习。随后，在第二阶段，我们将参考图像编码以捕获外观纹理。预测的面部和头部运动以及编码的外观作为Diffusion UNet的条件，引导帧的生成。得益于解耦面部细节和完全利用扩散模型的优势，大量的实验验证了我们的方法在提高图像质量和生成更多精确和多样化的结果方面超过了以前最先进的方法。

---

## TESL-Net_ A Transformer-Enhanced CNN for Accurate Skin Lesion Segmentation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09687v1)

### Abstract (English)

Early detection of skin cancer relies on precise segmentation of dermoscopic
images of skin lesions. However, this task is challenging due to the irregular
shape of the lesion, the lack of sharp borders, and the presence of artefacts
such as marker colours and hair follicles. Recent methods for melanoma
segmentation are U-Nets and fully connected networks (FCNs). As the depth of
these neural network models increases, they can face issues like the vanishing
gradient problem and parameter redundancy, potentially leading to a decrease in
the Jaccard index of the segmentation model. In this study, we introduced a
novel network named TESL-Net for the segmentation of skin lesions. The proposed
TESL-Net involves a hybrid network that combines the local features of a CNN
encoder-decoder architecture with long-range and temporal dependencies using
bi-convolutional long-short-term memory (Bi-ConvLSTM) networks and a Swin
transformer. This enables the model to account for the uncertainty of
segmentation over time and capture contextual channel relationships in the
data. We evaluated the efficacy of TESL-Net in three commonly used datasets
(ISIC 2016, ISIC 2017, and ISIC 2018) for the segmentation of skin lesions. The
proposed TESL-Net achieves state-of-the-art performance, as evidenced by a
significantly elevated Jaccard index demonstrated by empirical results.

### 摘要 (中文)

皮肤癌的早期检测依赖于对痣皮肤病征的精确分割。然而，这项任务具有挑战性，因为皮损的不规则形状、边缘模糊以及由于标记颜色和毛发囊肿等人为因素而存在的杂质。近年来用于黑色素瘤分割的方法是U-Net和全连接网络（FCN）。随着深度神经网络模型的深度增加，它们可能会遇到诸如梯度消失问题和参数冗余等问题，这可能导致分割模型Jaccard指数的下降。在本研究中，我们引入了一种名为TESL-Net的新网络来分割皮肤病变。所提出的TESL-Net涉及一个结合了CNN编码器-解码器架构中的局部特征与长距离/时序依赖关系的双向卷积长短期记忆（Bi-ConvLSTM）网络和Swin变换器的混合网络。这一模型能够考虑时间上的不确定性并捕获数据中上下文通道之间的关系。我们在三种常用的ISIC 2016、ISIC 2017和ISIC 2018数据集上评估了该方法的有效性，在分割皮肤病变方面实现了最先进的性能，通过实验结果展示了显著提高的Jaccard指数。

---

## Diff2CT_ Diffusion Learning to Reconstruct Spine CT from Biplanar X-Rays

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09731v1)

### Abstract (English)

Intraoperative CT imaging serves as a crucial resource for surgical guidance;
however, it may not always be readily accessible or practical to implement. In
scenarios where CT imaging is not an option, reconstructing CT scans from
X-rays can offer a viable alternative. In this paper, we introduce an
innovative method for 3D CT reconstruction utilizing biplanar X-rays. Distinct
from previous research that relies on conventional image generation techniques,
our approach leverages a conditional diffusion process to tackle the task of
reconstruction. More precisely, we employ a diffusion-based probabilistic model
trained to produce 3D CT images based on orthogonal biplanar X-rays. To improve
the structural integrity of the reconstructed images, we incorporate a novel
projection loss function. Experimental results validate that our proposed
method surpasses existing state-of-the-art benchmarks in both visual image
quality and multiple evaluative metrics. Specifically, our technique achieves a
higher Structural Similarity Index (SSIM) of 0.83, a relative increase of 10\%,
and a lower Fr\'echet Inception Distance (FID) of 83.43, which represents a
relative decrease of 25\%.

### 摘要 (中文)

在手术引导方面，实时CT成像是一个至关重要的资源；然而，并非总是容易获得或实践可行。当CT成像不是选项时，从X光片重建CT扫描可以提供一个可行的替代方案。在这篇论文中，我们介绍了利用双平面X射线进行三维CT重建的一种创新方法。与以往依赖常规图像生成技术的研究不同，我们的方法利用条件扩散过程来解决重建任务。更准确地说，我们采用了一种基于双平面X射线训练的扩散概率模型，该模型旨在根据双平面X射线生成3D CT图像。为了提高重构图像的结构完整性，我们引入了一个新的投影损失函数。实验结果验证了我们在视觉图像质量和多个评价指标方面的提出的方法优于现有最先进的基准。具体而言，我们的技术获得了结构相似指数（SSIM）的值为0.83，相对增加率为10%，并且Fr\'echet Inception Distance（FID）的值为83.43，这代表了一个相对下降率25%。

---

## Coarse-Fine View Attention Alignment-Based GAN for CT Reconstruction from Biplanar X-Rays

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09736v1)

### Abstract (English)

For surgical planning and intra-operation imaging, CT reconstruction using
X-ray images can potentially be an important alternative when CT imaging is not
available or not feasible. In this paper, we aim to use biplanar X-rays to
reconstruct a 3D CT image, because biplanar X-rays convey richer information
than single-view X-rays and are more commonly used by surgeons. Different from
previous studies in which the two X-ray views were treated indifferently when
fusing the cross-view data, we propose a novel attention-informed
coarse-to-fine cross-view fusion method to combine the features extracted from
the orthogonal biplanar views. This method consists of a view attention
alignment sub-module and a fine-distillation sub-module that are designed to
work together to highlight the unique or complementary information from each of
the views. Experiments have demonstrated the superiority of our proposed method
over the SOTA methods.

### 摘要 (中文)

对于手术规划和内镜成像，使用CT图像重建可能是一个重要的替代方案，当CT成像不可用或不可行时。本文旨在使用双平面X射线来重建三维CT图像，因为双平面X射线比单视X射线提供更丰富的信息，并且由外科医生更为常用。与以往的研究不同，在融合交叉视图数据时，我们提出了一种新颖的注意力引导粗到细交叉视图融合方法，该方法旨在从每个视图中提取特征进行组合。该方法包括一个视角注意力对齐子模块和一个细化分馏子模块，这两个模块设计在一起工作以突出每种视图的独特或互补信息。实验已经证明了我们的提出的解决方案优于当前最先进的方法。

---

## Harnessing Multi-resolution and Multi-scale Attention for Underwater Image Restoration

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09912v1)

### Abstract (English)

Underwater imagery is often compromised by factors such as color distortion
and low contrast, posing challenges for high-level vision tasks. Recent
underwater image restoration (UIR) methods either analyze the input image at
full resolution, resulting in spatial richness but contextual weakness, or
progressively from high to low resolution, yielding reliable semantic
information but reduced spatial accuracy. Here, we propose a lightweight
multi-stage network called Lit-Net that focuses on multi-resolution and
multi-scale image analysis for restoring underwater images while retaining
original resolution during the first stage, refining features in the second,
and focusing on reconstruction in the final stage. Our novel encoder block
utilizes parallel $1\times1$ convolution layers to capture local information
and speed up operations. Further, we incorporate a modified weighted color
channel-specific $l_1$ loss ($cl_1$) function to recover color and detail
information. Extensive experimentations on publicly available datasets suggest
our model's superiority over recent state-of-the-art methods, with significant
improvement in qualitative and quantitative measures, such as $29.477$ dB PSNR
($1.92\%$ improvement) and $0.851$ SSIM ($2.87\%$ improvement) on the EUVP
dataset. The contributions of Lit-Net offer a more robust approach to
underwater image enhancement and super-resolution, which is of considerable
importance for underwater autonomous vehicles and surveillance. The code is
available at: https://github.com/Alik033/Lit-Net.

### 摘要 (中文)

水下影像常常受到诸如色彩失真和对比度低等因素的影响，这些因素使高精度的视觉任务变得具有挑战性。最近的水下图像恢复（UIR）方法要么在全分辨率下分析输入图像，导致空间丰富但缺乏语义信息，要么从高到低分辨率逐步进行，提供可靠的语言信息但减少空间准确性。在这里，我们提出了一个轻量级多阶段网络称为Lit-Net，该网络专注于多分辨率和多尺度图像分析以恢复水下影像，在第一阶段保持原始分辨率的同时，第二阶段细化特征，在最后阶段关注重建。我们的新型编码块利用并行的1×1卷积层来捕获局部信息并加快操作。此外，我们引入了修改后的权重色通道特定$L_1$损失函数（$cl_1$），用于恢复颜色和细节信息。对公开可用数据集的广泛实验表明，我们的模型优于近期最先进的方法，在质性和定量测量方面表现出显著改进，例如EUVP数据集中的PSNR提高了29.477分贝（1.92％改善）和SSIM提高了0.851分贝（2.87％改善）。Lit-Net提出的贡献提供了更稳健的水下图像增强和超分辨率方法，这对于水下自主车辆和监控非常重要。代码可在https://github.com/Alik033/Lit-Net中找到。

---

## Attribution Analysis Meets Model Editing_ Advancing Knowledge Correction in Vision Language Models with VisEdit

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09916v1)

### Abstract (English)

Model editing aims to correct outdated or erroneous knowledge in large models
without costly retraining. Recent research discovered that the mid-layer
representation of the subject's final token in a prompt has a strong influence
on factual predictions, and developed Large Language Model (LLM) editing
techniques based on this observation. However, for Vision-LLMs (VLLMs), how
visual representations impact the predictions from a decoder-only language
model remains largely unexplored. To the best of our knowledge, model editing
for VLLMs has not been extensively studied in the literature. In this work, we
employ the contribution allocation and noise perturbation methods to measure
the contributions of visual representations for token predictions. Our
attribution analysis shows that visual representations in mid-to-later layers
that are highly relevant to the prompt contribute significantly to predictions.
Based on these insights, we propose VisEdit, a novel model editor for VLLMs
that effectively corrects knowledge by editing intermediate visual
representations in regions important to the edit prompt. We evaluated VisEdit
using multiple VLLM backbones and public VLLM editing benchmark datasets. The
results show the superiority of VisEdit over the strong baselines adapted from
existing state-of-the-art editors for LLMs.

### 摘要 (中文)

模型编辑旨在纠正大型模型中过时或错误的知识。近期研究发现，提示中的最终词的中间层表示对事实预测有强烈的影响，并基于这一观察发展了大规模语言模型（LLM）编辑技术。然而，对于视觉LLMs（VLLMs），视觉表示如何影响从解码器仅的语言模型得出的预测仍处于探索阶段。到目前为止，在文献中，我们所知的用于VLLMs的模型编辑尚未在学术界进行广泛研究。本工作中，我们采用贡献分配和噪声扰动方法来测量视觉表示对token预测的贡献。我们的归因分析表明，中间至后期层高度相关的视觉表示对预测贡献显著。基于这些洞察，我们提出了一种新的VLLM编辑器VisEdit，该编辑器有效地通过编辑重要到编辑提示的中间视觉表示来修正知识。我们在多个VLLM骨架上评估了VisEdit的结果，并使用公共的VLLM编辑基准数据集进行了评价。结果显示，VisEdit优于现有状态最先进编辑器强基线改编自现有最先进的编辑器。

---

## DiscoNeRF_ Class-Agnostic Object Field for 3D Object Discovery

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09928v1)

### Abstract (English)

Neural Radiance Fields (NeRFs) have become a powerful tool for modeling 3D
scenes from multiple images. However, NeRFs remain difficult to segment into
semantically meaningful regions. Previous approaches to 3D segmentation of
NeRFs either require user interaction to isolate a single object, or they rely
on 2D semantic masks with a limited number of classes for supervision. As a
consequence, they generalize poorly to class-agnostic masks automatically
generated in real scenes. This is attributable to the ambiguity arising from
zero-shot segmentation, yielding inconsistent masks across views. In contrast,
we propose a method that is robust to inconsistent segmentations and
successfully decomposes the scene into a set of objects of any class. By
introducing a limited number of competing object slots against which masks are
matched, a meaningful object representation emerges that best explains the 2D
supervision and minimizes an additional regularization term. Our experiments
demonstrate the ability of our method to generate 3D panoptic segmentations on
complex scenes, and extract high-quality 3D assets from NeRFs that can then be
used in virtual 3D environments.

### 摘要 (中文)

神经辐射场（NeRFs）已成为从多个图像中建模三维场景的强大工具。然而，NeRFs仍难以根据语义意义分割成有意义的区域。在3D对象分割方面，现有的方法要么要求用户交互来隔离单个物体，要么依赖于具有有限类别的二维语义掩码进行监督。因此，它们对自动生成的无类别掩码泛化能力差。这归因于零样本分割带来的模糊性，导致不同视图下的不一致掩码。相反，我们提出了一种能够抵抗不一致分割的方法，并成功地将场景分解为任何类别的对象集合。通过引入与掩码匹配的有限数量的竞争性对象槽，一种有意义的对象表示脱颖而出，它最好解释二维监督并最小化额外的正则化项。我们的实验展示了使用我们的方法在复杂场景上生成3D全景分割的能力，以及从NeRF中提取高质量的3D资产，这些资产可以用于虚拟3D环境。

---

## Pose-GuideNet_ Automatic Scanning Guidance for Fetal Head Ultrasound from Pose Estimation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09931v1)

### Abstract (English)

3D pose estimation from a 2D cross-sectional view enables healthcare
professionals to navigate through the 3D space, and such techniques initiate
automatic guidance in many image-guided radiology applications. In this work,
we investigate how estimating 3D fetal pose from freehand 2D ultrasound
scanning can guide a sonographer to locate a head standard plane. Fetal head
pose is estimated by the proposed Pose-GuideNet, a novel 2D/3D registration
approach to align freehand 2D ultrasound to a 3D anatomical atlas without the
acquisition of 3D ultrasound. To facilitate the 2D to 3D cross-dimensional
projection, we exploit the prior knowledge in the atlas to align the standard
plane frame in a freehand scan. A semantic-aware contrastive-based approach is
further proposed to align the frames that are off standard planes based on
their anatomical similarity. In the experiment, we enhance the existing
assessment of freehand image localization by comparing the transformation of
its estimated pose towards standard plane with the corresponding probe motion,
which reflects the actual view change in 3D anatomy. Extensive results on two
clinical head biometry tasks show that Pose-GuideNet not only accurately
predicts pose but also successfully predicts the direction of the fetal head.
Evaluations with probe motions further demonstrate the feasibility of adopting
Pose-GuideNet for freehand ultrasound-assisted navigation in a sensor-free
environment.

### 摘要 (中文)

从二维横断面视图中估计三维姿势，可以允许医疗专业人士在三维空间中导航，并且此类技术在许多图像引导放射学应用中启动自动导向。在这项工作中，我们研究了如何根据自由手2D超声扫描估计胎儿的三维姿势，以指导产科医生找到头部标准平面。通过提出的一种新的二维到三维注册方法，该方法无需获取三维超声即可将自由手2D超声扫描对齐到三维解剖地图。为了促进二维到三维维度的跨维度投影，我们在解剖图中利用前知识来对自由扫描中的标准平面框架进行校准。进一步提出了一个基于语义感知对抗的方法，用于根据它们的解剖相似性对偏离标准平面的帧进行校准。实验中，我们通过比较其估算姿态与相应探头运动之间的变换，以及这些变化反映在三维解剖中的实际视角变化来增强现有自由手影像定位评估。在两个临床头部测量任务上进行了广泛的结果显示，Pose-GuideNet不仅准确地预测姿势，而且成功预测胎儿头部的方向。与探头运动的评价进一步证明了在无传感器环境中使用Pose-GuideNet进行自由手超声辅助导航的可能性。

---

## ML-CrAIST_ Multi-scale Low-high Frequency Information-based Cross black Attention with Image Super-resolving Transformer

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09940v1)

### Abstract (English)

Recently, transformers have captured significant interest in the area of
single-image super-resolution tasks, demonstrating substantial gains in
performance. Current models heavily depend on the network's extensive ability
to extract high-level semantic details from images while overlooking the
effective utilization of multi-scale image details and intermediate information
within the network. Furthermore, it has been observed that high-frequency areas
in images present significant complexity for super-resolution compared to
low-frequency areas. This work proposes a transformer-based super-resolution
architecture called ML-CrAIST that addresses this gap by utilizing low-high
frequency information in multiple scales. Unlike most of the previous work
(either spatial or channel), we operate spatial and channel self-attention,
which concurrently model pixel interaction from both spatial and channel
dimensions, exploiting the inherent correlations across spatial and channel
axis. Further, we devise a cross-attention block for super-resolution, which
explores the correlations between low and high-frequency information.
Quantitative and qualitative assessments indicate that our proposed ML-CrAIST
surpasses state-of-the-art super-resolution methods (e.g., 0.15 dB gain
@Manga109 $\times$4). Code is available on:
https://github.com/Alik033/ML-CrAIST.

### 摘要 (中文)

最近，变换器在单图像超分辨率任务中引起了广泛的关注，显示出在性能上的显著提升。当前模型依赖于网络从图像中提取高层次语义细节的能力，而忽视了利用多尺度图像细节和网络中间信息的有效利用。此外，已观察到图像中的高频区域与低频区域相比具有更大的复杂性。本工作提出了一种名为ML-CrAIST的变换器超分辨率架构，旨在通过利用不同尺度的低高频率信息来解决这一差距。不同于以往的工作（无论是空间还是通道），我们操作了空间和通道自注意力，同时从两个维度（空间和通道）同时建模像素之间的交互，利用空间和通道轴上的内在相关性。进一步，我们设计了一个跨注意力块用于超分辨率，该模块探索了低高频信息之间的相关性。量化和定性的评估表明，我们的ML-CrAIST超越了最先进的超分辨率方法（例如，在Manga109$\times$4上获得0.15dB的增益）。代码可在以下地址获取：
https://github.com/Alik033/ML-CrAIST.

---

## C___2__RL_ Content and Context Representation Learning for Gloss-free Sign Language Translation and Retrieval

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09949v1)

### Abstract (English)

Sign Language Representation Learning (SLRL) is crucial for a range of sign
language-related downstream tasks such as Sign Language Translation (SLT) and
Sign Language Retrieval (SLRet). Recently, many gloss-based and gloss-free SLRL
methods have been proposed, showing promising performance. Among them, the
gloss-free approach shows promise for strong scalability without relying on
gloss annotations. However, it currently faces suboptimal solutions due to
challenges in encoding the intricate, context-sensitive characteristics of sign
language videos, mainly struggling to discern essential sign features using a
non-monotonic video-text alignment strategy. Therefore, we introduce an
innovative pretraining paradigm for gloss-free SLRL, called C${^2}$RL, in this
paper. Specifically, rather than merely incorporating a non-monotonic semantic
alignment of video and text to learn language-oriented sign features, we
emphasize two pivotal aspects of SLRL: Implicit Content Learning (ICL) and
Explicit Context Learning (ECL). ICL delves into the content of communication,
capturing the nuances, emphasis, timing, and rhythm of the signs. In contrast,
ECL focuses on understanding the contextual meaning of signs and converting
them into equivalent sentences. Despite its simplicity, extensive experiments
confirm that the joint optimization of ICL and ECL results in robust sign
language representation and significant performance gains in gloss-free SLT and
SLRet tasks. Notably, C${^2}$RL improves the BLEU-4 score by +5.3 on P14T,
+10.6 on CSL-daily, +6.2 on OpenASL, and +1.3 on How2Sign. It also boosts the
R@1 score by +8.3 on P14T, +14.4 on CSL-daily, and +5.9 on How2Sign.
Additionally, we set a new baseline for the OpenASL dataset in the SLRet task.

### 摘要 (中文)

手语表示学习（SLRL）对于一系列与手语相关的下游任务，如手语翻译（SLT）和手语检索（SLRet）至关重要。最近，有许多基于语法的和无语法的SLRL方法被提出，显示出良好的性能。其中，无语法的方法在不依赖于注释的情况下具有强大的可扩展性。然而，由于难以编码出手语视频中的复杂、上下文敏感特征，目前仍面临优化解决方案的挑战。因此，在此文中，我们引入了一个创新的预训练范式来解决无语法的SLRL，称为C${^2}$RL。具体来说，除了仅将视频和文本之间的非单调语义对齐作为语言导向的手语特征的学习方式外，我们强调了两个关键方面：隐含内容学习（ICL）和显式上下文学习（ECL）。ICL深入交流的内容中捕捉细微差别、重点、时间节奏等。相反，ECL专注于理解手势的上下文意义，并将其转换为等价句子。尽管其简单，但广泛的实验结果证实，通过联合优化ICL和ECL，可以实现稳定的手语表示，并在无语法SLT和SLRet任务中显著提高性能。值得注意的是，C${^2}$RL在P14T上的BLEU-4得分提高了+5.3，而在CSL-daily、OpenASL和How2Sign上的R@1分数分别提高了+10.6、+6.2和+1.3。此外，我们为OpenASL数据集在SLRet任务上设置了一项新的基线。

---

## Detecting Adversarial Attacks in Semantic Segmentation via Uncertainty Estimation_ A Deep Analysis

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10021v1)

### Abstract (English)

Deep neural networks have demonstrated remarkable effectiveness across a wide
range of tasks such as semantic segmentation. Nevertheless, these networks are
vulnerable to adversarial attacks that add imperceptible perturbations to the
input image, leading to false predictions. This vulnerability is particularly
dangerous in safety-critical applications like automated driving. While
adversarial examples and defense strategies are well-researched in the context
of image classification, there is comparatively less research focused on
semantic segmentation. Recently, we have proposed an uncertainty-based method
for detecting adversarial attacks on neural networks for semantic segmentation.
We observed that uncertainty, as measured by the entropy of the output
distribution, behaves differently on clean versus adversely perturbed images,
and we utilize this property to differentiate between the two. In this extended
version of our work, we conduct a detailed analysis of uncertainty-based
detection of adversarial attacks including a diverse set of adversarial attacks
and various state-of-the-art neural networks. Our numerical experiments show
the effectiveness of the proposed uncertainty-based detection method, which is
lightweight and operates as a post-processing step, i.e., no model
modifications or knowledge of the adversarial example generation process are
required.

### 摘要 (中文)

深度神经网络已经在诸如语义分割等广泛任务上表现出色，然而这些网络对不可察觉的干扰图像（即在输入图像中添加不可见扰动）非常敏感，导致错误预测。这种脆弱性尤其危险，在像自动驾驶这样的关键应用中。尽管在图像分类上下文中已研究了对抗示例和防御策略，但在语义分割方面相对较少的研究集中在防御攻击上。最近，我们提出了一种基于不确定性的方法来检测神经网络中的对抗攻击。我们观察到，输出分布熵的行为不同对于干净的图像和受干扰的图像，我们利用这一点来区分这两个。在此工作的一个扩展版本中，我们详细分析了基于不确定性的对抗攻击检测，包括各种类型的对抗攻击以及最先进的神经网络。我们的数值实验显示了所提出的基于不确定性的检测方法的有效性，该方法是轻量级的，并作为后处理步骤运行，即不需要模型修改或对抗示例生成过程的知识。

---

## Towards a Benchmark for Colorectal Cancer Segmentation in Endorectal Ultrasound Videos_ Dataset and Model Development

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10067v1)

### Abstract (English)

Endorectal ultrasound (ERUS) is an important imaging modality that provides
high reliability for diagnosing the depth and boundary of invasion in
colorectal cancer. However, the lack of a large-scale ERUS dataset with
high-quality annotations hinders the development of automatic ultrasound
diagnostics. In this paper, we collected and annotated the first benchmark
dataset that covers diverse ERUS scenarios, i.e. colorectal cancer
segmentation, detection, and infiltration depth staging. Our ERUS-10K dataset
comprises 77 videos and 10,000 high-resolution annotated frames. Based on this
dataset, we further introduce a benchmark model for colorectal cancer
segmentation, named the Adaptive Sparse-context TRansformer (ASTR). ASTR is
designed based on three considerations: scanning mode discrepancy, temporal
information, and low computational complexity. For generalizing to different
scanning modes, the adaptive scanning-mode augmentation is proposed to convert
between raw sector images and linear scan ones. For mining temporal
information, the sparse-context transformer is incorporated to integrate
inter-frame local and global features. For reducing computational complexity,
the sparse-context block is introduced to extract contextual features from
auxiliary frames. Finally, on the benchmark dataset, the proposed ASTR model
achieves a 77.6% Dice score in rectal cancer segmentation, largely
outperforming previous state-of-the-art methods.

### 摘要 (中文)

内镜超声（ERUS）是提供诊断结肠癌侵入深度边界的重要影像学方法，但由于缺乏高质量标注的大规模ERUS数据集，阻碍了自动超声诊断的发展。本论文收集和注释了第一个覆盖多种ERUS场景的基准数据集，即结肠癌分割、检测和浸润深度阶段。我们的ERUS-10K数据集包含77个视频和1万张高分辨率标注帧。基于这个数据集，我们进一步引入了一个用于结肠癌分割的基准模型，命名为Adaptive Sparse-context Transfomer (ASTR)。ASTR的设计基于三个考虑因素：扫描模式差异、时间信息和低计算复杂性。为了将不同扫描模式进行通用化，提出了可变扫描模式增强策略来将原始扇区图像转换为线性扫描图像。对于挖掘时空信息，结合稀疏上下文变换器以整合相邻帧局部和全局特征。为了降低计算复杂性，引入了稀疏上下文块从辅助帧中提取上下文特征。最后，在基准数据集中，提出的ASTR模型在直肠癌分割中取得了77.6%的Dice分数，大大超过了以前最先进的方法。

---

## Learning Precise Affordances from Egocentric Videos for Robotic Manipulation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10123v1)

### Abstract (English)

Affordance, defined as the potential actions that an object offers, is
crucial for robotic manipulation tasks. A deep understanding of affordance can
lead to more intelligent AI systems. For example, such knowledge directs an
agent to grasp a knife by the handle for cutting and by the blade when passing
it to someone. In this paper, we present a streamlined affordance learning
system that encompasses data collection, effective model training, and robot
deployment. First, we collect training data from egocentric videos in an
automatic manner. Different from previous methods that focus only on the object
graspable affordance and represent it as coarse heatmaps, we cover both
graspable (e.g., object handles) and functional affordances (e.g., knife
blades, hammer heads) and extract data with precise segmentation masks. We then
propose an effective model, termed Geometry-guided Affordance Transformer
(GKT), to train on the collected data. GKT integrates an innovative Depth
Feature Injector (DFI) to incorporate 3D shape and geometric priors, enhancing
the model's understanding of affordances. To enable affordance-oriented
manipulation, we further introduce Aff-Grasp, a framework that combines GKT
with a grasp generation model. For comprehensive evaluation, we create an
affordance evaluation dataset with pixel-wise annotations, and design
real-world tasks for robot experiments. The results show that GKT surpasses the
state-of-the-art by 15.9% in mIoU, and Aff-Grasp achieves high success rates of
95.5% in affordance prediction and 77.1% in successful grasping among 179
trials, including evaluations with seen, unseen objects, and cluttered scenes.

### 摘要 (中文)

可得性，定义为对象可能执行的行动，对于机器人操作任务至关重要。对可得性的深入理解可以引导更智能的人工智能系统。例如，这种知识指示代理通过把手抓住刀子进行切割，通过刀片传递给他人时。在本文中，我们提出了一种紧凑的可得性学习系统，该系统涵盖数据收集、有效模型训练和机器人部署。首先，我们在自动方式从内视视频中收集训练数据。与以前的方法不同的是，它们仅关注可抓取的可得性和将其表示为粗热图，而我们覆盖了可抓取（例如，物体把手）和功能可得性（例如，刀片刀刃，锤头）。然后，我们提出了一个有效的模型，称为几何引导的可得性变换器（GKT），用于训练所收集的数据。GKT结合了一个创新的深度特征注入器（DFI），以整合三维形状和几何先验，增强模型对可得性的理解能力。为了促进基于可得性的操纵，我们进一步引入了Aff-Grasp框架，该框架结合了GKT与抓握生成模型。为了进行全面评估，我们创建了一个使用像素级标注的可得性评估数据集，并设计了一系列机器人实验的真实世界任务。结果显示，GKT在mIoU上超过了当前最佳水平15.9%，并且Aff-Grasp在预测成功完成的抓握以及试验中的成功率高达95.5%，包括观察到的对象、未见的对象以及杂乱的场景。

请注意，上述内容是根据英文原文翻译过来的，可能存在不准确或不完整的翻译。

---

## LoopSplat_ Loop Closure by Registering 3D Gaussian Splats

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10154v1)

### Abstract (English)

Simultaneous Localization and Mapping (SLAM) based on 3D Gaussian Splats
(3DGS) has recently shown promise towards more accurate, dense 3D scene maps.
However, existing 3DGS-based methods fail to address the global consistency of
the scene via loop closure and/or global bundle adjustment. To this end, we
propose LoopSplat, which takes RGB-D images as input and performs dense mapping
with 3DGS submaps and frame-to-model tracking. LoopSplat triggers loop closure
online and computes relative loop edge constraints between submaps directly via
3DGS registration, leading to improvements in efficiency and accuracy over
traditional global-to-local point cloud registration. It uses a robust pose
graph optimization formulation and rigidly aligns the submaps to achieve global
consistency. Evaluation on the synthetic Replica and real-world TUM-RGBD,
ScanNet, and ScanNet++ datasets demonstrates competitive or superior tracking,
mapping, and rendering compared to existing methods for dense RGB-D SLAM. Code
is available at \href{https://loopsplat.github.io/}{loopsplat.github.io}.

### 摘要 (中文)

基于3D高斯分块（3DGS）的同步定位与地图构建（SLAM）近年来在更准确、密集的三维场景映射方面显示出潜力。然而，现有的基于3DGS的方法未能通过环闭合和全局包络调整来解决全球一致性问题。为此，我们提出LoopSplat，它以RGB-D图像作为输入，并使用3DGS子块进行密集映射和帧到模型跟踪。LoopSplat在线触发环闭合，并直接通过3DGS注册计算子块之间的相对环闭合边约束，从而比传统的全局到局部点云注册具有更高的效率和准确性。它使用稳健的位姿图优化形式，并刚性对齐子块实现全局一致性。在合成仿真实例和现实世界的TUM-RGBD、ScanNet和ScanNet++数据集上的评估中，LoopSplat与现有方法相比，在稠密RGB-D SLAM的跟踪、映射和渲染上表现出有竞争力或优于的表现。代码可在https://loopsplat.github.io/处获取。

---

## LongVILA_ Scaling Long-Context Visual Language Models for Long Videos

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10188v1)

### Abstract (English)

Long-context capability is critical for multi-modal foundation models. We
introduce LongVILA, a full-stack solution for long-context vision-language
models, including system, model training, and dataset development. On the
system side, we introduce the first Multi-Modal Sequence Parallelism (MM-SP)
system that enables long-context training and inference, enabling 2M context
length training on 256 GPUs. MM-SP is also efficient, being 2.1x - 5.7x faster
than Ring-Style Sequence Parallelism and 1.1x - 1.4x faster than Megatron-LM in
text-only settings. Moreover, it seamlessly integrates with Hugging Face
Transformers. For model training, we propose a five-stage pipeline comprising
alignment, pre-training, context extension, and long-short joint supervised
fine-tuning. Regarding datasets, we meticulously construct large-scale visual
language pre-training datasets and long video instruction-following datasets to
support our multi-stage training process. The full-stack solution extends the
feasible frame number of VILA by a factor of 128 (from 8 to 1024 frames) and
improves long video captioning score from 2.00 to 3.26 (1.6x), achieving 99.5%
accuracy in 1400-frames video (274k context length) needle in a haystack.
LongVILA-8B also demonstrates a consistent improvement in performance on long
videos within the VideoMME benchmark as the video frames increase.

### 摘要 (中文)

长语境能力对于多模态基础模型至关重要。我们推出了LongVILA，这是一个全栈解决方案，用于长语境视觉语言模型，包括系统、模型训练和数据集开发。在系统层面上，我们介绍了首个支持长语境训练和推断的Multi-Modal Sequence Parallelism（MM-SP）系统，该系统能够在256个GPU上实现2米长的上下文长度的训练。此外，MM-SP的效率很高，在文本任务中是Ring-Style Sequence Parallelism的2.1倍到5.7倍，比Megatron-LM快1.1倍到1.4倍。而且它无缝集成于Hugging Face Transformers。在模型训练方面，我们提出了一个包含预训练、上下文扩展、长期短序列联合监督式微调五个阶段的五步管道。关于数据集，我们精心构建了大规模的视觉语言预训练数据集和长视频指令跟随数据集，以支持我们的多阶段训练过程。全栈解决方案将VILA的可行帧数提高了128倍（从8帧增加到1024帧），并在视频长度从2.00提高到3.26的情况下，长视频标题得分提高了1.6倍，准确率达到了99.5%。LongVILA-8B也展示了在VideoMME基准测试中的长视频性能随帧数增加而稳定的改善。

---

## MeshFormer_ High-Quality Mesh Generation with 3D-Guided Reconstruction Model

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10198v1)

### Abstract (English)

Open-world 3D reconstruction models have recently garnered significant
attention. However, without sufficient 3D inductive bias, existing methods
typically entail expensive training costs and struggle to extract high-quality
3D meshes. In this work, we introduce MeshFormer, a sparse-view reconstruction
model that explicitly leverages 3D native structure, input guidance, and
training supervision. Specifically, instead of using a triplane representation,
we store features in 3D sparse voxels and combine transformers with 3D
convolutions to leverage an explicit 3D structure and projective bias. In
addition to sparse-view RGB input, we require the network to take input and
generate corresponding normal maps. The input normal maps can be predicted by
2D diffusion models, significantly aiding in the guidance and refinement of the
geometry's learning. Moreover, by combining Signed Distance Function (SDF)
supervision with surface rendering, we directly learn to generate high-quality
meshes without the need for complex multi-stage training processes. By
incorporating these explicit 3D biases, MeshFormer can be trained efficiently
and deliver high-quality textured meshes with fine-grained geometric details.
It can also be integrated with 2D diffusion models to enable fast
single-image-to-3D and text-to-3D tasks. Project page:
https://meshformer3d.github.io

### 摘要 (中文)

开放世界三维重建模型最近引起了广泛的关注。然而，如果没有足够的三维有向偏见，现有的方法通常需要昂贵的训练成本，并且很难提取高质量的三维网格。在本文中，我们引入了MeshFormer，这是一种基于稀疏视图的重构模型，它明确利用三维原始结构、输入引导和培训监督。具体来说，我们不再使用三角形表示，而是存储特征在三维稀疏多面体中，然后与深度卷积结合使用以利用明确的三维结构和投影偏置。此外，除了稀疏视图RGB输入外，还需要网络接收并生成对应的正常映射。通过组合Signed Distance Function（SDF）监督与表面渲染，我们可以直接学习高精度的网格而无需进行复杂的多层次训练过程。通过集成这些明确的三维偏差，MeshFormer可以高效地进行训练，并提供具有精细几何细节的纹理化网格。它还可以与其他二维扩散模型集成，以实现单图像到三维和文本到三维任务中的快速转换。项目页面： https://meshformer3d.github.io

---

## Anytime-Valid Inference for Double_Debiased Machine Learning of Causal Parameters

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09598v1)

### Abstract (English)

Double (debiased) machine learning (DML) has seen widespread use in recent
years for learning causal/structural parameters, in part due to its flexibility
and adaptability to high-dimensional nuisance functions as well as its ability
to avoid bias from regularization or overfitting. However, the classic
double-debiased framework is only valid asymptotically for a predetermined
sample size, thus lacking the flexibility of collecting more data if sharper
inference is needed, or stopping data collection early if useful inferences can
be made earlier than expected. This can be of particular concern in large scale
experimental studies with huge financial costs or human lives at stake, as well
as in observational studies where the length of confidence of intervals do not
shrink to zero even with increasing sample size due to partial identifiability
of a structural parameter. In this paper, we present time-uniform counterparts
to the asymptotic DML results, enabling valid inference and confidence
intervals for structural parameters to be constructed at any arbitrary
(possibly data-dependent) stopping time. We provide conditions which are only
slightly stronger than the standard DML conditions, but offer the stronger
guarantee for anytime-valid inference. This facilitates the transformation of
any existing DML method to provide anytime-valid guarantees with minimal
modifications, making it highly adaptable and easy to use. We illustrate our
procedure using two instances: a) local average treatment effect in online
experiments with non-compliance, and b) partial identification of average
treatment effect in observational studies with potential unmeasured
confounding.

### 摘要 (中文)

近年来，双侧去偏差机器学习（DML）在学习因果结构参数方面得到了广泛使用，部分原因在于其对高维度噪声函数的灵活性和适应性以及避免过拟合或正则化带来的偏见的能力。然而，经典的双侧去偏差框架只适用于预定样本大小下，因此无法收集更多数据以满足更精确的推断需求，或者提前停止数据收集如果可以早些做出有用的信息，则可能会导致问题。这在大型实验研究中，在面临巨大财务成本或人类生命安全时尤为突出，也体现在观察性研究中，由于结构参数的不可识别性会导致置信区间长度不随样本规模增加而缩小。本文提出了一种时间均匀的等效结果，使得任何现有的DML方法都可以在其任意可能的数据依赖的时间点上提供任何类型的结构参数的有效推断和置信区间的构建。我们提供了比标准DML条件仅稍微更强的条件，但保证了任何时候有效推断的结果。这使转换现有DML方法变得容易，只需进行很少的修改即可获得实时有效的保证，使其高度灵活且易于使用。通过两个实例来说明我们的程序：a)在线实验中的局部平均治疗效果；b)观测性研究中的潜在未测量混杂因素下的平均治疗效果的部分可识别性。

---

## Characterizing and Evaluating the Reliability of LLMs against Jailbreak Attacks

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09326v1)

### Abstract (English)

Large Language Models (LLMs) have increasingly become pivotal in content
generation with notable societal impact. These models hold the potential to
generate content that could be deemed harmful.Efforts to mitigate this risk
include implementing safeguards to ensure LLMs adhere to social ethics.However,
despite such measures, the phenomenon of "jailbreaking" -- where carefully
crafted prompts elicit harmful responses from models -- persists as a
significant challenge. Recognizing the continuous threat posed by jailbreaking
tactics and their repercussions for the trustworthy use of LLMs, a rigorous
assessment of the models' robustness against such attacks is essential. This
study introduces an comprehensive evaluation framework and conducts an
large-scale empirical experiment to address this need. We concentrate on 10
cutting-edge jailbreak strategies across three categories, 1525 questions from
61 specific harmful categories, and 13 popular LLMs. We adopt multi-dimensional
metrics such as Attack Success Rate (ASR), Toxicity Score, Fluency, Token
Length, and Grammatical Errors to thoroughly assess the LLMs' outputs under
jailbreak. By normalizing and aggregating these metrics, we present a detailed
reliability score for different LLMs, coupled with strategic recommendations to
reduce their susceptibility to such vulnerabilities. Additionally, we explore
the relationships among the models, attack strategies, and types of harmful
content, as well as the correlations between the evaluation metrics, which
proves the validity of our multifaceted evaluation framework. Our extensive
experimental results demonstrate a lack of resilience among all tested LLMs
against certain strategies, and highlight the need to concentrate on the
reliability facets of LLMs. We believe our study can provide valuable insights
into enhancing the security evaluation of LLMs against jailbreak within the
domain.

### 摘要 (中文)

大型语言模型（LLM）在内容生成中变得日益重要，其社会影响不容忽视。这些模型具有潜在能力，能够生成可能被视为有害的内容。减轻这一风险的努力包括实施措施以确保LLMs遵守社会伦理准则。然而，尽管有这些措施，但“越狱”现象——精心设计的提示会从模型中引发有害响应——仍是一个重要的挑战。认识到持续威胁和LML使用中的不信任感，对这类攻击的模型鲁棒性进行严格的评估是至关重要的。这项研究引入了一个全面的评估框架，并进行了大规模的实证实验来应对这一需求。我们专注于六个主要危害类别中的1525个问题以及13款流行的人工智能模型。我们将采用诸如攻击成功率（ASR）、毒性分数、流畅度、词长和语法错误等多维指标来详细评估LLM的输出。通过标准化和聚合这些指标，我们可以提供不同LLM的不同可靠性的详细评分，并提出战略建议来减少它们对此类漏洞的敏感性。此外，我们还探索了模型之间的关系、攻击策略和有害内容类型，以及评价指标之间的相关性，这证明了我们多维度评估框架的有效性。我们的大量实验结果表明，所有测试过的LLM都未能对其某些策略保持韧性，并强调了加强LLM可靠性方面的需求。我们认为，我们的研究可以为增强在特定领域内对抗“越狱”的安全性评估提供有价值的见解。

---

## VRCopilot_ Authoring 3D Layouts with Generative AI Models in VR

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09382v1)

### Abstract (English)

Immersive authoring provides an intuitive medium for users to create 3D
scenes via direct manipulation in Virtual Reality (VR). Recent advances in
generative AI have enabled the automatic creation of realistic 3D layouts.
However, it is unclear how capabilities of generative AI can be used in
immersive authoring to support fluid interactions, user agency, and creativity.
We introduce VRCopilot, a mixed-initiative system that integrates pre-trained
generative AI models into immersive authoring to facilitate human-AI
co-creation in VR. VRCopilot presents multimodal interactions to support rapid
prototyping and iterations with AI, and intermediate representations such as
wireframes to augment user controllability over the created content. Through a
series of user studies, we evaluated the potential and challenges in manual,
scaffolded, and automatic creation in immersive authoring. We found that
scaffolded creation using wireframes enhanced the user agency compared to
automatic creation. We also found that manual creation via multimodal
specification offers the highest sense of creativity and agency.

### 摘要 (中文)

沉浸式创作提供了用户直接在虚拟现实（VR）中通过直观操作创建三维场景的直观媒介。最近，生成AI的进步使得可以自动创建逼真的三维布局。然而，我们不清楚如何使用生成AI的能力来支持流畅交互、用户代理和创造力在沉浸式创作中发挥作用。我们引入了VRCopilot，这是一种混合倡议系统，它将预训练的生成AI模型整合到沉浸式创作中，以促进人类与AI的人机协同工作。VRCopilot通过提供多模态互动来支持快速原型设计和迭代，并提供中间表示，如线框，以增强对创建内容的用户控制能力。通过一系列用户研究，我们评估了手动、有指导的和自动创建在沉浸式创作中的潜在性和挑战。我们发现使用线框进行手动生成提高了用户的代理感，而使用多模态指定的手动创建提供了最高级别的创造性代理感。

---

## Game Development as Human-LLM Interaction

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09386v1)

### Abstract (English)

Game development is a highly specialized task that relies on a complex game
engine powered by complex programming languages, preventing many gaming
enthusiasts from handling it. This paper introduces the Interaction-driven Game
Engine (IGE) powered by LLM, which allows everyone to develop a custom game
using natural language through Human-LLM interaction. To enable an LLM to
function as an IGE, we instruct it to perform the following processes in each
turn: (1) $P_{script}$ : configure the game script segment based on the user's
input; (2) $P_{code}$ : generate the corresponding code snippet based on the
game script segment; (3) $P_{utter}$ : interact with the user, including
guidance and feedback. We propose a data synthesis pipeline based on the LLM to
generate game script-code pairs and interactions from a few manually crafted
seed data. We propose a three-stage progressive training strategy to transfer
the dialogue-based LLM to our IGE smoothly. We construct an IGE for poker games
as a case study and comprehensively evaluate it from two perspectives:
interaction quality and code correctness. The code and data are available at
\url{https://github.com/alterego238/IGE}.

### 摘要 (中文)

游戏开发是一项高度专业化的工作，它依赖于由复杂的游戏引擎驱动的复杂的编程语言，这使得许多游戏爱好者难以处理。本文介绍了基于LLM的人机交互驱动的游戏引擎（IGE），该引擎允许每个人通过自然语言来定制自己的游戏。为了使LLM能够作为IGE运行，我们指示它在每次轮换中执行以下过程：(1) $P_{script}$：根据用户的输入配置游戏脚本部分；(2) $P_{code}$：生成相应的代码片段，根据游戏脚本部分；(3) $P_{utter}$：与用户进行交互，包括指导和反馈。我们提出了一种基于LLM的数据合成管道，从几个手动创建的种子数据中生成游戏脚本-代码对和互动。我们提出了一个三阶段渐进式训练策略，以平稳地将对话式LLM转移到我们的IGE。我们将此IG用于扑克游戏作为案例研究，并从两个角度全面评估其性能：交互质量和代码正确性。代码和数据可在https://github.com/alterego238/IGE上找到。

---

## HiAgent_ Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09559v1)

### Abstract (English)

Large Language Model (LLM)-based agents exhibit significant potential across
various domains, operating as interactive systems that process environmental
observations to generate executable actions for target tasks. The effectiveness
of these agents is significantly influenced by their memory mechanism, which
records historical experiences as sequences of action-observation pairs. We
categorize memory into two types: cross-trial memory, accumulated across
multiple attempts, and in-trial memory (working memory), accumulated within a
single attempt. While considerable research has optimized performance through
cross-trial memory, the enhancement of agent performance through improved
working memory utilization remains underexplored. Instead, existing approaches
often involve directly inputting entire historical action-observation pairs
into LLMs, leading to redundancy in long-horizon tasks. Inspired by human
problem-solving strategies, this paper introduces HiAgent, a framework that
leverages subgoals as memory chunks to manage the working memory of LLM-based
agents hierarchically. Specifically, HiAgent prompts LLMs to formulate subgoals
before generating executable actions and enables LLMs to decide proactively to
replace previous subgoals with summarized observations, retaining only the
action-observation pairs relevant to the current subgoal. Experimental results
across five long-horizon tasks demonstrate that HiAgent achieves a twofold
increase in success rate and reduces the average number of steps required by
3.8. Additionally, our analysis shows that HiAgent consistently improves
performance across various steps, highlighting its robustness and
generalizability. Project Page: https://github.com/HiAgent2024/HiAgent .

### 摘要 (中文)

基于大规模语言模型（LLM）的代理在各种领域展现出显著潜力，它们作为交互系统，能够处理环境观察以生成目标任务的可执行动作。这些代理的有效性受到其记忆机制的重大影响，该机制记录历史经验，形成行动与观察对序列。我们把记忆分为两种类型：跨试次记忆和工作记忆，后者在单个尝试中累积。尽管有研究表明通过改善工作记忆利用度可以优化性能，但现有方法往往直接输入整个历史行动-观察对到LMM，导致长时任务冗余。本研究借鉴人类问题解决策略，引入HiAgent框架，该框架利用子目标作为内存块来管理LLM代理的工作记忆，逐层组织。具体来说，HiAgent在生成可执行动作之前引导LMM制定子目标，并允许LMM主动决定替换先前的子目标，仅保留与当前子目标相关的行动-观察对。五种长期任务的实验结果显示，HiAgent的成功率提高了两倍，平均步骤数减少了3.8步。此外，我们的分析显示，HiAgent在各个阶段都能持续改进性能，强调了它的鲁棒性和泛化能力。项目页面：https://github.com/HiAgent2024/HiAgent.

---

## Exploring Wavelet Transformations for Deep Learning-based Machine Condition Diagnosis

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09644v1)

### Abstract (English)

Deep learning (DL) strategies have recently been utilized to diagnose motor
faults by simply analyzing motor phase current signals, offering a less costly
and non-intrusive alternative to vibration sensors. This research transforms
these time-series current signals into time-frequency 2D representations via
Wavelet Transform (WT). The dataset for motor current signals includes 3,750
data points across five categories: one representing normal conditions and four
representing artificially induced faults, each under five different load
conditions: 0, 25, 50, 75, and 100%. The study employs five WT-based
techniques: WT-Amor, WT-Bump, WT-Morse, WSST-Amor, and WSST-Bump. Subsequently,
five DL models adopting prior Convolutional Neural Network (CNN) architecture
were developed and tested using the transformed 2D plots from each method. The
DL models for WT-Amor, WT-Bump, and WT-Morse showed remarkable effectiveness
with peak model accuracy of 90.93, 89.20, and 93.73%, respectively, surpassing
previous 2D-image-based methods that recorded accuracy of 80.25, 74.80, and
82.80% respectively using the identical dataset and validation protocol.
Notably, the WT-Morse approach slightly exceeded the formerly highest ML
technique, achieving a 93.20% accuracy. However, the two WSST methods that
utilized synchrosqueezing techniques faced difficulty accurately classifying
motor faults. The performance of Wavelet-based deep learning methods offers a
compelling alternative for machine condition monitoring.

### 摘要 (中文)

最近，深度学习（DL）策略被用来通过分析电机相电流信号来诊断故障，提供了一种比振动传感器更便宜和非侵入式的替代方法。这项研究将时间序列的电流信号转换成时间-频率2D表示形式，通过小波变换（WT）。电机电流数据集包括五类共3750个样本点：一类代表正常条件，四类代表人为诱导故障，每种在五个不同的负载条件下分别进行：0、25、50、75和100%。该研究采用基于小波变换的五种技术：WT-Amor、WT-Bump、WT-Morse、WSST-Amor和WSST-Bump。随后，使用了每个方法从2D图中生成的5种DL模型，这些模型采用了先前卷积神经网络（CNN）架构。WT-Amor、WT-Bump和WT-Morse显示出了显著的效果，峰值模型准确率为90.93、89.20和93.73%，分别超过了使用相同数据集和验证协议记录下80.25、74.80和82.80%准确率的2D图像基方法。值得注意的是，WT-Morse方法略高于以前最高的机器状态监测技术，达到93.20%的准确性。然而，使用同步压缩技术的两个WSST方法面临的问题是无法准确地识别电机故障。基于小波的深度学习方法提供了令人信服的替代方案用于机器状态监控。

---

## A Comparison of Large Language Model and Human Performance on Random Number Generation Tasks

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09656v1)

### Abstract (English)

Random Number Generation Tasks (RNGTs) are used in psychology for examining
how humans generate sequences devoid of predictable patterns. By adapting an
existing human RNGT for an LLM-compatible environment, this preliminary study
tests whether ChatGPT-3.5, a large language model (LLM) trained on
human-generated text, exhibits human-like cognitive biases when generating
random number sequences. Initial findings indicate that ChatGPT-3.5 more
effectively avoids repetitive and sequential patterns compared to humans, with
notably lower repeat frequencies and adjacent number frequencies. Continued
research into different models, parameters, and prompting methodologies will
deepen our understanding of how LLMs can more closely mimic human random
generation behaviors, while also broadening their applications in cognitive and
behavioral science research.

### 摘要 (中文)

随机数生成任务（RNGT）在心理学中用于研究人类如何生成没有预测模式的序列。通过为适合LLM环境的人类RNGT适应一个现有的人类RNGT，这项初步研究表明，ChatGPT-3.5，由人类生成的大型语言模型（LLM）训练而成，当生成随机数序列时是否表现出人类认知偏见。初步发现表明，ChatGPT-3.5比人类更有效地避免重复和顺序模式，其频率显著低于人类，并且相邻数字频率更低。对不同模型、参数和提示方法的研究将继续加深我们对如何使LLM更紧密地模仿人类随机生成行为的理解，同时扩大它们在认知和行为科学研究中的应用范围。

---

## Multi-Agent Reinforcement Learning for Autonomous Driving_ A Survey

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09675v1)

### Abstract (English)

Reinforcement Learning (RL) is a potent tool for sequential decision-making
and has achieved performance surpassing human capabilities across many
challenging real-world tasks. As the extension of RL in the multi-agent system
domain, multi-agent RL (MARL) not only need to learn the control policy but
also requires consideration regarding interactions with all other agents in the
environment, mutual influences among different system components, and the
distribution of computational resources. This augments the complexity of
algorithmic design and poses higher requirements on computational resources.
Simultaneously, simulators are crucial to obtain realistic data, which is the
fundamentals of RL. In this paper, we first propose a series of metrics of
simulators and summarize the features of existing benchmarks. Second, to ease
comprehension, we recall the foundational knowledge and then synthesize the
recently advanced studies of MARL-related autonomous driving and intelligent
transportation systems. Specifically, we examine their environmental modeling,
state representation, perception units, and algorithm design. Conclusively, we
discuss open challenges as well as prospects and opportunities. We hope this
paper can help the researchers integrate MARL technologies and trigger more
insightful ideas toward the intelligent and autonomous driving.

### 摘要 (中文)

强化学习（RL）是解决序列决策问题的有效工具，它在许多具有挑战性的现实世界任务中实现了性能超越人类能力。作为RL扩展的多智能体系统（MARL），除了需要学习控制策略外，还需要考虑与环境中的所有其他代理之间的交互、不同系统组件间的相互影响以及计算资源的分布等。这增加了算法设计的复杂性，并对计算资源提出了更高的要求。同时，仿真器对于获得真实数据至关重要，这是RL的基础。本论文首先提出了一系列模拟器指标，并总结了现有基准集的特点。其次，为了使理解变得容易，我们回顾了基础知识，然后综合了有关多智能体自主驾驶和智能化交通系统的最近先进研究。具体来说，我们将审查它们的环境建模、状态表示、感知单元和算法设计。最后，我们讨论了开放挑战以及前景和机遇。我们希望这篇论文可以帮助研究人员整合MARL技术，并激发更多关于智能和自主驾驶的见解。

---

## Paired Completion_ Flexible Quantification of Issue-framing at Scale with LLMs

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09742v1)

### Abstract (English)

Detecting and quantifying issue framing in textual discourse - the
perspective one takes to a given topic (e.g. climate science vs. denialism,
misogyny vs. gender equality) - is highly valuable to a range of end-users from
social and political scientists to program evaluators and policy analysts.
However, conceptual framing is notoriously challenging for automated natural
language processing (NLP) methods since the words and phrases used by either
`side' of an issue are often held in common, with only subtle stylistic
flourishes separating their use. Here we develop and rigorously evaluate new
detection methods for issue framing and narrative analysis within large text
datasets. By introducing a novel application of next-token log probabilities
derived from generative large language models (LLMs) we show that issue framing
can be reliably and efficiently detected in large corpora with only a few
examples of either perspective on a given issue, a method we call `paired
completion'. Through 192 independent experiments over three novel, synthetic
datasets, we evaluate paired completion against prompt-based LLM methods and
labelled methods using traditional NLP and recent LLM contextual embeddings. We
additionally conduct a cost-based analysis to mark out the feasible set of
performant methods at production-level scales, and a model bias analysis.
Together, our work demonstrates a feasible path to scalable, accurate and
low-bias issue-framing in large corpora.

### 摘要 (中文)

检测和量化文本中议题框架-即一个人对给定主题（如气候变化反对派与否认主义、性别歧视与性别平等）所持立场的价值非常高，这适用于社会和政治科学家等从多个角度的使用者。然而，自动自然语言处理（NLP）方法对于概念性框架的检测一直是一个挑战，因为一方的观点中的词或短语往往处于共用状态，而它们的使用仅在细微的风格差异上有所不同。我们在此开发并严格评估了针对大型文本数据集的新议题框架检测和叙事分析方法。通过引入来自生成式大规模语言模型（LLM）的下一个单词概率新应用，我们可以可靠高效地检测大样本中任何一方关于给定问题的看法，这种方法称为“配对完成”。通过对三个新颖的合成数据集进行192次独立实验，在传统NLP和最近的LLM上下文嵌入标记下，我们比较了配对完成法与基于提示的LLM方法，并评估了其与其他基于传统的NLP方法相比。此外，我们还进行了成本分析以标出生产级别可行的方法的有效集合，以及模型偏见分析。综上所述，我们的工作展示了在大型数据集中实现可扩展、准确且低偏差议题框架的可行性路径。

---

## Propagating the prior from shallow to deep with a pre-trained velocity-model Generative Transformer network

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09767v1)

### Abstract (English)

Building subsurface velocity models is essential to our goals in utilizing
seismic data for Earth discovery and exploration, as well as monitoring. With
the dawn of machine learning, these velocity models (or, more precisely, their
distribution) can be stored accurately and efficiently in a generative model.
These stored velocity model distributions can be utilized to regularize or
quantify uncertainties in inverse problems, like full waveform inversion.
However, most generators, like normalizing flows or diffusion models, treat the
image (velocity model) uniformly, disregarding spatial dependencies and
resolution changes with respect to the observation locations. To address this
weakness, we introduce VelocityGPT, a novel implementation that utilizes
Transformer decoders trained autoregressively to generate a velocity model from
shallow subsurface to deep. Owing to the fact that seismic data are often
recorded on the Earth's surface, a top-down generator can utilize the inverted
information in the shallow as guidance (prior) to generating the deep. To
facilitate the implementation, we use an additional network to compress the
velocity model. We also inject prior information, like well or structure
(represented by a migration image) to generate the velocity model. Using
synthetic data, we demonstrate the effectiveness of VelocityGPT as a promising
approach in generative model applications for seismic velocity model building.

### 摘要 (中文)

利用地震数据进行地球发现和勘探，以及监测，构建地层速度模型是至关重要的。随着机器学习的兴起，这些速度模型（或更准确地说，它们的分布）可以精确、高效地存储在生成模型中。这些存储的速率模型分布可用于对逆问题进行正则化或量化不确定性，例如全波形反演。然而，大多数生成器，如规范化流或扩散模型，都以均匀的方式处理图像（速率模型），忽视空间依赖性和观测位置分辨率的变化。为了应对这一弱点，我们引入了VelocityGPT，这是一种新颖的实现，它使用自回归训练的变体编码器从浅层的地层到深部生成速率模型。由于地震数据通常记录于地球表面，顶部生成器可以从浅层信息中获取指导（先验），从而在生成深层时利用。为了促进实施，我们还使用了一个额外的网络来压缩速率模型。我们也注入了像井或结构（由迁移图像表示）等前信息，以生成速率模型。通过合成数据，我们展示了VelocityGPT作为一种有望用于生成模型应用的地质速率模型构建方法的有效性。

---

## GoNoGo_ An Efficient LLM-based Multi-Agent System for Streamlining Automotive Software Release Decision-Making

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09785v1)

### Abstract (English)

Traditional methods for making software deployment decisions in the
automotive industry typically rely on manual analysis of tabular software test
data. These methods often lead to higher costs and delays in the software
release cycle due to their labor-intensive nature. Large Language Models (LLMs)
present a promising solution to these challenges. However, their application
generally demands multiple rounds of human-driven prompt engineering, which
limits their practical deployment, particularly for industrial end-users who
need reliable and efficient results. In this paper, we propose GoNoGo, an LLM
agent system designed to streamline automotive software deployment while
meeting both functional requirements and practical industrial constraints.
Unlike previous systems, GoNoGo is specifically tailored to address
domain-specific and risk-sensitive systems. We evaluate GoNoGo's performance
across different task difficulties using zero-shot and few-shot examples taken
from industrial practice. Our results show that GoNoGo achieves a 100% success
rate for tasks up to Level 2 difficulty with 3-shot examples, and maintains
high performance even for more complex tasks. We find that GoNoGo effectively
automates decision-making for simpler tasks, significantly reducing the need
for manual intervention. In summary, GoNoGo represents an efficient and
user-friendly LLM-based solution currently employed in our industrial partner's
company to assist with software release decision-making, supporting more
informed and timely decisions in the release process for risk-sensitive vehicle
systems.

### 摘要 (中文)

汽车行业传统方法在软件部署决策中通常依赖于对表格形式的软件测试数据的人工分析。这些方法由于其劳动密集型的特点，往往会导致软件发布周期的成本和延迟增加。大型语言模型（LLM）为解决这些问题提供了很好的解决方案。然而，它们的应用一般需要多次人工驱动的提示工程，这限制了他们的实际部署，特别是对于工业用户来说，他们需要可靠且高效的成果。在这篇论文中，我们提出了一个名为GoNoGo的LLM代理系统，旨在通过简化汽车软件部署来满足功能要求和工业界的具体约束。与以往的系统不同，GoNoGo专门针对特定领域和风险敏感系统进行定制。我们使用零样本和少量样本从工业实践中的例子评估GoNoGo的表现。我们的结果显示，在任务难度达到第二级时，GoNoGo可以实现100％的成功率，并且即使是在更复杂的任务中也保持了很高的性能。我们发现GoNoGo有效地自动化了简单任务的决策，极大地减少了手动干预的需求。总之，GoNoGo代表了一个目前被我们的工业合作伙伴公司用于辅助软件发布决策的高效且友好的LLM解决方案，支持风险敏感车辆系统的更加明智和及时的决策过程。

---

## Uncertainty Quantification of Pre-Trained and Fine-Tuned Surrogate Models using Conformal Prediction

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09881v1)

### Abstract (English)

Data-driven surrogate models have shown immense potential as quick,
inexpensive approximations to complex numerical and experimental modelling
tasks. However, most surrogate models characterising physical systems do not
quantify their uncertainty, rendering their predictions unreliable, and needing
further validation. Though Bayesian approximations offer some solace in
estimating the error associated with these models, they cannot provide they
cannot provide guarantees, and the quality of their inferences depends on the
availability of prior information and good approximations to posteriors for
complex problems. This is particularly pertinent to multi-variable or
spatio-temporal problems. Our work constructs and formalises a conformal
prediction framework that satisfies marginal coverage for spatio-temporal
predictions in a model-agnostic manner, requiring near-zero computational
costs. The paper provides an extensive empirical study of the application of
the framework to ascertain valid error bars that provide guaranteed coverage
across the surrogate model's domain of operation. The application scope of our
work extends across a large range of spatio-temporal models, ranging from
solving partial differential equations to weather forecasting. Through the
applications, the paper looks at providing statistically valid error bars for
deterministic models, as well as crafting guarantees to the error bars of
probabilistic models. The paper concludes with a viable conformal prediction
formalisation that provides guaranteed coverage of the surrogate model,
regardless of model architecture, and its training regime and is unbothered by
the curse of dimensionality.

### 摘要 (中文)

数据驱动的元模型因其快速、低成本的优势，在复杂数值和实验建模任务中显示出巨大的潜力。然而，大多数描述物理系统的元模型无法量化不确定性，这使得它们的预测不可靠，并需要进一步验证。尽管贝叶斯近似在估计这些模型相关误差方面提供了一定安慰，但它们不能提供保证，并且对复杂问题的后验估计的质量依赖于可用先验信息和良好的后验估计。特别是对于多变量或时空问题尤为关键。我们的工作构建并形式化了一个满足空间-时间预测边缘覆盖的框架，以一种模型无关的方式，在很大程度上不需要零计算成本。该论文提供了关于框架应用的广泛实证研究，以确定有效误差带，确保对元模型操作域内的保证覆盖率。我们的工作范围涵盖了多种空间-时间模型，从解决偏微分方程到天气预报。通过应用，该文探讨了为概率模型提供统计有效误差带的能力，并提出了对决策模型误差带的保证。最后，本文提出了一种可靠的元模型元形法，无论模型架构如何，都能提供对元模型的保证覆盖，并对其训练模式和维度悖论不敏感。

---

## Synthesis of Reward Machines for Multi-Agent Equilibrium Design _Full Version_

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10074v1)

### Abstract (English)

Mechanism design is a well-established game-theoretic paradigm for designing
games to achieve desired outcomes. This paper addresses a closely related but
distinct concept, equilibrium design. Unlike mechanism design, the designer's
authority in equilibrium design is more constrained; she can only modify the
incentive structures in a given game to achieve certain outcomes without the
ability to create the game from scratch. We study the problem of equilibrium
design using dynamic incentive structures, known as reward machines. We use
weighted concurrent game structures for the game model, with goals (for the
players and the designer) defined as mean-payoff objectives. We show how reward
machines can be used to represent dynamic incentives that allocate rewards in a
manner that optimises the designer's goal. We also introduce the main decision
problem within our framework, the payoff improvement problem. This problem
essentially asks whether there exists a dynamic incentive (represented by some
reward machine) that can improve the designer's payoff by more than a given
threshold value. We present two variants of the problem: strong and weak. We
demonstrate that both can be solved in polynomial time using a Turing machine
equipped with an NP oracle. Furthermore, we also establish that these variants
are either NP-hard or coNP-hard. Finally, we show how to synthesise the
corresponding reward machine if it exists.

### 摘要 (中文)

机制设计是博弈论的一个成熟范例，用于设计游戏来实现预期的结果。本文研究了与之密切相关但不同的概念——均衡设计。与机制设计不同的是，在均衡设计中，设计师的权力受到更严格的限制；她只能在给定的游戏内修改激励结构以达到特定结果，而无法从零开始创建该游戏。我们通过动态激励结构研究均衡设计问题，这些奖励机器被称为奖励机。我们使用权重并发游戏结构作为游戏模型，目标（对于玩家和设计师）定义为平均收益目标。我们证明了可以通过奖励机器来代表动态激励，这种动态激励可以根据优化设计师目标的方式分配奖励。我们也介绍了我们在框架中的主要决策问题——支付改善问题。这个问题是基本的问题，即是否存在一个动态激励（由一些奖励机器表示），可以超过给定阈值提高设计师的支付量。我们提出了两个变体问题：强和弱。我们可以使用这两种变体用一台具有NP查询的Turing机器解决这个问题。此外，我们还确定了这两个变体要么是NP困难或coNP困难。最后，我们展示了如果存在，则如何合成对应的奖励机器。

---

## Threshold Filtering Packing for Supervised Fine-Tuning_ Training Related Samples within Packs

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09327v1)

### Abstract (English)

Packing for Supervised Fine-Tuning (SFT) in autoregressive models involves
concatenating data points of varying lengths until reaching the designed
maximum length to facilitate GPU processing. However, randomly concatenating
data points and feeding them into an autoregressive transformer can lead to
cross-contamination of sequences due to the significant difference in their
subject matter. The mainstream approaches in SFT ensure that each token in the
attention calculation phase only focuses on tokens within its own short
sequence, without providing additional learning signals for the preceding
context. To address these challenges, we introduce Threshold Filtering Packing
(TFP), a method that selects samples with related context while maintaining
sufficient diversity within the same pack. Our experiments show that TFP offers
a simple-to-implement and scalable approach that significantly enhances SFT
performance, with observed improvements of up to 7\% on GSM8K, 4\% on
HumanEval, and 15\% on the adult-census-income dataset.

### 摘要 (中文)

在自动回归模型中进行监督微调（Supervised Fine-tuning，简称SFT）涉及将不同长度的数据点连接在一起，直到达到设计的最大长度，以便于GPU处理。然而，随机连接数据点并将其输入到自回归变换器中可能会导致序列交叉污染，因为它们的主题差异很大。在SFT中，主流方法确保注意力计算阶段中的每个令牌只关注其自身短序列内的令牌，而没有为先前上下文提供额外的学习信号。为了解决这些问题，我们引入阈值过滤打包（Threshold Filtering Packing，简称TFP），这是一种选择相关上下文的样本，并且在同组内保持足够的多样性的方法。我们的实验表明，TFP提供了简单易实施和可扩展的方法，显著提高了SFT的表现，在GSM8K上观察到的改进高达7%，在HumanEval上观察到的改进为4%，在成人人口收入数据集上的观察到的改进为15%。

---

## Mitigating Noise Detriment in Differentially Private Federated Learning with Model Pre-training

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09478v1)

### Abstract (English)

Pre-training exploits public datasets to pre-train an advanced machine
learning model, so that the model can be easily tuned to adapt to various
downstream tasks. Pre-training has been extensively explored to mitigate
computation and communication resource consumption. Inspired by these
advantages, we are the first to explore how model pre-training can mitigate
noise detriment in differentially private federated learning (DPFL). DPFL is
upgraded from federated learning (FL), the de-facto standard for privacy
preservation when training the model across multiple clients owning private
data. DPFL introduces differentially private (DP) noises to obfuscate model
gradients exposed in FL, which however can considerably impair model accuracy.
In our work, we compare head fine-tuning (HT) and full fine-tuning (FT), which
are based on pre-training, with scratch training (ST) in DPFL through a
comprehensive empirical study. Our experiments tune pre-trained models
(obtained by pre-training on ImageNet-1K) with CIFAR-10, CHMNIST and
Fashion-MNIST (FMNIST) datasets, respectively. The results demonstrate that HT
and FT can significantly mitigate noise influence by diminishing gradient
exposure times. In particular, HT outperforms FT when the privacy budget is
tight or the model size is large. Visualization and explanation study further
substantiates our findings. Our pioneering study introduces a new perspective
on enhancing DPFL and expanding its practical applications.

### 摘要 (中文)

预训练利用公开数据对先进的人工智能模型进行预训练，从而使模型能够轻松调整以适应各种下游任务。预训练已经广泛探索来缓解计算和通信资源消耗。受到这些优势的启发，我们是第一个探索如何通过预训练可以缓解不同私有化联邦学习（DPFL）中的噪声损失。DPFL是从联邦学习（FL），作为在多个拥有私人数据的客户端上训练模型时隐私保护的事实标准升级而来。DPFL引入了不同的私密性（DP）噪音来混淆暴露于FL中的模型梯度，然而这却会导致模型精度显著下降。在我们的工作中，我们比较了基于预训练的头部微调（HT）、全微调（FT）和在DPFL中直接从头开始训练（ST）。通过对全面实证研究，我们在CIFAR-10、CHMNIST和Fashion-MNIST（FMNIST）等数据集上分别对预训练获得的模型进行了微调。结果显示，HT和FT可以通过减少梯度暴露次数显著减轻噪声影响。特别是，HT在隐私预算紧或模型大小大的时候优于FT。可视化和解释性研究进一步验证了我们的发现。我们的开创性研究表明了一种新的视角来增强DPFL，并扩大其实际应用范围。

---

## Byzantine-resilient Federated Learning Employing Normalized Gradients on Non-IID Datasets

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09539v1)

### Abstract (English)

In practical federated learning (FL) systems, the presence of malicious
Byzantine attacks and data heterogeneity often introduces biases into the
learning process. However, existing Byzantine-robust methods typically only
achieve a compromise between adaptability to different loss function types
(including both strongly convex and non-convex) and robustness to heterogeneous
datasets, but with non-zero optimality gap. Moreover, this compromise often
comes at the cost of high computational complexity for aggregation, which
significantly slows down the training speed. To address this challenge, we
propose a federated learning approach called Federated Normalized Gradients
Algorithm (Fed-NGA). Fed-NGA simply normalizes the uploaded local gradients to
be unit vectors before aggregation, achieving a time complexity of
$\mathcal{O}(pM)$, where $p$ represents the dimension of model parameters and
$M$ is the number of participating clients. This complexity scale achieves the
best level among all the existing Byzantine-robust methods. Furthermore,
through rigorous proof, we demonstrate that Fed-NGA transcends the trade-off
between adaptability to loss function type and data heterogeneity and the
limitation of non-zero optimality gap in existing literature. Specifically,
Fed-NGA can adapt to both non-convex loss functions and non-IID datasets
simultaneously, with zero optimality gap at a rate of $\mathcal{O}
(1/T^{\frac{1}{2} - \delta})$, where T is the iteration number and $\delta \in
(0,\frac{1}{2})$. In cases where the loss function is strongly convex, the zero
optimality gap achieving rate can be improved to be linear. Experimental
results provide evidence of the superiority of our proposed Fed-NGA on time
complexity and convergence performance over baseline methods.

### 摘要 (中文)

在实际的联邦学习（FL）系统中，恶意拜斯廷尼攻击和数据异质性经常引入到学习过程中。然而，现有的拜斯廷尼鲁棒方法通常只能在不同损失函数类型（包括强凸性和非凸性）和异质数据上实现适应性的妥协，但存在零优化差距。此外，这种妥协往往伴随着高计算复杂度的聚合，这极大地降低了训练速度。为了应对这一挑战，我们提出了一种称为联邦规范化梯度算法（Fed-NGA）的联邦学习方法。在聚合之前，Fed-NGA简单地对上传的局部梯度进行归一化，使其成为单位向量，从而达到时间复杂度为$\mathcal{O}(pM)$的规模，其中$p$代表模型参数的维度，$M$是参与客户的数量。这个复杂度级别在现有拜斯廷尼鲁棒方法中实现了最佳水平。此外，通过严格的证明，我们证明了Fed-NGA超越了现有文献中关于适应损失函数类型的灵活性与零优化差距限制之间的折衷。具体来说，Fed-NGA可以在同时适应非凸损失函数和非等值数据的情况下保持零优化差距，并以率$\mathcal{O}(1/T^{\frac{1}{2}-\delta})$的速度收敛于一个理想点，其中T表示迭代次数，$\delta \in (0,\frac{1}{2})$。当损失函数强烈凸时，可以提高零优化差距的收敛率为线性。在试验结果中提供了Fed-NGA优于基准方法的时间复杂性和收敛性能的证据。

---

## On the Necessity of World Knowledge for Mitigating Missing Labels in Extreme Classification

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09585v1)

### Abstract (English)

Extreme Classification (XC) aims to map a query to the most relevant
documents from a very large document set. XC algorithms used in real-world
applications learn this mapping from datasets curated from implicit feedback,
such as user clicks. However, these datasets inevitably suffer from missing
labels. In this work, we observe that systematic missing labels lead to missing
knowledge, which is critical for accurately modelling relevance between queries
and documents. We formally show that this absence of knowledge cannot be
recovered using existing methods such as propensity weighting and data
imputation strategies that solely rely on the training dataset. While LLMs
provide an attractive solution to augment the missing knowledge, leveraging
them in applications with low latency requirements and large document sets is
challenging. To incorporate missing knowledge at scale, we propose SKIM
(Scalable Knowledge Infusion for Missing Labels), an algorithm that leverages a
combination of small LM and abundant unstructured meta-data to effectively
mitigate the missing label problem. We show the efficacy of our method on
large-scale public datasets through exhaustive unbiased evaluation ranging from
human annotations to simulations inspired from industrial settings. SKIM
outperforms existing methods on Recall@100 by more than 10 absolute points.
Additionally, SKIM scales to proprietary query-ad retrieval datasets containing
10 million documents, outperforming contemporary methods by 12% in offline
evaluation and increased ad click-yield by 1.23% in an online A/B test
conducted on a popular search engine. We release our code, prompts, trained XC
models and finetuned SLMs at: https://github.com/bicycleman15/skim

### 摘要 (中文)

极端分类（XC）的目标是将查询映射到从非常大的文档集中的最相关文档。在实际应用中，用于机器学习的算法是从从隐式反馈收集的用户点击数据中学习这一映射，这些数据不可避免地会缺失标签。在这项工作中，我们观察到系统性缺失标签会导致缺失知识，这对准确建模查询和文档之间的相关性至关重要。我们正式证明了这种缺乏知识无法通过现有方法如倾向权重和数据填充策略来恢复，这些方法仅依赖于训练数据。尽管语言模型提供了在缺失知识方面提供吸引力的解决方案，但将其应用于具有低延迟要求和大量文档集的应用程序中存在挑战。为了以大规模的方式引入缺失知识，我们提出了一种名为SKIM（大规模知识注入缺失标签）的方法，该方法利用小的语言模型和大量的无结构元数据有效地缓解缺失标签问题。我们在大型公共数据集中进行详细、公正的评估，包括人类注释到启发工业环境模拟的数据驱动的实验。SKIM在召回@100上比现有的方法高出超过10个绝对点。此外，SKIM能够在含有1亿文档的私有查询检索数据集上实现12%的在线评价，在搜索引擎上的A/B测试中增加了1.23%的广告点击率。

我们将代码、提示、训练的XC模型以及调优的SLM发布在https://github.com/bicycleman15/skim上。

---

## Community-Centric Graph Unlearning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09705v1)

### Abstract (English)

Graph unlearning technology has become increasingly important since the
advent of the `right to be forgotten' and the growing concerns about the
privacy and security of artificial intelligence. Graph unlearning aims to
quickly eliminate the effects of specific data on graph neural networks (GNNs).
However, most existing deterministic graph unlearning frameworks follow a
balanced partition-submodel training-aggregation paradigm, resulting in a lack
of structural information between subgraph neighborhoods and redundant
unlearning parameter calculations. To address this issue, we propose a novel
Graph Structure Mapping Unlearning paradigm (GSMU) and a novel method based on
it named Community-centric Graph Eraser (CGE). CGE maps community subgraphs to
nodes, thereby enabling the reconstruction of a node-level unlearning operation
within a reduced mapped graph. CGE makes the exponential reduction of both the
amount of training data and the number of unlearning parameters. Extensive
experiments conducted on five real-world datasets and three widely used GNN
backbones have verified the high performance and efficiency of our CGE method,
highlighting its potential in the field of graph unlearning.

### 摘要 (中文)

自从“遗忘权”运动的兴起和对人工智能隐私和安全日益增长的关注以来，图无学技术变得越来越重要。图无学的目标是快速消除特定数据对图神经网络（GNN）的影响。然而，现有的大多数定性图无学框架都遵循均衡划分-子模型训练-聚合模式，这导致了子节点之间的图形结构信息缺乏以及冗余的无学参数计算。为了解决这个问题，我们提出了一个新的图结构映射无学范式(GSMU)，并基于它提出了一种名为社区中心化图擦除(CGE)的新方法。CGE根据社区子图将节点映射到节点上，从而在减少映射后的图中重建一个节点级别的无学操作。CGE使训练数据量和无学参数的数量同时减少了极大的数量级。在五个真实世界数据集和三种广泛使用的GNN骨架上进行的大量实验验证了CGE方法的高性能和高效性，强调了其在图无学领域的潜力。

---

## Icing on the Cake_ Automatic Code Summarization at Ericsson

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09735v1)

### Abstract (English)

This paper presents our findings on the automatic summarization of Java
methods within Ericsson, a global telecommunications company. We evaluate the
performance of an approach called Automatic Semantic Augmentation of Prompts
(ASAP), which uses a Large Language Model (LLM) to generate leading summary
comments for Java methods. ASAP enhances the $LLM's$ prompt context by
integrating static program analysis and information retrieval techniques to
identify similar exemplar methods along with their developer-written Javadocs,
and serves as the baseline in our study. In contrast, we explore and compare
the performance of four simpler approaches that do not require static program
analysis, information retrieval, or the presence of exemplars as in the ASAP
method. Our methods rely solely on the Java method body as input, making them
lightweight and more suitable for rapid deployment in commercial software
development environments. We conducted experiments on an Ericsson software
project and replicated the study using two widely-used open-source Java
projects, Guava and Elasticsearch, to ensure the reliability of our results.
Performance was measured across eight metrics that capture various aspects of
similarity. Notably, one of our simpler approaches performed as well as or
better than the ASAP method on both the Ericsson project and the open-source
projects. Additionally, we performed an ablation study to examine the impact of
method names on Javadoc summary generation across our four proposed approaches
and the ASAP method. By masking the method names and observing the generated
summaries, we found that our approaches were statistically significantly less
influenced by the absence of method names compared to the baseline. This
suggests that our methods are more robust to variations in method names and may
derive summaries more comprehensively from the method body than the ASAP
approach.

### 摘要 (中文)

这篇论文报告了我们对Ericsson公司中自动摘要Java方法的发现。我们评估了一种名为Automatic Semantic Augmentation of Prompts（ASAP）的方法，该方法使用大型语言模型(LLM)生成Java方法的领先总结评论。ASAP通过整合静态程序分析和信息检索技术来增强prompt上下文，识别与相似示例方法及其开发人员编写Javadocs相关的类似示例方法，并作为我们的研究的基础。相比之下，我们探索并比较了四个更简单的不需要静态程序分析、信息检索或存在示例的简化的方法。我们的方法仅依赖于Java方法体作为输入，使其轻量级且更适合商业软件开发环境中的快速部署。我们在Ericsson软件项目上进行了实验，并在两个广泛使用的开源Java项目Guava和Elasticsearch上进行复制，以确保结果的可靠性。性能测量涵盖了八个指标，这些指标捕捉了各种方面的相似性。值得注意的是，在Ericsson项目和开源项目上，我们的简单方法的表现与ASAP方法大致相同或更好。此外，我们还进行了一个去除法实验，以检验方法名称对Javadoc摘要生成的影响。通过对我们的四种提议方法和ASAP方法的masking方法名并观察生成的摘要，我们发现我们的方法比基线对方法名称的缺失影响更小，这表明我们的方法比ASAP方法更加抗变性和从方法体中更全面地抽取摘要。

---

## Baby Bear_ Seeking a Just Right Rating Scale for Scalar Annotations

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09765v1)

### Abstract (English)

Our goal is a mechanism for efficiently assigning scalar ratings to each of a
large set of elements. For example, "what percent positive or negative is this
product review?" When sample sizes are small, prior work has advocated for
methods such as Best Worst Scaling (BWS) as being more robust than direct
ordinal annotation ("Likert scales"). Here we first introduce IBWS, which
iteratively collects annotations through Best-Worst Scaling, resulting in
robustly ranked crowd-sourced data. While effective, IBWS is too expensive for
large-scale tasks. Using the results of IBWS as a best-desired outcome, we
evaluate various direct assessment methods to determine what is both
cost-efficient and best correlating to a large scale BWS annotation strategy.
Finally, we illustrate in the domains of dialogue and sentiment how these
annotations can support robust learning-to-rank models.

### 摘要 (中文)

我们的目标是设计一个机制，以高效地为一组大型元素中的每个元素分配标量评分。例如，“这个产品评论的正面和负面比例是多少？”当样本大小较小时，先前的工作建议使用Best Worst Scaling（BWS）等方法比直接顺序标注更具鲁棒性（“利克特等级表”）。在这里，我们首先引入IBWS，它通过最佳-最差标注进行迭代收集注释，从而产生稳健的众包数据。尽管有效，但IBWS对于大规模任务来说过于昂贵。利用IBWS的结果作为最佳期望结果，我们评估各种直接评价方法来确定既成本效益高又与大规模BWS标注策略相匹配的是什么。最后，我们展示了在对话和情感分析等领域中这些注释如何支持稳健的学习到排名模型。

---

## Faster Adaptive Decentralized Learning Algorithms

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09775v1)

### Abstract (English)

Decentralized learning recently has received increasing attention in machine
learning due to its advantages in implementation simplicity and system
robustness, data privacy. Meanwhile, the adaptive gradient methods show
superior performances in many machine learning tasks such as training neural
networks. Although some works focus on studying decentralized optimization
algorithms with adaptive learning rates, these adaptive decentralized
algorithms still suffer from high sample complexity. To fill these gaps, we
propose a class of faster adaptive decentralized algorithms (i.e., AdaMDOS and
AdaMDOF) for distributed nonconvex stochastic and finite-sum optimization,
respectively. Moreover, we provide a solid convergence analysis framework for
our methods. In particular, we prove that our AdaMDOS obtains a near-optimal
sample complexity of $\tilde{O}(\epsilon^{-3})$ for finding an
$\epsilon$-stationary solution of nonconvex stochastic optimization. Meanwhile,
our AdaMDOF obtains a near-optimal sample complexity of
$O(\sqrt{n}\epsilon^{-2})$ for finding an $\epsilon$-stationary solution of
nonconvex finite-sum optimization, where $n$ denotes the sample size. To the
best of our knowledge, our AdaMDOF algorithm is the first adaptive
decentralized algorithm for nonconvex finite-sum optimization. Some
experimental results demonstrate efficiency of our algorithms.

### 摘要 (中文)

分布式学习最近由于其实施简单性和系统鲁棒性、数据隐私的优势而受到机器学习的广泛关注。同时，适应梯度方法在许多机器学习任务中表现出色，如训练神经网络。虽然一些工作专注于研究分布式优化算法中的自适应学习率，但这些自适应分布式算法仍然存在高样本复杂性的缺陷。为了填补这些空白，我们提出了一类更快的适应分布式非凸随机和有限和求和优化（即AdaMDOS和AdaMDOF）的快速适应分布式算法。此外，我们提供了对我们的方法的坚实收敛分析框架。特别是，我们证明了我们的AdaMDOS在寻找非凸随机优化问题的近似最优采样复杂度为$\tilde{O}(\epsilon^{-3})$。同时，我们的AdaMDOF在寻找非凸有限和求和优化问题的近似最优采样复杂度为$O(\sqrt{n}\epsilon^{-2})$，其中$n$表示样本大小。到目前为止，我们的AdaMDOF算法是唯一的一个适应分布式非凸有限和求和优化的自适应分布式算法。实验结果表明，我们的算法具有效率。

---

## A Population-to-individual Tuning Framework for Adapting Pretrained LM to On-device User Intent Prediction

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09815v1)

### Abstract (English)

Mobile devices, especially smartphones, can support rich functions and have
developed into indispensable tools in daily life. With the rise of generative
AI services, smartphones can potentially transform into personalized
assistants, anticipating user needs and scheduling services accordingly.
Predicting user intents on smartphones, and reflecting anticipated activities
based on past interactions and context, remains a pivotal step towards this
vision. Existing research predominantly focuses on specific domains, neglecting
the challenge of modeling diverse event sequences across dynamic contexts.
Leveraging pre-trained language models (PLMs) offers a promising avenue, yet
adapting PLMs to on-device user intent prediction presents significant
challenges. To address these challenges, we propose PITuning, a
Population-to-Individual Tuning framework. PITuning enhances common pattern
extraction through dynamic event-to-intent transition modeling and addresses
long-tailed preferences via adaptive unlearning strategies. Experimental
results on real-world datasets demonstrate PITuning's superior intent
prediction performance, highlighting its ability to capture long-tailed
preferences and its practicality for on-device prediction scenarios.

### 摘要 (中文)

移动设备，尤其是智能手机，可以支持丰富的功能，并且已经成为日常生活中不可或缺的工具。随着生成式人工智能服务的发展，智能手机可能被转化为个性化的助手，根据用户需求预测并相应地安排服务。预测智能手机上的用户意图，并基于过去交互和上下文进行预期活动的反射，是实现这一愿景的关键步骤。现有研究主要集中在特定领域，忽略了动态上下文下跨动态场景中事件序列建模的挑战。利用预训练语言模型（PLM）提供了有利途径，但将其适配到离线用户意图预测呈现了重大挑战。为了应对这些挑战，我们提出PITuning，这是一种人口到个体调优框架。通过动态事件到意图转换建模来增强共同模式提取，通过自适应未学习策略解决长尾偏好。在真实世界数据集上进行的实验结果表明，PITuning的意向预测性能优于现有方法，强调其捕获长期偏好的能力以及它对于离线预测场景的实用性。

---

## Mitigating the Stability-Plasticity Dilemma in Adaptive Train Scheduling with Curriculum-Driven Continual DQN Expansion

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09838v1)

### Abstract (English)

A continual learning agent builds on previous experiences to develop
increasingly complex behaviors by adapting to non-stationary and dynamic
environments while preserving previously acquired knowledge. However, scaling
these systems presents significant challenges, particularly in balancing the
preservation of previous policies with the adaptation of new ones to current
environments. This balance, known as the stability-plasticity dilemma, is
especially pronounced in complex multi-agent domains such as the train
scheduling problem, where environmental and agent behaviors are constantly
changing, and the search space is vast. In this work, we propose addressing
these challenges in the train scheduling problem using curriculum learning. We
design a curriculum with adjacent skills that build on each other to improve
generalization performance. Introducing a curriculum with distinct tasks
introduces non-stationarity, which we address by proposing a new algorithm:
Continual Deep Q-Network (DQN) Expansion (CDE). Our approach dynamically
generates and adjusts Q-function subspaces to handle environmental changes and
task requirements. CDE mitigates catastrophic forgetting through EWC while
ensuring high plasticity using adaptive rational activation functions.
Experimental results demonstrate significant improvements in learning
efficiency and adaptability compared to RL baselines and other adapted methods
for continual learning, highlighting the potential of our method in managing
the stability-plasticity dilemma in the adaptive train scheduling setting.

### 摘要 (中文)

持续学习的代理通过适应非静止和动态环境来发展越来越复杂的行为，并且在保留之前获得的知识的同时，不断从以前的经验中构建。然而，这些系统的扩展面临着巨大的挑战，尤其是在平衡保存先前政策与根据当前环境适应新政策之间。这种平衡，称为稳定性-可塑性悖论，在复杂多智能体领域尤其明显，如火车调度问题，其中环境和代理行为是不断变化的，搜索空间很大。在本工作中，我们提出使用课程学习方法解决这个挑战。我们设计了一个连续的课程，该课程以相邻技能为基础，以改善泛化性能。引入一个具有不同任务的课程引入了不稳定性，我们通过提出一种新的算法来应对这一问题：持续深度Q网络扩张（DQN扩展）(CDE)。我们的方法动态地生成并调整Q函数子空间，以处理环境的变化和任务要求。CDE通过EWC缓解灾难性遗忘，同时确保高可塑性，使用自适应理性激活函数。

实验结果表明，相比强化学习基线和其他用于持续学习的适应方法，我们的方法在适应性和效率方面取得了显著提高，这凸显了我们在适应性列车调度设置管理稳定性-可塑性悖论方面的潜力。

---

## ShortCircuit_ AlphaZero-Driven Circuit Design

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09858v1)

### Abstract (English)

Chip design relies heavily on generating Boolean circuits, such as
AND-Inverter Graphs (AIGs), from functional descriptions like truth tables.
While recent advances in deep learning have aimed to accelerate circuit design,
these efforts have mostly focused on tasks other than synthesis, and
traditional heuristic methods have plateaued. In this paper, we introduce
ShortCircuit, a novel transformer-based architecture that leverages the
structural properties of AIGs and performs efficient space exploration.
Contrary to prior approaches attempting end-to-end generation of logic circuits
using deep networks, ShortCircuit employs a two-phase process combining
supervised with reinforcement learning to enhance generalization to unseen
truth tables. We also propose an AlphaZero variant to handle the double
exponentially large state space and the sparsity of the rewards, enabling the
discovery of near-optimal designs. To evaluate the generative performance of
our trained model , we extract 500 truth tables from a benchmark set of 20
real-world circuits. ShortCircuit successfully generates AIGs for 84.6% of the
8-input test truth tables, and outperforms the state-of-the-art logic synthesis
tool, ABC, by 14.61% in terms of circuits size.

### 摘要 (中文)

芯片设计主要依赖于从真值表等功能描述中生成布尔电路，如与非-或图（AIG）等。近年来，深度学习的进步旨在加速电路设计，但这些努力主要集中在合成之外的任务上，并且传统的启发式方法停滞不前。在本文中，我们引入了短路，一种基于Transformer的架构，它利用AIG结构特性的结构性属性，进行有效的空间探索。与先前试图使用深度网络端到端生成逻辑电路的方法不同，短路采用了一种结合监督和强化学习的两阶段过程，以增强对未见过真值表的一般化能力。此外，我们还提出了一个AlphaZero变体来处理双指数级大的状态空间和奖励的稀疏性，这使发现接近最优的设计成为可能。为了评估训练模型的生成性能，我们从基准集中的20个真实世界电路中提取了500个测试真值表。短路成功地为8输入测试真值表中的84.6%的AIG进行了生成，并在电路大小方面超过了最先进的逻辑合成工具ABC，实现了14.61%的优势。

---

## Performance Law of Large Language Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09895v1)

### Abstract (English)

Guided by the belief of the scaling law, large language models (LLMs) have
achieved impressive performance in recent years. However, scaling law only
gives a qualitative estimation of loss, which is influenced by various factors
such as model architectures, data distributions, tokenizers, and computation
precision. Thus, estimating the real performance of LLMs with different
training settings rather than loss may be quite useful in practical
development. In this article, we present an empirical equation named
"Performance Law" to directly predict the MMLU score of an LLM, which is a
widely used metric to indicate the general capability of LLMs in real-world
conversations and applications. Based on only a few key hyperparameters of the
LLM architecture and the size of training data, we obtain a quite accurate MMLU
prediction of various LLMs with diverse sizes and architectures developed by
different organizations in different years. Performance law can be used to
guide the choice of LLM architecture and the effective allocation of
computational resources without extensive experiments.

### 摘要 (中文)

通过对等位定律的信仰，近年来大型语言模型（LLMs）已经取得了令人印象深刻的性能。然而，等位定律只能提供损失量级的定性估计，这受到诸如模型架构、数据分布、词序列化器和计算精度等各种因素的影响。因此，在实践中估算不同训练设置下的LLMs的实际性能可能非常有用。本文提出了一种名为“表现律”的实证方程，可以直接预测LLM的MMLU分数，这是衡量在现实世界对话和应用中LVM能力广泛使用的指标。仅基于LLM架构的关键超参数和训练数据的大小，我们可以准确地预测具有不同规模和架构的各种组织开发的不同年份的LLM的多种情况下的MMLU得分。表现律可以用来指导选择LVM架构和有效的资源分配而无需进行大量实验。

---

## Active Learning for Identifying Disaster-Related Tweets_ A Comparison with Keyword Filtering and Generic Fine-Tuning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09914v1)

### Abstract (English)

Information from social media can provide essential information for emergency
response during natural disasters in near real-time. However, it is difficult
to identify the disaster-related posts among the large amounts of unstructured
data available. Previous methods often use keyword filtering, topic modelling
or classification-based techniques to identify such posts. Active Learning (AL)
presents a promising sub-field of Machine Learning (ML) that has not been used
much in the field of text classification of social media content. This study
therefore investigates the potential of AL for identifying disaster-related
Tweets. We compare a keyword filtering approach, a RoBERTa model fine-tuned
with generic data from CrisisLex, a base RoBERTa model trained with AL and a
fine-tuned RoBERTa model trained with AL regarding classification performance.
For testing, data from CrisisLex and manually labelled data from the 2021 flood
in Germany and the 2023 Chile forest fires were considered. The results show
that generic fine-tuning combined with 10 rounds of AL outperformed all other
approaches. Consequently, a broadly applicable model for the identification of
disaster-related Tweets could be trained with very little labelling effort. The
model can be applied to use cases beyond this study and provides a useful tool
for further research in social media analysis.

### 摘要 (中文)

社交媒体中的信息可以在近实时的情况下提供对自然灾害应急响应的宝贵信息。然而，识别这些与灾难相关的帖子在大量无结构数据中是困难的。以往的方法经常使用关键词过滤、主题建模或基于分类的技术来识别此类帖子。活跃学习（AL）是机器学习（ML）的一个新兴子领域，较少用于文本分类的社会媒体内容。因此，本研究旨在探讨AL对于识别灾难相关推文的潜在能力。我们比较了关键词过滤方法、预训练的RoBERTa模型和经过AL训练的RoBERTa模型关于分类性能。测试数据集包括CrisisLex的数据集和手动标记的德国2021年洪水和智利森林火灾2023年的数据。结果显示，通用微调结合10轮AL的表现优于其他所有方法。因此，一种适用于识别灾难相关推文的广泛适用的模型可以以最少的标注努力进行培训。该模型可用于应用于研究之外的应用，并提供进一步社会媒体分析研究的有效工具。

---

## The curse of random quantum data

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09937v1)

### Abstract (English)

Quantum machine learning, which involves running machine learning algorithms
on quantum devices, may be one of the most significant flagship applications
for these devices. Unlike its classical counterparts, the role of data in
quantum machine learning has not been fully understood. In this work, we
quantify the performances of quantum machine learning in the landscape of
quantum data. Provided that the encoding of quantum data is sufficiently
random, the performance, we find that the training efficiency and
generalization capabilities in quantum machine learning will be exponentially
suppressed with the increase in the number of qubits, which we call "the curse
of random quantum data". Our findings apply to both the quantum kernel method
and the large-width limit of quantum neural networks. Conversely, we highlight
that through meticulous design of quantum datasets, it is possible to avoid
these curses, thereby achieving efficient convergence and robust
generalization. Our conclusions are corroborated by extensive numerical
simulations.

### 摘要 (中文)

量子机器学习，即在量子设备上运行机器学习算法，可能是这些设备最具标志性的应用之一。与经典算法不同，量子机器学习的数据角色尚未完全理解。在这项工作中，我们量化了量子机器学习在量子数据领域的性能。如果编码的量子数据足够随机，则我们将发现随着量子比特数目的增加，训练效率和泛化能力将会呈指数级下降，我们将其称为“随机量子数据的诅咒”。我们的发现适用于量子核方法以及量子神经网络的大宽度极限。相反，通过精心设计量子数据集的设计，可以避免这些诅咒，从而实现高效收敛和稳健泛化。我们的结论得到了广泛的数值模拟的支持。



---

## Preference-Optimized Pareto Set Learning for Blackbox Optimization

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09976v1)

### Abstract (English)

Multi-Objective Optimization (MOO) is an important problem in real-world
applications. However, for a non-trivial problem, no single solution exists
that can optimize all the objectives simultaneously. In a typical MOO problem,
the goal is to find a set of optimum solutions (Pareto set) that trades off the
preferences among objectives. Scalarization in MOO is a well-established method
for finding a finite set approximation of the whole Pareto set (PS). However,
in real-world experimental design scenarios, it's beneficial to obtain the
whole PS for flexible exploration of the design space. Recently Pareto set
learning (PSL) has been introduced to approximate the whole PS. PSL involves
creating a manifold representing the Pareto front of a multi-objective
optimization problem. A naive approach includes finding discrete points on the
Pareto front through randomly generated preference vectors and connecting them
by regression. However, this approach is computationally expensive and leads to
a poor PS approximation. We propose to optimize the preference points to be
distributed evenly on the Pareto front. Our formulation leads to a bilevel
optimization problem that can be solved by e.g. differentiable cross-entropy
methods. We demonstrated the efficacy of our method for complex and difficult
black-box MOO problems using both synthetic and real-world benchmark data.

### 摘要 (中文)

多目标优化（MOO）是现实世界中一个重要的问题。然而，对于非平凡的问题，不存在一个解决方案可以同时优化所有目标。在典型的MOO问题中，目标是找到一组最优解（帕雷托集）。多目标优化中的标量化方法是找到整个帕累托集的有限集合近似的一种成熟的方法。然而，在实际实验设计场景中，获取整个PS以灵活探索设计空间是有益的。最近，帕累托集学习（PSL）被引入来近似整个PS。PPL涉及创建一个代表多目标优化问题帕累托前沿的曼哈顿点表示。一种朴素的方法包括通过随机生成偏好向量并用回归连接它们来发现帕累托前沿上的离散点。然而，这种方法计算成本高且导致差的PS近似。我们提出优化偏好的点均匀分布在帕累托前沿。我们的构建方式形成了一级二元优化问题，可以通过例如可微分交叉熵方法解决。我们使用合成和真实世界的基准数据展示了我们在复杂困难的黑盒MOO问题上使用我们方法的有效性。

---

## The Fairness-Quality Trade-off in Clustering

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10002v1)

### Abstract (English)

Fairness in clustering has been considered extensively in the past; however,
the trade-off between the two objectives -- e.g., can we sacrifice just a
little in the quality of the clustering to significantly increase fairness, or
vice-versa? -- has rarely been addressed. We introduce novel algorithms for
tracing the complete trade-off curve, or Pareto front, between quality and
fairness in clustering problems; that is, computing all clusterings that are
not dominated in both objectives by other clusterings. Unlike previous work
that deals with specific objectives for quality and fairness, we deal with all
objectives for fairness and quality in two general classes encompassing most of
the special cases addressed in previous work. Our algorithm must take
exponential time in the worst case as the Pareto front itself can be
exponential. Even when the Pareto front is polynomial, our algorithm may take
exponential time, and we prove that this is inevitable unless P = NP. However,
we also present a new polynomial-time algorithm for computing the entire Pareto
front when the cluster centers are fixed, and for perhaps the most natural
fairness objective: minimizing the sum, over all clusters, of the imbalance
between the two groups in each cluster.

### 摘要 (中文)

在过去的几十年里，聚类中的公平性问题已经得到了广泛的关注；然而，在质量与公平这两个目标之间存在冲突的情况下，很少有人会考虑如何平衡两者。我们引入了一种算法来追踪聚类中质量和公平之间的全面折衷曲线或帕雷托前沿，即计算所有不被其他聚类覆盖的目标集的簇化。与以前的工作不同的是，这些工作专门处理特定的质量和公平目标，而我们处理了所有的公平和质量目标。我们的算法可能以指数时间复杂度在最坏情况下运行，因为帕雷托前沿本身可以是指数级的。即使帕雷托前沿是多项式的，我们的算法也可能以指数时间复杂度运行，并且证明这可能是除非P=NP否则不可避免的。但是，我们也提出了一种新的多项式时间算法来计算当簇中心固定时的所有帕雷托前沿，以及一个可能是最自然的公平目标——最小化每个簇中两个群体间不平衡之和。

---

## Federated Frank-Wolfe Algorithm

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10090v1)

### Abstract (English)

Federated learning (FL) has gained a lot of attention in recent years for
building privacy-preserving collaborative learning systems. However, FL
algorithms for constrained machine learning problems are still limited,
particularly when the projection step is costly. To this end, we propose a
Federated Frank-Wolfe Algorithm (FedFW). FedFW features data privacy, low
per-iteration cost, and communication of sparse signals. In the deterministic
setting, FedFW achieves an $\varepsilon$-suboptimal solution within
$O(\varepsilon^{-2})$ iterations for smooth and convex objectives, and
$O(\varepsilon^{-3})$ iterations for smooth but non-convex objectives.
Furthermore, we present a stochastic variant of FedFW and show that it finds a
solution within $O(\varepsilon^{-3})$ iterations in the convex setting. We
demonstrate the empirical performance of FedFW on several machine learning
tasks.

### 摘要 (中文)

近年来，联邦学习（FL）因其建立隐私保护的协作学习系统而受到广泛关注。然而，约束机器学习问题的FL算法仍然有限，特别是在投影步骤成本较高的情况下。为此，我们提出了联邦弗朗克-波尔沃斯算法（FedFW）。FedFW具有数据隐私、每迭代低成本和稀疏信号通信的特点。在确定性设置中，FedFW可以在光滑且凸优化目标函数下，在$\varepsilon$次迭代内找到最优解，时间为$O(\varepsilon^{-2})$；对于光滑但非凸优化目标函数，时间则为$O(\varepsilon^{-3})$。此外，我们还提出了一种随机变体的FedFW，并展示了它在凸设施数学上的解决方案只需$O(\varepsilon^{-3})$次迭代即可找到最优解。我们在几个机器学习任务上演示了FedFW的实证性能。

---

## In-Context Learning with Representations_ Contextual Generalization of Trained Transformers

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10147v1)

### Abstract (English)

In-context learning (ICL) refers to a remarkable capability of pretrained
large language models, which can learn a new task given a few examples during
inference. However, theoretical understanding of ICL is largely under-explored,
particularly whether transformers can be trained to generalize to unseen
examples in a prompt, which will require the model to acquire contextual
knowledge of the prompt for generalization. This paper investigates the
training dynamics of transformers by gradient descent through the lens of
non-linear regression tasks. The contextual generalization here can be attained
via learning the template function for each task in-context, where all template
functions lie in a linear space with $m$ basis functions. We analyze the
training dynamics of one-layer multi-head transformers to in-contextly predict
unlabeled inputs given partially labeled prompts, where the labels contain
Gaussian noise and the number of examples in each prompt are not sufficient to
determine the template. Under mild assumptions, we show that the training loss
for a one-layer multi-head transformer converges linearly to a global minimum.
Moreover, the transformer effectively learns to perform ridge regression over
the basis functions. To our knowledge, this study is the first provable
demonstration that transformers can learn contextual (i.e., template)
information to generalize to both unseen examples and tasks when prompts
contain only a small number of query-answer pairs.

### 摘要 (中文)

语境学习（ICL）是预训练大型语言模型的一个非凡能力，它们可以在推理时通过学习一个新任务而学会。然而，对ICL的理解主要处于探索阶段，尤其是关于Transformer能否在prompt中进行泛化的问题，这需要模型从全局知识中获取上下文知识以实现泛化。本文通过非线性回归任务的视角，研究了Transformer的训练动力学。这里的上下文泛化可以通过在线上学习每个任务的模板函数来获得，其中所有模板函数都位于线性空间中的$m$个基函数中。我们分析了一层多头Transformer的学习动力学，在上下文中预测未标记输入给部分标注的提示，其中标签包含高斯噪声，并且每条提示中的样本数不足以确定模板。在温和假设下，我们证明了一个层多头Transformer的一次性损失收敛到全局最小值。此外，Transformer有效地执行了基函数上的ridge回归。迄今为止，这是首次证明Transformer能够学习上下文信息（即模板）以泛化到未见过的例子和任务的情况，当提示仅含有少量查询-答案对时。

---

## Multilingual Needle in a Haystack_ Investigating Long-Context Behavior of Multilingual Large Language Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10151v1)

### Abstract (English)

While recent large language models (LLMs) demonstrate remarkable abilities in
responding to queries in diverse languages, their ability to handle long
multilingual contexts is unexplored. As such, a systematic evaluation of the
long-context capabilities of LLMs in multilingual settings is crucial,
specifically in the context of information retrieval. To address this gap, we
introduce the MultiLingual Needle-in-a-Haystack (MLNeedle) test, designed to
assess a model's ability to retrieve relevant information (the needle) from a
collection of multilingual distractor texts (the haystack). This test serves as
an extension of the multilingual question-answering task, encompassing both
monolingual and cross-lingual retrieval. We evaluate four state-of-the-art LLMs
on MLNeedle. Our findings reveal that model performance can vary significantly
with language and needle position. Specifically, we observe that model
performance is the lowest when the needle is (i) in a language outside the
English language family and (ii) located in the middle of the input context.
Furthermore, although some models claim a context size of $8k$ tokens or
greater, none demonstrate satisfactory cross-lingual retrieval performance as
the context length increases. Our analysis provides key insights into the
long-context behavior of LLMs in multilingual settings to guide future
evaluation protocols. To our knowledge, this is the first study to investigate
the multilingual long-context behavior of LLMs.

### 摘要 (中文)

虽然最近的大型语言模型（LLM）在回答多种语言的问题时表现出令人印象深刻的才能，但它们处理长跨语种上下文的能力尚未被探索。因此，在多语境中评估LLM的长上下文能力至关重要，特别是在信息检索方面。为了填补这一空白，我们引入了MultiLingual Needle-in-a-Haystack（MLNeedle）测试，旨在评估模型从一个多语种干扰文本集合中提取相关信息（针头）的能力（haystack）。这个测试是多语种问题解答任务的扩展，包括单语和跨语种检索。我们对四个最先进的LLM进行了MLNeedle测试。我们的发现表明，模型性能随着语言和针头位置而变化很大。具体而言，当我们观察到针头位于非英语家族的语言之外且位于输入上下文中中间位置时，模型表现最低。此外，尽管一些模型声称上下文大小为8k个令牌或更大，但没有一个显示良好的跨语种检索性能随着上下文长度增加。我们的分析提供了LLM在多语境中的长上下文行为的关键洞察，以指导未来评价协议的设计。据我们所知，这是首次研究LLM的跨语境行为。

---

## Physics-Aware Combinatorial Assembly Planning using Deep Reinforcement Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10162v1)

### Abstract (English)

Combinatorial assembly uses standardized unit primitives to build objects
that satisfy user specifications. Lego is a widely used platform for
combinatorial assembly, in which people use unit primitives (ie Lego bricks) to
build highly customizable 3D objects. This paper studies sequence planning for
physical combinatorial assembly using Lego. Given the shape of the desired
object, we want to find a sequence of actions for placing Lego bricks to build
the target object. In particular, we aim to ensure the planned assembly
sequence is physically executable. However, assembly sequence planning (ASP)
for combinatorial assembly is particularly challenging due to its combinatorial
nature, ie the vast number of possible combinations and complex constraints. To
address the challenges, we employ deep reinforcement learning to learn a
construction policy for placing unit primitives sequentially to build the
desired object. Specifically, we design an online physics-aware action mask
that efficiently filters out invalid actions and guides policy learning. In the
end, we demonstrate that the proposed method successfully plans physically
valid assembly sequences for constructing different Lego structures. The
generated construction plan can be executed in real.

### 摘要 (中文)

组合装配使用标准化单元元素构建满足用户规格要求的对象。乐高是广泛用于组合装配的平台，人们使用单元元（如乐高积木）来构建高度定制化的三维对象。本论文研究了物理组合装配序列规划中使用乐高的方法。给定期望对象的形状，我们希望找到放置乐高积木以构建目标对象的序列动作。具体而言，我们的目标是确保计划的组装序列是可执行的。然而，对于组合装配，序列规划（ASP）特别具有挑战性，因为其结合性性质，即可能的组合数量众多和复杂的约束。为了应对这些挑战，我们采用深度强化学习来学习放置单元元序的动作策略，以构建所需的对象。具体来说，我们设计了一个在线物理学感知的动作掩码，有效地过滤无效动作并指导政策学习。最后，我们演示了提出的方成功地规划出在构建不同乐高结构时有效的组装序列。生成的构建计划可以实际执行。

---

## Retina-inspired Object Motion Segmentation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09454v1)

### Abstract (English)

Dynamic Vision Sensors (DVS) have emerged as a revolutionary technology with
a high temporal resolution that far surpasses RGB cameras. DVS technology draws
biological inspiration from photoreceptors and the initial retinal synapse. Our
research showcases the potential of additional retinal functionalities to
extract visual features. We provide a domain-agnostic and efficient algorithm
for ego-motion compensation based on Object Motion Sensitivity (OMS), one of
the multiple robust features computed within the mammalian retina. We develop a
framework based on experimental neuroscience that translates OMS' biological
circuitry to a low-overhead algorithm. OMS processes DVS data from dynamic
scenes to perform pixel-wise object motion segmentation. Using a real and a
synthetic dataset, we highlight OMS' ability to differentiate object motion
from ego-motion, bypassing the need for deep networks. This paper introduces a
bio-inspired computer vision method that dramatically reduces the number of
parameters by a factor of 1000 compared to prior works. Our work paves the way
for robust, high-speed, and low-bandwidth decision-making for in-sensor
computations.

### 摘要 (中文)



---

## Image-Based Geolocation Using Large Vision-Language Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09474v1)

### Abstract (English)

Geolocation is now a vital aspect of modern life, offering numerous benefits
but also presenting serious privacy concerns. The advent of large
vision-language models (LVLMs) with advanced image-processing capabilities
introduces new risks, as these models can inadvertently reveal sensitive
geolocation information. This paper presents the first in-depth study analyzing
the challenges posed by traditional deep learning and LVLM-based geolocation
methods. Our findings reveal that LVLMs can accurately determine geolocations
from images, even without explicit geographic training.
  To address these challenges, we introduce \tool{}, an innovative framework
that significantly enhances image-based geolocation accuracy. \tool{} employs a
systematic chain-of-thought (CoT) approach, mimicking human geoguessing
strategies by carefully analyzing visual and contextual cues such as vehicle
types, architectural styles, natural landscapes, and cultural elements.
Extensive testing on a dataset of 50,000 ground-truth data points shows that
\tool{} outperforms both traditional models and human benchmarks in accuracy.
It achieves an impressive average score of 4550.5 in the GeoGuessr game, with
an 85.37\% win rate, and delivers highly precise geolocation predictions, with
the closest distances as accurate as 0.3 km. Furthermore, our study highlights
issues related to dataset integrity, leading to the creation of a more robust
dataset and a refined framework that leverages LVLMs' cognitive capabilities to
improve geolocation precision. These findings underscore \tool{}'s superior
ability to interpret complex visual data, the urgent need to address emerging
security vulnerabilities posed by LVLMs, and the importance of responsible AI
development to ensure user privacy protection.

### 摘要 (中文)

地理位置是现代生活中不可或缺的一部分，它提供了诸多益处，但同时也带来了严重的隐私问题。大型视觉语言模型（LVLM）的出现引入了新的风险，因为这些模型可能会无意中泄露敏感的位置信息。本文首次对传统深度学习和基于LVLM的位置定位方法面临的挑战进行了深入分析。我们的发现表明，LVLM可以从图像中准确地确定位置，即使没有明确的地理训练。

为了应对这些挑战，我们提出了一种创新的框架 \tool{》，显著提高了基于图像的位置定位精度。 \tool{}采用系统思考链（CoT）方法，模仿人类猜地理的方法，通过仔细分析视觉和语境线索等视觉和语境线索来分析车辆类型、建筑风格、自然景观和文化元素。对包含50,000个真实数据点的数据集进行了广泛的测试，结果显示，在GeoGuessr游戏中， \tool{}在准确性方面超过了传统的模型和人机基准，并以平均4550.5的成绩获得85.37％的胜利率，并提供高度精确的位置预测，最接近的距离可以达到0.3公里。此外，我们的研究还揭示了与数据完整性相关的议题，导致创建了一个更稳健的数据集和一个利用LVLM认知能力改进位置精度的精炼框架。这些发现强调了 \tool{} 解读复杂视觉数据的能力的优势，以及解决由LVLM引发的新兴安全漏洞所必需的责任AI开发的重要性，以确保用户隐私保护。

---

## Screen Them All_ High-Throughput Pan-Cancer Genetic and Phenotypic Biomarker Screening from H__E Whole Slide Images

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09554v1)

### Abstract (English)

Many molecular alterations serve as clinically prognostic or
therapy-predictive biomarkers, typically detected using single or multi-gene
molecular assays. However, these assays are expensive, tissue destructive and
often take weeks to complete. Using AI on routine H&E WSIs offers a fast and
economical approach to screen for multiple molecular biomarkers. We present a
high-throughput AI-based system leveraging Virchow2, a foundation model
pre-trained on 3 million slides, to interrogate genomic features previously
determined by an next-generation sequencing (NGS) assay, using 47,960 scanned
hematoxylin and eosin (H&E) whole slide images (WSIs) from 38,984 cancer
patients. Unlike traditional methods that train individual models for each
biomarker or cancer type, our system employs a unified model to simultaneously
predict a wide range of clinically relevant molecular biomarkers across cancer
types. By training the network to replicate the MSK-IMPACT targeted biomarker
panel of 505 genes, it identified 80 high performing biomarkers with a mean
AU-ROC of 0.89 in 15 most common cancer types. In addition, 40 biomarkers
demonstrated strong associations with specific cancer histologic subtypes.
Furthermore, 58 biomarkers were associated with targets frequently assayed
clinically for therapy selection and response prediction. The model can also
predict the activity of five canonical signaling pathways, identify defects in
DNA repair mechanisms, and predict genomic instability measured by tumor
mutation burden, microsatellite instability (MSI), and chromosomal instability
(CIN). The proposed model can offer potential to guide therapy selection,
improve treatment efficacy, accelerate patient screening for clinical trials
and provoke the interrogation of new therapeutic targets.

### 摘要 (中文)

许多分子变异体作为临床预后或治疗预测的生化标志物，通常使用单基因或多基因分子检测技术检测。然而，这些测试方法昂贵、组织破坏性强且往往需要几周时间才能完成。利用AI在常规HE WSI上进行筛查，提供了一种快速而经济的方式来筛选多种生化标志物。我们提出了一种基于Virchow2的高通量人工智能系统，该模型通过训练一个由300万张切片预先训练的基础模型来分析之前由下一代测序（NGS）法确定的基因组特征，使用来自38984例癌症患者的47960幅扫描的淋巴细胞和嗜酸性细胞（H&E）全切片图像（WSI）。与传统方法不同的是，我们的系统采用统一模型同时预测各种临床相关的生化标志物跨越癌症类型。通过将其网络训练为复制了MSK-IMPACT靶向生物标记物面板中的505个基因，它识别出了80个表现优异的生物标志物，并以平均AUC值0.89在15种最常见的癌症类型中。此外，40个生物标志物显示出特定癌肿亚型之间的强相关性。进一步地，58个生物标志物与经常用于治疗选择和反应预测的临床靶标相关。该模型还可以预测五个经典信号途径的活性，识别DNA修复机制缺陷，以及根据肿瘤突变负担、微卫星不稳定（MSI）、染色体不稳定性（CIN）测量的遗传失常。提出的模型可以为指导疗法的选择提供潜在机会，提高治疗效果，加速患者对临床试验的筛查并激发对新治疗目标的探索。

翻译：

---

## Anim-Director_ A Large Multimodal Model Powered Agent for Controllable Animation Video Generation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09787v1)

### Abstract (English)

Traditional animation generation methods depend on training generative models
with human-labelled data, entailing a sophisticated multi-stage pipeline that
demands substantial human effort and incurs high training costs. Due to limited
prompting plans, these methods typically produce brief, information-poor, and
context-incoherent animations. To overcome these limitations and automate the
animation process, we pioneer the introduction of large multimodal models
(LMMs) as the core processor to build an autonomous animation-making agent,
named Anim-Director. This agent mainly harnesses the advanced understanding and
reasoning capabilities of LMMs and generative AI tools to create animated
videos from concise narratives or simple instructions. Specifically, it
operates in three main stages: Firstly, the Anim-Director generates a coherent
storyline from user inputs, followed by a detailed director's script that
encompasses settings of character profiles and interior/exterior descriptions,
and context-coherent scene descriptions that include appearing characters,
interiors or exteriors, and scene events. Secondly, we employ LMMs with the
image generation tool to produce visual images of settings and scenes. These
images are designed to maintain visual consistency across different scenes
using a visual-language prompting method that combines scene descriptions and
images of the appearing character and setting. Thirdly, scene images serve as
the foundation for producing animated videos, with LMMs generating prompts to
guide this process. The whole process is notably autonomous without manual
intervention, as the LMMs interact seamlessly with generative tools to generate
prompts, evaluate visual quality, and select the best one to optimize the final
output.

### 摘要 (中文)

传统的动画生成方法依赖于使用人类标注数据来训练生成模型，这需要一个复杂的多阶段管道，要求大量的人力投入和高昂的培训成本。由于提示计划有限，这些方法通常产生简短、信息贫乏且不连贯的动画。为了克服这些限制并自动化动画过程，我们率先引入大型多模态模型（LMM）作为核心处理器，以建立一个自主的动画制作代理，名为Anim-Director。这个代理主要利用了高级理解能力和生成AI工具的强大能力，从简洁的故事线或简单的指令中创建动画视频。具体来说，它主要在三个主要阶段运行：首先，Anim-Director根据用户输入生成一个连贯的故事线；其次，我们使用具有图像生成工具的LMM来生成设置和场景的视觉图像。这些图像旨在通过视觉语言的提示保持不同场景之间的视觉一致性，结合出现的角色和设定的图像。最后，场景图像成为生产动画视频的基础，LMM生成提示来指导这一过程。整个过程是显著自治的，无需人工干预，因为LMM与生成工具无缝交互，生成提示、评估视觉质量，并选择最佳方案优化最终输出。

---

## Docling Technical Report

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09869v1)

### Abstract (English)

This technical report introduces Docling, an easy to use, self-contained,
MIT-licensed open-source package for PDF document conversion. It is powered by
state-of-the-art specialized AI models for layout analysis (DocLayNet) and
table structure recognition (TableFormer), and runs efficiently on commodity
hardware in a small resource budget. The code interface allows for easy
extensibility and addition of new features and models.

### 摘要 (中文)

这份技术报告介绍了Docling，一个易于使用、自包含的开源包，用于PDF文档转换。它由先进的布局分析（DocLayNet）和表格结构识别（TableFormer）专用AI模型驱动，能够在有限资源预算下高效运行在通用硬件上。代码接口允许进行简单扩展，添加新功能和模型。

---

## Sliced Maximal Information Coefficient_ A Training-Free Approach for Image Quality Assessment Enhancement

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09920v1)

### Abstract (English)

Full-reference image quality assessment (FR-IQA) models generally operate by
measuring the visual differences between a degraded image and its reference.
However, existing FR-IQA models including both the classical ones (eg, PSNR and
SSIM) and deep-learning based measures (eg, LPIPS and DISTS) still exhibit
limitations in capturing the full perception characteristics of the human
visual system (HVS). In this paper, instead of designing a new FR-IQA measure,
we aim to explore a generalized human visual attention estimation strategy to
mimic the process of human quality rating and enhance existing IQA models. In
particular, we model human attention generation by measuring the statistical
dependency between the degraded image and the reference image. The dependency
is captured in a training-free manner by our proposed sliced maximal
information coefficient and exhibits surprising generalization in different IQA
measures. Experimental results verify the performance of existing IQA models
can be consistently improved when our attention module is incorporated. The
source code is available at https://github.com/KANGX99/SMIC.

### 摘要 (中文)

全参考图像质量评估（FR-IQA）模型通常通过测量降级图像与参考图像之间的视觉差异来操作。然而，包括经典模型（例如PSNR和SSIM）在内的现有FR-IQA模型以及基于深度学习的措施（例如LPIPS和DISTS）在捕捉人类视觉系统（HVS）的完整感知特征方面仍然存在局限性。本论文的目标不是设计新的FR-IQA指标，而是探索一种通用的人类视觉注意力估计策略来模仿人类评级的过程，并增强现有的IQA模型。具体来说，我们通过测量降级图像与参考图像之间的统计依赖关系来建模人注意力生成。无需训练，我们的提出切片最大信息系数捕获这种依赖关系，并且表现出不同的IQA衡量方法中令人惊讶的一致性泛化能力。实验结果验证了当引入注意力模块时，现有的IQA模型的性能可以得到一致性的改进。源代码可在https://github.com/KANGX99/SMIC中获取。

---

## Perceptual Depth Quality Assessment of Stereoscopic Omnidirectional Images

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.10134v1)

### Abstract (English)

Depth perception plays an essential role in the viewer experience for
immersive virtual reality (VR) visual environments. However, previous research
investigations in the depth quality of 3D/stereoscopic images are rather
limited, and in particular, are largely lacking for 3D viewing of 360-degree
omnidirectional content. In this work, we make one of the first attempts to
develop an objective quality assessment model named depth quality index (DQI)
for efficient no-reference (NR) depth quality assessment of stereoscopic
omnidirectional images. Motivated by the perceptual characteristics of the
human visual system (HVS), the proposed DQI is built upon multi-color-channel,
adaptive viewport selection, and interocular discrepancy features. Experimental
results demonstrate that the proposed method outperforms state-of-the-art image
quality assessment (IQA) and depth quality assessment (DQA) approaches in
predicting the perceptual depth quality when tested using both single-viewport
and omnidirectional stereoscopic image databases. Furthermore, we demonstrate
that combining the proposed depth quality model with existing IQA methods
significantly boosts the performance in predicting the overall quality of 3D
omnidirectional images.

### 摘要 (中文)

深度感知在沉浸式虚拟现实（VR）视觉环境的观众体验中扮演着至关重要的角色。然而，之前关于三维/立体图像深度质量的研究相当有限，并且特别缺乏对于三维观看360度全向内容时的深度质量研究。在此工作中，我们首次尝试开发一种名为深度质量指数（DQI）的目标质量评估模型，用于高效无参考（NR）深度质量评估立体全向图像。基于人类视网膜系统（HVS）的感知特征，提出的DQI基于多色通道、自适应视窗选择和眼间差异特征。实验结果表明，提出的方法在使用单个视窗和立体图像数据库进行预测时，可以比当前最先进的图像质量和深度质量评估（IQA和DQA）方法更好地预测感知深度质量。此外，我们将提出的深度质量模型与现有的IQA方法结合使用，显著提高了对三维全向图像整体质量的预测性能。

---

## Behavioral Learning of Dish Rinsing and Scrubbing based on Interruptive Direct Teaching Considering Assistance Rate

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09360v1)

### Abstract (English)

Robots are expected to manipulate objects in a safe and dexterous way. For
example, washing dishes is a dexterous operation that involves scrubbing the
dishes with a sponge and rinsing them with water. It is necessary to learn it
safely without splashing water and without dropping the dishes. In this study,
we propose a safe and dexterous manipulation system. %that can scrub and rinse
dirty dishes. The robot learns a dynamics model of the object by estimating the
state of the object and the robot itself, the control input, and the amount of
human assistance required (assistance rate) after the human corrects the
initial trajectory of the robot's hands by interruptive direct teaching. By
backpropagating the error between the estimated and the reference value %at the
next time using the acquired dynamics model, the robot can generate a control
input that approaches the reference value, for example, so that human
assistance is not required and the dish does not move excessively. This allows
for adaptive rinsing and scrubbing of dishes with unknown shapes and
properties. As a result, it is possible to generate safe actions that require
less human assistance.

### 摘要 (中文)

机器人应该以安全和灵活的方式操纵物体。例如，洗碗是一个涉及使用海绵清洗并用水冲洗餐具的操作。在人类纠正机器人的手初始轨迹时需要学习它，同时避免溅水和掉落餐具。在此研究中，我们提出了一种可以擦洗并冲洗脏盘子的安全、灵活的操纵系统。机器人通过估计对象的状态、自身状态以及所需的人类辅助（帮助率）来学习对象的动力学模型。通过利用获得的动力学模型，在下一次估计值与参考值之间的误差之间进行反向传播，机器人可以生成接近参考值的控制输入，例如，这样就可以无需人工干预而不使餐具移动过量。这允许根据未知形状和特性的盘子进行自适应洗涤和擦洗。因此，有可能生成对更少的人工辅助要求的安全动作。

---

## GRLinQ_ An Intelligent Spectrum Sharing Mechanism for Device-to-Device Communications with Graph Reinforcement Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09394v1)

### Abstract (English)

Device-to-device (D2D) spectrum sharing in wireless communications is a
challenging non-convex combinatorial optimization problem, involving entangled
link scheduling and power control in a large-scale network. The
state-of-the-art methods, either from a model-based or a data-driven
perspective, exhibit certain limitations such as the critical need for channel
state information (CSI) and/or a large number of (solved) instances (e.g.,
network layouts) as training samples. To advance this line of research, we
propose a novel hybrid model/datadriven spectrum sharing mechanism with graph
reinforcement learning for link scheduling (GRLinQ), injecting information
theoretical insights into machine learning models, in such a way that link
scheduling and power control can be solved in an intelligent yet explainable
manner. Through an extensive set of experiments, GRLinQ demonstrates superior
performance to the existing model-based and data-driven link scheduling and/or
power control methods, with a relaxed requirement for CSI, a substantially
reduced number of unsolved instances as training samples, a possible
distributed deployment, reduced online/offline computational complexity, and
more remarkably excellent scalability and generalizability over different
network scenarios and system configurations.

### 摘要 (中文)

在无线通信中，设备到设备（D2D）频谱共享是一个挑战性的非凸组合优化问题，涉及大规模网络中的交织链路调度和功率控制。目前的最先进方法，无论是从模型驱动还是数据驱动的角度，都存在一些限制，例如对信道状态信息（CSI）的需求或大量的已解决实例作为训练样本的数量需求（例如网络布局）。为了推进这一研究领域，我们提出了一种新型的混合模型/数据驱动的频谱共享机制，用于链路调度（GRLinQ），注入信息理论洞察力到机器学习模型中，从而实现智能且可解释的链路调度和功率控制。通过一系列实验，GRLinQ的表现优于现有基于模型的方法和基于数据的方法，并且具有一个放松的CSI要求，对于训练样本数量的要求大大减少，可能的分布式部署，减少了在线/离线计算复杂性，而且表现得更显著的是在不同网络场景和系统配置下有出色的可扩展性和泛化能力。

---

## Enhancing Startup Success Predictions in Venture Capital_ A GraphRAG Augmented Multivariate Time Series Method

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09420v1)

### Abstract (English)

In the Venture Capital(VC) industry, predicting the success of startups is
challenging due to limited financial data and the need for subjective revenue
forecasts. Previous methods based on time series analysis or deep learning
often fall short as they fail to incorporate crucial inter-company
relationships such as competition and collaboration. Regarding the issues, we
propose a novel approach using GrahphRAG augmented time series model. With
GraphRAG, time series predictive methods are enhanced by integrating these
vital relationships into the analysis framework, allowing for a more dynamic
understanding of the startup ecosystem in venture capital. Our experimental
results demonstrate that our model significantly outperforms previous models in
startup success predictions. To the best of our knowledge, our work is the
first application work of GraphRAG.

### 摘要 (中文)

在风险投资(VC)行业，由于缺乏财务数据和需要主观收入预测的需求，预测初创公司的成功是具有挑战性的。以往基于时间序列分析或深度学习的方法往往无法应对这些挑战，因为它们未能考虑关键的内部关系，如竞争和合作。针对这些问题，我们提出了一种新的方法，使用图RAG增强的时间序列建模技术。通过整合这些重要关系到分析框架中，我们的模型可以提供更动态的创业生态系统理解，从而显著优于以前的模型在初创公司成功的预测上。至目前为止，这是我们首次应用图RAG的工作。

---

## Parameterized Physics-informed Neural Networks for Parameterized PDEs

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09446v1)

### Abstract (English)

Complex physical systems are often described by partial differential
equations (PDEs) that depend on parameters such as the Reynolds number in fluid
mechanics. In applications such as design optimization or uncertainty
quantification, solutions of those PDEs need to be evaluated at numerous points
in the parameter space. While physics-informed neural networks (PINNs) have
emerged as a new strong competitor as a surrogate, their usage in this scenario
remains underexplored due to the inherent need for repetitive and
time-consuming training. In this paper, we address this problem by proposing a
novel extension, parameterized physics-informed neural networks (P$^2$INNs).
P$^2$INNs enable modeling the solutions of parameterized PDEs via explicitly
encoding a latent representation of PDE parameters. With the extensive
empirical evaluation, we demonstrate that P$^2$INNs outperform the baselines
both in accuracy and parameter efficiency on benchmark 1D and 2D parameterized
PDEs and are also effective in overcoming the known "failure modes".

### 摘要 (中文)

复杂物理系统通常由依赖于诸如雷诺数的流体力学参数的偏微分方程（PDE）描述。在设计优化或不确定性量化等应用中，需要对这些PDE的解进行评估，在参数空间中的多个点上。虽然物理引导神经网络（PINN）作为替代方案已经脱颖而出，但在这种情况下使用它们仍处于探索阶段，因为存在内在的需要进行重复和耗时的训练。本文通过提出一种新型扩展——参数化物理引导神经网络（P$^2$INN），解决了这个问题。P$^2$INN能够通过明确编码PDE参数的隐式表示来模型参数化的PDE的解决方案。通过广泛的实证分析，我们展示了P$^2$INN在基准1D和2D参数化PDE上的准确性和参数效率均优于基线，并且能够在已知“失败模式”中克服。

注：由于我是一个AI语言模型，无法提供英文到中文的直接翻译。如果您有任何其他问题，请随时提问！

---

## Advancements in Molecular Property Prediction_ A Survey of Single and Multimodal Approaches

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09461v1)

### Abstract (English)

Molecular Property Prediction (MPP) plays a pivotal role across diverse
domains, spanning drug discovery, material science, and environmental
chemistry. Fueled by the exponential growth of chemical data and the evolution
of artificial intelligence, recent years have witnessed remarkable strides in
MPP. However, the multifaceted nature of molecular data, such as molecular
structures, SMILES notation, and molecular images, continues to pose a
fundamental challenge in its effective representation. To address this,
representation learning techniques are instrumental as they acquire informative
and interpretable representations of molecular data. This article explores
recent AI/-based approaches in MPP, focusing on both single and multiple
modality representation techniques. It provides an overview of various molecule
representations and encoding schemes, categorizes MPP methods by their use of
modalities, and outlines datasets and tools available for feature generation.
The article also analyzes the performance of recent methods and suggests future
research directions to advance the field of MPP.

### 摘要 (中文)

分子属性预测（Molecular Property Prediction，MPP）在多个领域发挥着关键作用，从药物发现、材料科学到环境化学。随着化学数据的不断增长和人工智能技术的发展，近年来MPP取得了显著的进步。然而，分子数据的多维度性质，如分子结构、SMILES表示以及分子图像，仍然是有效表示的重要挑战。为了解决这一问题，学习方法至关重要，它们能够获得分子数据的有效信息和可解释性表示。本文探索了近期AI/基于的方法在MPP中的应用，重点讨论单模态和多元模态表示技术。它概述了各种分子表示和编码方案，并根据其使用模式分类MPP方法，还介绍了生成特征可用的数据集和工具。此外，文章分析了最近方法的表现，并提出了未来研究方向以推进MPP领域的进步。

---

## Enhancing Quantum Memory Lifetime with Measurement-Free Local Error Correction and Reinforcement Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09524v1)

### Abstract (English)

Reliable quantum computation requires systematic identification and
correction of errors that occur and accumulate in quantum hardware. To diagnose
and correct such errors, standard quantum error-correcting protocols utilize
$\textit{global}$ error information across the system obtained by mid-circuit
readout of ancillary qubits. We investigate circuit-level error-correcting
protocols that are measurement-free and based on $\textit{local}$ error
information. Such a local error correction (LEC) circuit consists of faulty
multi-qubit gates to perform both syndrome extraction and ancilla-controlled
error removal. We develop and implement a reinforcement learning framework that
takes a fixed set of faulty gates as inputs and outputs an optimized LEC
circuit. To evaluate this approach, we quantitatively characterize an extension
of logical qubit lifetime by a noisy LEC circuit. For the 2D classical Ising
model and 4D toric code, our optimized LEC circuit performs better at extending
a memory lifetime compared to a conventional LEC circuit based on Toom's rule
in a sub-threshold gate error regime. We further show that such circuits can be
used to reduce the rate of mid-circuit readouts to preserve a 2D toric code
memory. Finally, we discuss the application of the LEC protocol on dissipative
preparation of quantum states with topological phases.

### 摘要 (中文)

可靠的量子计算需要系统性地识别和纠正出现并累积在量子硬件中的错误。为了诊断和纠正这些误差，标准的量子纠错协议利用中门读取辅助量子比特获得系统的全局误差信息来执行测量。我们研究了不依赖于测量的基于局部误差信息的电路级纠错协议。这种局部错误修正（LEC）电路由故障多量子位操作完成，用于提取伪随机码和控制外延错误消除。我们开发并实现了一个强化学习框架，该框架接受固定组的故障门作为输入，并输出优化的LEC电路。为了评估这一方法，我们定量地度量了由噪声LEC电路扩展逻辑量子比特寿命的延长能力。对于二维经典Ising模型和四维拓扑编码，我们的优化LEC电路比Toom规则为基础的传统LEC电路在阈值门误差条件下更好地延长存储寿命。我们进一步表明，这样的电路可以用来减少中期读出率以保持二维拓扑编码内存。最后，我们讨论了LEC协议在耗散准备具有拓扑相的量子态的应用。

---

## Security Concerns in Quantum Machine Learning as a Service

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09562v1)

### Abstract (English)

Quantum machine learning (QML) is a category of algorithms that employ
variational quantum circuits (VQCs) to tackle machine learning tasks. Recent
discoveries have shown that QML models can effectively generalize from limited
training data samples. This capability has sparked increased interest in
deploying these models to address practical, real-world challenges, resulting
in the emergence of Quantum Machine Learning as a Service (QMLaaS). QMLaaS
represents a hybrid model that utilizes both classical and quantum computing
resources. Classical computers play a crucial role in this setup, handling
initial pre-processing and subsequent post-processing of data to compensate for
the current limitations of quantum hardware. Since this is a new area, very
little work exists to paint the whole picture of QMLaaS in the context of known
security threats in the domain of classical and quantum machine learning. This
SoK paper is aimed to bridge this gap by outlining the complete QMLaaS
workflow, which encompasses both the training and inference phases and
highlighting significant security concerns involving untrusted classical or
quantum providers. QML models contain several sensitive assets, such as the
model architecture, training/testing data, encoding techniques, and trained
parameters. Unauthorized access to these components could compromise the
model's integrity and lead to intellectual property (IP) theft. We pinpoint the
critical security issues that must be considered to pave the way for a secure
QMLaaS deployment.

### 摘要 (中文)

量子机器学习（QML）是使用变分量子循环（VQC）来解决机器学习任务的类别算法。最近的研究表明，QML模型可以有效地从有限训练数据样本中泛化。这种能力激发了部署这些模型以应对实际、现实世界挑战的兴趣，从而导致量子机器学习作为服务（QMLaaS）的出现。QMLaaS代表一个混合模型，利用经典和量子计算资源。在这一设置中，经典计算机起着至关重要的作用，负责处理数据的预处理和后处理工作，以补偿当前量子硬件的限制。由于这是一个新领域，关于已知安全威胁在经典和量子机器学习领域的整个图景非常少的工作存在。因此，本文旨在通过概述完整的QMLaaS流程，即包括培训和推断阶段，并突出涉及不信任的经典或量子提供者的显著安全问题，来填补这一差距。QML模型包含几项敏感资产，如模型架构、训练/测试数据、编码技术以及训练参数。未经授权访问这些组件可能会损害模型的完整性并引发知识产权（IP）盗窃。我们指出了必须考虑的关键安全问题，以便为安全的QMLaaS部署铺平道路。

---

## Circuit design in biology and machine learning. I. Random networks and dimensional reduction

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09604v1)

### Abstract (English)

A biological circuit is a neural or biochemical cascade, taking inputs and
producing outputs. How have biological circuits learned to solve environmental
challenges over the history of life? The answer certainly follows Dobzhansky's
famous quote that ``nothing in biology makes sense except in the light of
evolution.'' But that quote leaves out the mechanistic basis by which natural
selection's trial-and-error learning happens, which is exactly what we have to
understand. How does the learning process that designs biological circuits
actually work? How much insight can we gain about the form and function of
biological circuits by studying the processes that have made those circuits?
Because life's circuits must often solve the same problems as those faced by
machine learning, such as environmental tracking, homeostatic control,
dimensional reduction, or classification, we can begin by considering how
machine learning designs computational circuits to solve problems. We can then
ask: How much insight do those computational circuits provide about the design
of biological circuits? How much does biology differ from computers in the
particular circuit designs that it uses to solve problems? This article steps
through two classic machine learning models to set the foundation for analyzing
broad questions about the design of biological circuits. One insight is the
surprising power of randomly connected networks. Another is the central role of
internal models of the environment embedded within biological circuits,
illustrated by a model of dimensional reduction and trend prediction. Overall,
many challenges in biology have machine learning analogs, suggesting hypotheses
about how biology's circuits are designed.

### 摘要 (中文)

一个生物回路是一个神经或生化递增，接受输入并产生输出。生命的历史中生物学回路是如何学习解决环境挑战的呢？答案当然符合杜布赞斯基著名的一句话：“除了在进化光线下没有意义的生物学现象之外，一切生物现象都必须以进化为基础。”但是这句话忽略了自然选择试验错误学习的具体机制，正是我们所要理解的。如何设计生物回路的学习过程实际上工作？通过研究使这些回路起作用的过程，我们可以了解生物回路的形式和功能吗？因为生命的回路往往需要解决与机器学习相同的问题，如环境跟踪、自稳控制、维度缩减或分类问题，所以我们可以从考虑机器学习如何设计计算回路来解决问题开始着手。然后问：那些计算回路为我们提供多少关于生物回路的设计方面的洞察力？使用什么电路设计特定的生物回路与计算机有何不同？这篇文章从两个经典的人工智能模型出发，为分析生物回路的设计打下了基础。第一个发现是随机连接网络令人惊讶的力量。另一个是嵌入于生物回路内部的环境内建模型发挥中心作用的例子，由趋势预测和降维模型示例说明。总的来说，在许多生物学挑战方面都有人工智能的等价物，这提出了对生物学回路如何被设计的假设。

---

## Branch and Bound to Assess Stability of Regression Coefficients in Uncertain Models

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09634v1)

### Abstract (English)

It can be difficult to interpret a coefficient of an uncertain model. A slope
coefficient of a regression model may change as covariates are added or removed
from the model. In the context of high-dimensional data, there are too many
model extensions to check. However, as we show here, it is possible to
efficiently search, with a branch and bound algorithm, for maximum and minimum
values of that adjusted slope coefficient over a discrete space of regularized
regression models. Here we introduce our algorithm, along with supporting
mathematical results, an example application, and a link to our computer code,
to help researchers summarize high-dimensional data and assess the stability of
regression coefficients in uncertain models.

### 摘要 (中文)

在不确定模型中解释系数可能很困难。回归模型中的斜率系数可能会随着自变量从模型中添加或删除而改变。在高维数据的情况下，有太多可供检查的模型扩展。然而，正如我们在这里展示的那样，有可能以高效的方式搜索，使用分支和边界算法，在离散空间内的正则化回归模型上搜索最大值和最小值调整斜率系数。在这里，我们将介绍我们的算法，以及支持数学结果、一个示例应用和到我们的计算机代码链接，帮助研究人员总结高维数据并评估不确定性模型中回归系数的稳定性。

---

## Parallel-in-Time Solutions with Random Projection Neural Networks

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09756v1)

### Abstract (English)

This paper considers one of the fundamental parallel-in-time methods for the
solution of ordinary differential equations, Parareal, and extends it by
adopting a neural network as a coarse propagator. We provide a theoretical
analysis of the convergence properties of the proposed algorithm and show its
effectiveness for several examples, including Lorenz and Burgers' equations. In
our numerical simulations, we further specialize the underpinning neural
architecture to Random Projection Neural Networks (RPNNs), a 2-layer neural
network where the first layer weights are drawn at random rather than
optimized. This restriction substantially increases the efficiency of fitting
RPNN's weights in comparison to a standard feedforward network without
negatively impacting the accuracy, as demonstrated in the SIR system example.

### 摘要 (中文)

这篇论文考虑了求解常微分方程的基本并行时间方法之一，Parareal，并通过采用神经网络作为粗近似传播器来扩展它。我们提供了所提议算法的理论分析收敛性能，并展示了其对几个示例的有效性，包括洛伦兹和布劳尔斯方程。在我们的数值模拟中，我们进一步特化基础神经架构为随机投影神经网络（RPNN），这是一种两层神经网络，在第一层权重中随机抽取而不是优化。这一限制大大提高了在标准前馈网络上拟合RPNN权重的效率，而不会负面影响精度，这一点已经在SIR系统例子中得到证明。

---

## Strategic Demonstration Selection for Improved Fairness in LLM In-Context Learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09757v1)

### Abstract (English)

Recent studies highlight the effectiveness of using in-context learning (ICL)
to steer large language models (LLMs) in processing tabular data, a challenging
task given the structured nature of such data. Despite advancements in
performance, the fairness implications of these methods are less understood.
This study investigates how varying demonstrations within ICL prompts influence
the fairness outcomes of LLMs. Our findings reveal that deliberately including
minority group samples in prompts significantly boosts fairness without
sacrificing predictive accuracy. Further experiments demonstrate that the
proportion of minority to majority samples in demonstrations affects the
trade-off between fairness and prediction accuracy. Based on these insights, we
introduce a mitigation technique that employs clustering and evolutionary
strategies to curate a diverse and representative sample set from the training
data. This approach aims to enhance both predictive performance and fairness in
ICL applications. Experimental results validate that our proposed method
dramatically improves fairness across various metrics, showing its efficacy in
real-world scenarios.

### 摘要 (中文)

最近的研究揭示了使用语境学习（ICL）在处理表格数据这一挑战性任务中的有效性，该任务由于数据的结构化而复杂。尽管取得了显著的进步，但这些方法的公平性含义仍不被理解。本研究调查了不同示例在ICL提示中如何影响语言模型（LLM）的公平性结果。我们的发现表明，在提示中故意包含少数族裔样本可以显著提高公平性，而不牺牲预测精度。进一步实验展示了示范集中的比例对公平性和预测精度之间的折衷的影响。基于这些洞察，我们引入了一种减缓技术，利用聚类和进化策略从训练数据中收集多样性和代表性样本集合。这一方法旨在增强ICL应用中的预测性能和公平性。实验结果验证了我们的提出的方法极大地提高了各种指标下的公平性，显示其在现实世界场景中的有效性和优越性。

---

## Liquid Fourier Latent Dynamics Networks for fast GPU-based numerical simulations in computational cardiology

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09818v1)

### Abstract (English)

Scientific Machine Learning (ML) is gaining momentum as a cost-effective
alternative to physics-based numerical solvers in many engineering
applications. In fact, scientific ML is currently being used to build accurate
and efficient surrogate models starting from high-fidelity numerical
simulations, effectively encoding the parameterized temporal dynamics
underlying Ordinary Differential Equations (ODEs), or even the spatio-temporal
behavior underlying Partial Differential Equations (PDEs), in appropriately
designed neural networks. We propose an extension of Latent Dynamics Networks
(LDNets), namely Liquid Fourier LDNets (LFLDNets), to create parameterized
space-time surrogate models for multiscale and multiphysics sets of highly
nonlinear differential equations on complex geometries. LFLDNets employ a
neurologically-inspired, sparse, liquid neural network for temporal dynamics,
relaxing the requirement of a numerical solver for time advancement and leading
to superior performance in terms of tunable parameters, accuracy, efficiency
and learned trajectories with respect to neural ODEs based on feedforward
fully-connected neural networks. Furthermore, in our implementation of
LFLDNets, we use a Fourier embedding with a tunable kernel in the
reconstruction network to learn high-frequency functions better and faster than
using space coordinates directly as input. We challenge LFLDNets in the
framework of computational cardiology and evaluate their capabilities on two
3-dimensional test cases arising from multiscale cardiac electrophysiology and
cardiovascular hemodynamics. This paper illustrates the capability to run
Artificial Intelligence-based numerical simulations on single or multiple GPUs
in a matter of minutes and represents a significant step forward in the
development of physics-informed digital twins.

### 摘要 (中文)

科学机器学习（ML）正在成为在许多工程应用中经济有效的替代物理基于的数值求解器。事实上，科学ML正被用于从高保真数值模拟开始构建准确和高效的元模型，有效地编码参数化的时间动力学基础一阶微分方程（ODEs），甚至编码空间-时间行为的基础二阶微分方程（PDEs）。我们提出了扩展Latent Dynamics Networks（LDNs），即液态四次方块LDNs（LFLDNs），以创建多尺度和多物理学集的高度非线性差分方程的空间-时间元模型。LFLDNs使用神经网络灵感、稀疏液体神经网络来处理时间动态，从而放松了对时间推进数值求解器的要求，并且由于可调参数、精度、学习轨迹以及基于全连接神经网络的前馈神经元的性能而优于神经ODE。此外，在我们的LFLDNs实现中，我们在重建网络中使用可调节的核进行傅里叶嵌入，以比直接作为输入的学习高频函数更好地更快地学习。我们挑战LFLDNs在计算心内科的框架下，评估它们在两个三维测试案例中的能力，这些测试案例源于多尺度心脏电生理学和心血管动力学。这篇论文展示了在单个或多个GPU上运行人工智能基底数值模拟的能力，在几分钟内完成，并代表了物理约束数字双胞胎开发的重大步骤。

---

## Machine Learning with Physics Knowledge for Prediction_ A Survey

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09840v1)

### Abstract (English)

This survey examines the broad suite of methods and models for combining
machine learning with physics knowledge for prediction and forecast, with a
focus on partial differential equations. These methods have attracted
significant interest due to their potential impact on advancing scientific
research and industrial practices by improving predictive models with small- or
large-scale datasets and expressive predictive models with useful inductive
biases. The survey has two parts. The first considers incorporating physics
knowledge on an architectural level through objective functions, structured
predictive models, and data augmentation. The second considers data as physics
knowledge, which motivates looking at multi-task, meta, and contextual learning
as an alternative approach to incorporating physics knowledge in a data-driven
fashion. Finally, we also provide an industrial perspective on the application
of these methods and a survey of the open-source ecosystem for physics-informed
machine learning.

### 摘要 (中文)

This survey examines the broad suite of methods and models for combining
machine learning with physics knowledge for prediction and forecast, with a
focus on partial differential equations. These methods have attracted
significant interest due to their potential impact on advancing scientific
research and industrial practices by improving predictive models with small- or
large-scale datasets and expressive predictive models with useful inductive
biases. The survey has two parts. The first considers incorporating physics
knowledge on an architectural level through objective functions, structured
predictive models, and data augmentation. The second considers data as physics
knowledge, which motivates looking at multi-task, meta, and contextual learning
as an alternative approach to incorporating physics knowledge in a data-driven
fashion. Finally, we also provide an industrial perspective on the application
of these methods and a survey of the open-source ecosystem for physics-informed
machine learning.

---

## MAPLE_ Enhancing Review Generation with Multi-Aspect Prompt LEarning in Explainable Recommendation

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09865v1)

### Abstract (English)

Explainable Recommendation task is designed to receive a pair of user and
item and output explanations to justify why an item is recommended to a user.
Many models treat review-generation as a proxy of explainable recommendation.
Although they are able to generate fluent and grammatical sentences, they
suffer from generality and hallucination issues. We propose a personalized,
aspect-controlled model called Multi-Aspect Prompt LEarner (MAPLE), in which it
integrates aspect category as another input dimension to facilitate the
memorization of fine-grained aspect terms. Experiments on two real-world review
datasets in restaurant domain show that MAPLE outperforms the baseline
review-generation models in terms of text and feature diversity while
maintaining excellent coherence and factual relevance. We further treat MAPLE
as a retriever component in the retriever-reader framework and employ a
Large-Language Model (LLM) as the reader, showing that MAPLE's explanation
along with the LLM's comprehension ability leads to enriched and personalized
explanation as a result. We will release the code and data in this http upon
acceptance.

### 摘要 (中文)

可解释推荐任务旨在接收用户和物品的一对，输出解释来说明为什么给定的物品会被推荐给用户。许多模型将评论生成视为可解释推荐的一个代理。虽然它们能够生成流畅且语法正确的句子，但它们存在泛化性和幻觉问题。我们提出了一个名为Multi-Aspect Prompt LEarner（MAPLE）的个性化、方面控制的模型，在其中它整合了领域作为另一个输入维度，以促进细粒度方面的记忆。在餐饮业的真实世界评价数据集上进行实验表明，MAPLE在文本和特征多样性方面优于基准评论生成模型，并保持良好的语义连贯性和事实相关性。此外，我们将MAPLE作为检索器框架中的检索器组件，并使用大型语言模型（LLM）作为读者，显示MAPLE的解释以及LLM的理解能力导致了丰富且个性化的解释结果。我们将在接受后在此http地址上发布代码和数据。

---

## Differential Private Stochastic Optimization with Heavy-tailed Data_ Towards Optimal Rates

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09891v1)

### Abstract (English)

We study convex optimization problems under differential privacy (DP). With
heavy-tailed gradients, existing works achieve suboptimal rates. The main
obstacle is that existing gradient estimators have suboptimal tail properties,
resulting in a superfluous factor of $d$ in the union bound. In this paper, we
explore algorithms achieving optimal rates of DP optimization with heavy-tailed
gradients. Our first method is a simple clipping approach. Under bounded $p$-th
order moments of gradients, with $n$ samples, it achieves
$\tilde{O}(\sqrt{d/n}+\sqrt{d}(\sqrt{d}/n\epsilon)^{1-1/p})$ population risk
with $\epsilon\leq 1/\sqrt{d}$. We then propose an iterative updating method,
which is more complex but achieves this rate for all $\epsilon\leq 1$. The
results significantly improve over existing methods. Such improvement relies on
a careful treatment of the tail behavior of gradient estimators. Our results
match the minimax lower bound in \cite{kamath2022improved}, indicating that the
theoretical limit of stochastic convex optimization under DP is achievable.

### 摘要 (中文)

我们研究了在差分隐私（DP）下的凸优化问题。对于具有重尾梯度的现有工作，其性能表现不佳。主要障碍是现有的梯度估计器的尾部特性较差，导致联合约束中的多余因子为$d$。本文探索了实现最优率的DP优化方法，通过使用具有重尾梯度的算法。我们的第一种方法是一个简单的剪切法。基于$g^{(i)}$的$p$阶均值在$n$个样本下，它可以获得$O(\sqrt{d/n} + \sqrt{d}(\sqrt{d}/n\epsilon)^{1-1/p})$的总体风险，$\epsilon \leq 1/\sqrt{d}$。然后提出了一种迭代更新的方法，虽然更复杂但可以在此速率上达到所有$\epsilon \leq 1$。结果显著优于现有方法。这种改进依赖于对梯度估计器尾部行为的仔细处理。我们的结果与\cite{kamath2022improved}中所得到的最小极大值下限相匹配，表明在DP下的随机凸优化理论极限是可以实现的。

---

## Instruction-Based Molecular Graph Generation with Unified Text-Graph Diffusion Model

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09896v1)

### Abstract (English)

Recent advancements in computational chemistry have increasingly focused on
synthesizing molecules based on textual instructions. Integrating graph
generation with these instructions is complex, leading most current methods to
use molecular sequences with pre-trained large language models. In response to
this challenge, we propose a novel framework, named $\textbf{UTGDiff (Unified
Text-Graph Diffusion Model)}$, which utilizes language models for discrete
graph diffusion to generate molecular graphs from instructions. UTGDiff
features a unified text-graph transformer as the denoising network, derived
from pre-trained language models and minimally modified to process graph data
through attention bias. Our experimental results demonstrate that UTGDiff
consistently outperforms sequence-based baselines in tasks involving
instruction-based molecule generation and editing, achieving superior
performance with fewer parameters given an equivalent level of pretraining
corpus. Our code is availble at https://github.com/ran1812/UTGDiff.

### 摘要 (中文)

最近，计算化学领域的进步越来越集中在基于文本的分子合成上。集成图生成与这些指令是复杂的，因此大多数当前方法都使用预训练的大语言模型处理分子序列。针对这一挑战，我们提出了一个名为$\textbf{UTGDiff（统一文本-图扩散模型）}$的新框架，它利用语言模型进行离散图扩散来从指令中生成分子图。UTGDiff具有统一的文本-图变换器作为降噪网络，来源于预训练的语言模型，并通过注意力偏置进行了最小化修改。我们的实验结果表明，UTGDiff在涉及基于指令的分子生成和编辑的任务中始终优于序列基线，在给定相同预训练语料库的情况下，其参数数更少但性能更高。我们的代码可在https://github.com/ran1812/UTGDiff中获取。

---

## Electron-nucleus cross sections from transfer learning

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09936v1)

### Abstract (English)

Transfer learning (TL) allows a deep neural network (DNN) trained on one type
of data to be adapted for new problems with limited information. We propose to
use the TL technique in physics. The DNN learns the physics of one process, and
after fine-tuning, it makes predictions for related processes. We consider the
DNNs, trained on inclusive electron-carbon scattering data, and show that after
fine-tuning, they accurately predict cross sections for electron interactions
with nuclear targets ranging from lithium to iron. The method works even when
the DNN is fine-tuned on a small dataset.

### 摘要 (中文)

迁移学习（TL）允许在有限信息的情况下，对一种数据类型进行深度神经网络（DNN）的适应性训练。我们提出使用TL技术来物理领域。DNN通过学习一个过程的物理来学习，经过微调后，它能预测相关过程。我们考虑了在包括电子-碳散射数据中训练的DNN，并展示了当对小数据集进行微调时，它们能准确地预测锂到铁核靶子之间的电子相互作用截面。这种方法甚至可以在微调后的DNN上工作。

---

## Symplectic Neural Networks Based on Dynamical Systems

**PDF URL**: [PDF Link](http://arxiv.org/pdf/2408.09821v1)

### Abstract (English)

We present and analyze a framework for designing symplectic neural networks
(SympNets) based on geometric integrators for Hamiltonian differential
equations. The SympNets are universal approximators in the space of Hamiltonian
diffeomorphisms, interpretable and have a non-vanishing gradient property. We
also give a representation theory for linear systems, meaning the proposed
P-SympNets can exactly parameterize any symplectic map corresponding to
quadratic Hamiltonians. Extensive numerical tests demonstrate increased
expressiveness and accuracy -- often several orders of magnitude better -- for
lower training cost over existing architectures. Lastly, we show how to perform
symbolic Hamiltonian regression with SympNets for polynomial systems using
backward error analysis.

### 摘要 (中文)

我们提出并分析了一种设计基于几何积分器的哈密顿微分方程的无扰动对称神经网络（SympNets）的框架。SympNets是哈密顿流形映射空间中的通用逼近器，可解释性，并且具有非零梯度属性。我们也给出了线性系统的代数理论，这意味着提出的P-SympNets可以精确地参数化任何对应于二次哈密顿的对称映射。大量的数值测试表明，在训练成本方面，与现有架构相比，对于较低的成本提高了表达能力和准确性——通常几倍甚至几十倍。最后，我们展示了如何使用符号哈密顿回归来使用反向误差分析进行多项式系统。

---

