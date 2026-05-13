# 基于ICFIE-YOLO的低照度图像目标检测方法

秦嘉奇，江泽涛*，雷晓春

（桂林电子科技大学广西图像图形与智能处理重点实验室，广西桂林 ）

摘　要：低照度环境下获取的图像往往亮度低、对比度低、光照不均匀，从而造成图像特征变弱及模糊难于提取，同时在有限提取的特征中也存在大量噪声信息，导致目标难于检测识别，因而现有低照度目标检测成果极少.针对低照度目标特征难于提取及特征空间噪声大的问题，本文提出一种基于光照矫正与特征交互增强（Illumination Cor⁃rection and Feature Interacted Enhancement，ICFIE-YOLO）网络的低照度目标检测方法.该方法首先利用提出的ICFIE YOLO内部多尺度光照矫正网络（Multi Scale Illumination Correction Network，MSICN）对低照度图像进行光照矫正，突出隐藏在图像背景中目标的模糊特征，使特征提取模块能更好地提取到目标特征；其次，为充分利用有效特征信息，过滤特征图中的噪声干扰，提出特征交互增强（Feature Interacted Enhancement，FIE）检测头，通过特征注意力交互方式实现特征增强，建立低照度图像中各个区域特征之间的空间关联和语义关联，从而抑制噪声对有效特征的干扰，实现降噪效果；最后，在增强特征及去除噪声的基础上用改进的检测头实现高精度目标检测.在ExDark和DarkFace数据集上的实验表明，所提方法较主流目标检测方法mAP提高2.1个百分点以上，较现有低照度目标检测方法召回率提高4.2个百分点以上，同时召回率较基线模型提高了2.6个百分点，所提方法具有较好的泛化性.

关键词：　目标检测；低照度；光照矫正；特征去噪；特征增强  

基金项目：　国家自然科学基金（No.62172118）；广西自然学科基金（No.2021GXNSFDA196002）；广西图像图形智能处理重点实验项目（No.GIIP2302，No.GIIP2303，No.GIIP2304）；研究生创新基金（No.2024YCXB09，No.2024YCXS039）

中图分类号： TP391

文献标识码： A

文章编号： 0372-2112(2025)02-0514-13

电子学报 URL:http://www.ejournal.org.cn

DOI:10.12263/DZXB.20240648 

# Low Illumination Image Object Detection Method Based on ICFIE-YOLO

QIN Jia-qi, JIANG Ze-tao* , LEI Xiao-chun 

(Guangxi Key Laboratory of Image and Graphic Intelligent Processing, Guilin University of Electronic Technology, Guilin, Guangxi 541004, China) 

Abstract: Images obtained in low light environments often have low brightness, low contrast, and uneven lighting, re⁃ sulting in weakened and blurred image features that are difficult to extract. At the same time, there is also a large amount of noise information in the limited extracted features, making it difficult to detect and recognize objects. Therefore, there are very few existing low light object detection results. This paper proposes a low illumination object detection method based on the Illumination Correction and Feature Interaction Enhancement (ICFIE-YOLO) network to address the difficulties in extracting features from low illumination objects and the large noise in the feature space. This method first utilizes the pro⁃ posed ICFIE-YOLO internal Multi Scale Illumination Correction Network (MSICN) to correct low illumination images, highlighting the blurry features of objects hidden in the image’s background, and enabling the feature extraction module to better extract object features. Secondly, to fully utilize effective feature information and filter out noise interference in fea⁃ ture maps, a Feature Interacted Enhancement (FIE) detection head is proposed. Through feature attention interaction, feature enhancement is achieved, establishing spatial and semantic correlations between features in different regions of low illumi⁃ nation images, thereby suppressing the interference of noise on effective features and achieving feature enhancement. Final⁃ ly, on the basis of enhancing features and removing noise, an improved detection head is used to achieve high-precision ob⁃ ject detection. Experiments on the ExDark and DarkFace datasets show that the proposed Method improves mAP by over 

2.1 percentage points compared to mainstream object detection models, increases recall by over 4.2 percentage points com⁃ pared to existing low light object detection Methods, and improves recall by 2.6 percentage points compared to baseline models. The proposed Method has good generalization performance. 

Key words: object detection; low illumination; light correction; feature denoising; feature enhancement 

Foundation Item(s) National Natural Science Foundation of China (No. ); Natural Science Foundation of Guangxi(No.2021GXNSFDA196002); Guangxi Key Laboratory of Image and Graphic Intelligent Processing (No.GIIP2302, No.GIIP2303, No.GIIP2304); Innovation Project of Guangxi Graduate Education (No.2024YCXB09, No.2024YCXS039) 

# 1　引言

目标检测的任务是找出图像中所有感兴趣的目标物体，确定它们的类别和位置是计算机视觉领域的核心问题之一.由于各类物体有不同的外观、形状和姿态，加上成像时距离、光照、遮挡等因素的干扰，目标检测一直是计算机视觉领域具有挑战性的问题.在该领域，使用深度学习方法进行目标检测已经成为主流.具有代表性的是两阶段目标检测算法Faster-RCNN及其衍生模型［1~5］，以及单阶段的YOLO、SSD目标检测框架及其衍生框架［6~10］.随着Transformer结构的提出和发展，人们开始探索将此结构用于目标检测任务中［11］.

现有的目标检测方法取得了较好的效果并得到广泛应用，但现有的方法大都基于正常照度环境进行目标检测，而低照度环境下的目标检测研究成果却很少， 将现有的正常照度环境下的目标检测方法移植到低照度环境中效果不理想.现实生活中具有很多低照度环境下进行目标检测的实际需要，如夜间监控、夜间无人机侦察与攻击、夜间汽车自动驾驶等工作场景，因此低照度场景下的目标检测具有重要的实用价值.

在基于深度学习的低照度目标检测方法中，可分为先对图像增强再进行检测以及级联训练增强网络和检测器两类方法，如图1（a）和图1（b）所所示.为实现低照度图像目标检测，Loh等人［12］，数字上标提出了低照度图像数据集ExDark，并指出提高低照度图像目标检测性能的两个方向是去噪以及提升亮度.在此基础上，Yuan等人［13］提出面向目标检测的即插即用低光图像增强方法LLIE，实验结果表明该方法对低照度图像目标检测性能有一定提升，但是其增强方式过于简单， 没有考虑到环境光的复杂性，因此提升有限. Xue等人［14］在DETR检测器前加入多尺度金字塔增强网络 MPE，提升低照度图像目标检测的性能.上述方法将低照度目标检测拆分成增强和检测两个步骤，增强模型与检测模型独立训练，因此无法保证增强结果有助于检测网络，很难获得较好的性能.级联训练增强网络和检测模型的方法在一定程度上保证了图像的增强有利于目标检测精度的提高. Chen等人［15］提出YOLO in the Dark通过域适应的方法来合并多个模型，使得新的模型可以适应低照度图像上的目标检测任务，但域适应方法仍然面临成对数据集获取困难的问题. Yin等人［16］ 提出PE-YOLO，通过在检测器之前加入金字塔增强网络并与检测器级联训练，在低照度数据集上取得了较好的效果；Cui等人［17］利用多任务自动编码变换对低照度图像的光照变换进行编码和解码，辅助特征提取网络学习内在的低照度图像退化过程，从而提升检测器的性能；江泽涛等人［18］首先利用一个像素映射提升低照度图像的显著性，然后对主干网络的输出特征进行增强，最终检测器取得了较好的检测性能.上述方法利用增强模型和检测模型的损失函数相加共同训练，在一定程度上保证了图像增强有助于目标检测精度的提高，但是总体损失函数中仍然存在图像增强的部分损失，导致增强网络的优化方向并不完全有助于目标检测任务，部分基于重建的增强网络会导致待检测图像的信息丢失，检测性能提升有限.为解决上述问题，本文所提低照度目标检测方法使用与上述方法不同的框架策略，如图1（c）所示.为约束图像增强网络对低照度图像的增强过程，防止欠拟合导致对图像进行错误的处理，对后续检测产生不利影响，因此图1（b）方法需要加入图像增强损失进行约束.但若是加上增强损失，则根据损失函数的选取和设计，可能需要成对的数据集， 并且数据集分布和损失函数都会影响检测器的优化， 导致检测方法泛化性较差；根据反向传播优化权重的方式，图像增强网络会受到检测损失和增强损失两者的调整，造成检测网络和图像增强网络优化方向不一致，不利于检测任务的性能提升.图1（c）方法将图像光照调整网络和检测器集成在一起，光照调整网络使用检测损失进行优化，其优化方向完全由检测器决定，统一了检测器和光照调整网络的优化方向.由于不需要加入增强损失约束，因此图1（c）方法降低了对数据集的依赖，具有更好的泛化性.但是，为了解决光照调整网络的优化问题，防止无增强损失约束导致欠拟合，光照调整网络的结构和计算过程需要特别设计，并使用基于检测损失对齐的训练方法进行训练.除上述3种模型策略外，江泽涛等人［19］将低照度目标检测视为一种困难目标检测任务，将注意力机制、多尺度特征感知引入正常照度目标检测框架中，提高原有目标检测框架对特征的表达能力，从而提高目标检测精度.  

受局部光照影响，低照度图像的特征提取要比正常照度图像困难，因此本文在低照度目标检测器中设计了一个利用检测损失进行训练且能够无参考调整图像光照的结构.另外，由于低照度图像中包含大量的背景信息，会导致提取到的特征空间中存在大量无用背景特征和噪声特征，严重干扰目标检测模型对目标分类和定位，因此在特征输出阶段引入特征交互注意力机制以对有效特征进行增强.  

综上所述，本文提出ICFIE-YOLO目标检测方法.  本文方法具有以下特点：  
（1）为解决低照度图像特征模糊难以提取和特征空间噪声较多导致检测精度不高的问题，提出基于多尺度光照矫正和特征交互增强的ICFIE-YOLO低照度图像目标检测方法.该方法首先利用提出的多尺度光照矫正网络（Multi Scale Illumination Correction Net⁃ work，MSICN）实现对低照度图像的光照矫正以利于目标特征提取过程；然后使用特征交互增强（Feature In⁃ teracted Enhancement，FIE）检测头滤除特征图中的噪声，实现特征清洗和增强，最终达到较高的检测精度. 在ExDark数据集和DarkFace数据集上的实验表明，所提方法在检测精度和召回率上较其他现有方法均具有更好表现，能够有效提高低照度图像目标检测性能.  

（2）MSICN对低照度图像成像时全局光照弱、局部光源不均匀等不利光照条件进行光照矫正.图像经过光照矫正后，图像中目标的边缘、色彩、纹理得到增强， 突出原本隐藏在背景中的模糊目标，避免有用目标特征消失在黑暗的背景中，图像中的目标特征能够更容易被目标检测器的主干网络提取，解决低照度环境带来的特征提取困难问题.结合基于检测损失对齐的训练方式，统一MSICN与检测器的优化方向，最大可能减少对目标检测不利的调整，保证矫正后图像像素对于
目标检测器的有效性.   

（3）为充分利用有效特征信息，过滤特征中的噪声，提出特征交互增强FIE. FIE工作在检测器的特征编码端，利用交互注意力机制分别对图像特征的空间上下文关联和语义关联进行建模表达，从而达到抑制背景和噪声特征、强化目标特征的目的.  



![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/61494a3756285a871cd8909d87aa4dffed41998c7ed4c8775f70ef177672754f.jpg)

a 先增强再检测


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/486775e79af2edac23948ff120e22732a37bb4dfd1344adf4bb40dad3d10576b.jpg)

(b) 增强与检测级联训练


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/9c6fc116ff3f7a265a7f91d733b340f3d0854382484b9f42cd35a7e8c3133ef3.jpg)

(c) 本文策略框架

图1 　低照度目标检测不同框架策略示意图


# 2　相关工作

# 2. 1　目标检测中引入图像增强方法

为实现高质量的低照度目标检测方法，一个典型的思路是将图像先进行增强，再送入目标检测模型得到检测结果.这一思路可以灵活运用现有的图像增强方法和目标检测方法，在一定程度上解决低照度图像整体亮度不足、光照不均匀、对比度低、信噪比低等问题.目前有很多低照度图像增强算法被提出.早期一般使用基于直方图均衡化的方法，该方法通过扩大图像通道像素值之间的差值提升图像对比度；基于色彩恒常（Retinex）理论，学者提出MSRCR（Multi-Scale Ret⁃ inex with Color Restoration）方法，该方法利用多尺度高斯核卷积计算光照分量，从而解出原始图像［20］；基于大气散射模型理论，Dong等人［21］直接用去雾算法对低光照图像的反转图像处理，经过去雾后再取反得到低照度增强结果.由于上述传统图像增强方法均基于人为设计的模型对增强参数进行估计，并不能很好拟合低照度图像到正常照度图像的映射关系，加上深度学习的快速发展，很快就有学者将神经网络用于低照度图像增强.基于深度学习的图像增强将图像增强视为从低照度图像到正常照度图像的一种映射学习任务，通过对卷积神经网络（Convolutional Neural Network，CNN） 进行合理建模就能得到低照度图像增强网络［22~24］；利用CNN对Retinex理论中的未知参数进行估计，则可以得到基于Retinex理论的低照度图像增强模型［25，26］； Jiang等人［27，28］则通过深度学习模型估计图像增强所需的曲线映射参数，提出无须成对图像的图像增强方法.  

现有低照度图像增强算法较好实现了低照度图像到正常光照图像的转换，但在“先增强，再检测”的低照度目标检测框架中无法保证图像增强的形式有利于后续的目标检测任务，增强任务和检测任务的优化方向并不一致.尽管提升图像亮度确实在一定程度上可以提升低照度图像目标检测效果，但是这种调整应当是以提升目标检测精度为目的的调整，而以增强为目的的低照度图像调整会导致增强后的图像出现色彩畸变、目标特征变形，从而导致待检测图像特征信息出现损坏，被同步放大的噪声和背景信息也会对目标检测任务造成干扰，所以直接将增强后的图像送入目标检测网络很难取得较好的检测精度，文献［16］的研究证实了此观点.
  


# 2. 2　目标检测框架

随着深度学习的发展，基于深度学习的目标检测方法因其优异的性能已经得到广泛的使用.所有基于深度学习的目标检测方法均可划分为特征提取和分类回归两个部分.如图2所示，首先利用特征提取网络提取图像特征，然后再经过编码输出分类和位置信息，其中特征提取器、编码输出均使用CNN或者MLP 实现.   

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/53e95221d3d15aa233c1839a7d43c3d546399cea4caec525a501cbd894159984.jpg)

图2 　目标检测通用框架

目前经典的目标检测方法主要包括单阶段和两阶段方法.单阶段目标检测方法将特征提取、目标定位、 分类集成在一个神经网络模型上，其速度比两阶段的方法快，但是在精度上略微欠佳.两阶段目标检测方法在第一阶段使用一个区域推荐网络（Region Proposal Network，RPN）得到可能存在目标的区域，然后再将此区域的特征送入分类和回归网络得到检测结果，这导致两阶段目标检测方法优劣势恰好与单阶段方法相反，速度较慢，但是精度较高.两阶段目标检测在曾经的一段时间内精度普遍优于单阶段算法，但是随着网络模型结构的优化和多尺度特征金字塔结构的提出与优化［29］，目前单阶段目标检测方法在速度、精度上都比两阶段目标检测方法有优势.单阶段目标检测方法主要思想是首先利用深度网络提取图像特征，然后使用深度神经网络将特征编码为目标类别、 目标大小和位置、目标置信度等信息，再据此设计损失函数进行模型训练.单阶段目标检测方法将特征提取、目标分类、位置定位等过程设计成一个整体， 中间无须其他人工干预，因此可以实现端到端的训练和预测.

对于不同的目标检测方法，其主要区别是特征提取部分结构不一样或者输出编码部分不一样，有大量的改进模型均基于上述两点思想.上述模型及其衍生、 改进模型已经被应用于人脸检测、自动驾驶、智能监控、医学图像检测等领域，目标检测方法及相关应用逐渐成熟，但面对低照度图像目标检测仍然存在许多挑战.  


# 3　本文方法

# 3. 1 ICFIE-YOLO总体设计

由于低照度图像目标特征难于提取且提取到的特征中存在大量噪声，因此正常照度目标检测方法在低照度图像上达不到理想精度.为实现更高效的低照度目标检测器，本文提出一种ICFIE-YOLO低照度目标检测方法. ICFIE-YOLO使用YOLOv7主干网络作为特征提取网络，通过引入多尺度光照矫正网络（MSICN）和特征交互增强（FIE）检测头，实现更加高效的特征提取和特征编码，其结构如图3所示.  

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/a859204b65aa430af30b51a1b6b344e091551192a99bce00a8fccb1e6460d648.jpg)

图3 ICFIE-YOLO整体结构  


针对低照度图像中目标和背景模糊造成特征难以提取以及图像特征中含有大量噪声的现状，使目标检测模型的主干网络更容易提取低照度图像中的特征， 设计MSICN对低照度图像进行光照矫正. MSICN网络包含光照特征提取模块IFE（Illumination Feature Ex⁃ traction）、全局光照矫正模块GIC（Global Illumination Correction）、局部光照矫正模块LIC（Local Illumination Correction）和非线性光照堆叠NLIS（Non-Linear Illumi⁃ nation Stacking）.由于一幅低照度图像中存在一个较弱的全局光源和较少的、影响范围不同的局部光源， 因此IFE模块通过多层级、多尺度卷积操作提取低照度图像的光照信息，然后通过GIC和LIC利用卷积感受野堆叠增长的性质将此光照信息转换为具有多尺度性质的光照矫正系数矩阵.光照矫正系数矩阵指示了图像中全局光照和局部光照的矫正关系，NLIS通过堆叠的乘法操作将光照矫正系数矩阵应用到低照度图像中，实现图像全局光照和多个局部光照的调整. 利用基于检测损失的对齐训练方式仅使用目标检测损失对MSICN中的参数进行调整，补偿低照度图像特征提取过程中暗处目标损失的特征，从而使目标检测模型主干网络更加容易提取到有效目标特征.低照度图像中还存在较多噪声，为抑制特征图中噪声对目标特征造成的影响，提高检测模型对低照度图像中目标的召回率和分类精度，在检测模型特征输出部分设计特征交互增强检测头FIE. FIE使用特征的内部关联建立低照度图像各个区域中目标特征之间的空间关联和语义关联，强化对检测结果具有更多正面影响的特征， 同时过滤特征中的噪声，减少低照度区域特征造成的干扰.  



# 3. 2　多尺度光照矫正网络

MSICN对低照度图像光照进行矫正，使黑暗背景中的目标凸显出来，从而解决低照度图像特征难以提取的问题. MSICN结构如图4所示.  

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/b65b7efd1bb8110e2930bda212ba0b8d5c94451400d0325c4cab89ecbeb6b963.jpg)

图4 MSICN结构图

MSICN使用基于检测损失的对齐训练保证优化方向与检测器一致，因此能够根据检测器的需要对低照度图像进行调整，调整的目的是提高检测精度.由于低照度图像存在全局光照不足和局部光照不均的问题， MSICN需要对图像的全局光照和局部光照做调整.其中，GIC负责生成全局光照矫正所需的参数，LIC生成局部光照矫正所需的参数.为了让GIC和LIC输出的光照矫正参数作用到低照度图像中，并且保证输出的图像像素具有有效性，光照矫正图像输出最终由NLIS使用光照矫正参数对低照度图像经过若干次堆叠的光照曲线调整实现.如图4中光照矫正中间过程图像所示， GIC输出的参数实现对图像全局光照和色调的矫正， LIC输出的参数则对图像中光照不均匀的部分进行调整，最终得到有助于后续检测任务的图像.
为给光照矫正过程提供包含多个尺度的公用特征，同时考虑到光照特征信息所需感受野较大，IFE使用少量小尺寸卷积多层次递进提取图像的光照特征. IFE由6个卷积核大小为3×3、步幅为1的卷积层堆叠而成，并使用ReLU激活函数保证输出非负.连续的卷积层可以提供感受野从小到大的特征图，对应不同尺度的光照特征，同时为了保证后续计算的多尺度特性， IFE还将具有不同感受野的不同层级的多个特征图进行通道维度拼接，最终得到32通道的光照特征.
GIC模块通过对彩色图像RGB的3个通道分别做曲线变换，调整低照度图像得到矫正图像. 3个通道分别利用不同的全局光照系数做曲线变换实现全局光照矫正.因为低照度环境下环境光照中R、G、B颜色分量是不均衡的，为了将全局光照矫正为对目标检测影响最小的白色，对于R、G、B这3个通道要做不同的调整. 由于是对全局光照进行修正，因此该单元利用大尺寸的空洞卷积来获取更大的感受野，同时考虑到全局光照的拟合曲线不会像局部光照那样过于复杂，所以只使用一个浅层卷积网络来拟合全局光照的调整. GIC使用光照特征作为输入，经过2个5×5大小空洞率为3的卷积层得到全局光照特征.全局光照特征经过一个全局平均池化和Sigmoid函数输出为全局光照矫正系数 k，其形状为1×1×3，且其值在（0，1）之间.  

GIC提供的参数仅能矫正全局光照，图像仍然存在局部亮度过高或者过低、对比度过低的问题.因此，LIC 利用光照特征对图像局部光照进行估计，其输出是一个与输入图像同维度的局部光照矫正矩阵.在具有多个局部光源的情况下，多个不同尺度、亮度的局部光照不能像全局光照那样使用一个二次曲线拟合得到，而是需要更加高阶的曲线进行拟合.考虑到局部光照估计的结果是多个不同尺度、亮度的光源照射估计的叠加，且局部光照存在尺度、亮度两个维度上的不同，因此LIC通过多尺度卷积结构拟合不同范围大小的局部光源对图像造成的影响，同时通过多个卷积组的叠加来拟合多个不同亮度但大小相同的局部光源. LIC分为非下采样光照矫正和下采样光照矫正两个部分，非下采样光照矫正使用连续3×3卷积对局部光照调整系数进行计算，可以拟合作用范围较小的局部光源；而下采样光照矫正则用于拟合作用范围较大的局部光源，因此在模型更深层次以获取更大的感受野.考虑到大范围局部光源之间可能具有相关性，因此利用池化核为 3、步幅为1的平均池化模拟光源之间的叠加.为了保证局部光照矫正能够拟合复杂的局部光照矫正过程所需参数，LIC一共输出30通道的局部光照矫正系数，这些光照矫正系数分为10组，通过迭代运算相当于给每个通道都使用二次曲线调整10次，光照矫正系数值域为（-1，1），可以拟合出几乎覆盖值域（0，1）的所有调整曲线.局部光照矫正系数经过激活函数“2Sigmoid（x） -1”输出，保证其值在（-1，1）之间，从而保证后续非线性光照堆叠计算的有效性.  

NLIS模块中没有可训练参数，其作用是将LIC和 GIC中计算得到的光照矫正系数应用到输入图像中，实现最终的光照矫正. NLIS的工作原理是将光照调整系数设置为一个像素映射曲线的参数，且该像素映射曲线需要满足如下要求：

（1）为保证应用该映射后图像像素值之间的大小关系不变，因此该曲线在［0，1］区间必须是单调递增的；  
（2）要提升图像整体亮度，该曲线在区间［0，1］必须处于直线y=x的上方；  
（3）为保证图像像素值的有效性，该曲线的值域应为［0，1］.  

为满足上述要求，本文使用一个经过原点的二次曲线来表示像素映射曲线.因此，GIC输出的全局光照矫正计算过程可由式(1)表示.

$$
I _ { \mathrm { g \_ o u t } } = I _ {\mathrm {i n}} \cdot (- k \cdot I _ {\mathrm {i n}} + (1 + k)), k \in (0, 1) \tag {1}
$$

其中， $I _ { \mathrm { g \_ o u t } }$ 表示输出图像； $I _ { \mathrm { i n } }$ 表示输入图像； $k$ 是有3个元素的一维向量，表示图像3个通道的全局光照矫正系数 . $k$ 是经过 Sigmoid 激活函数输出，因此 $k \in ( 0 , 1 )$ ，对于$I _ { \mathrm { i n } } \in ( 0 , 1 )$ . 式（1）是一个经过原点和点（1，1）的二次曲线，且曲线在（0，1）上均处于直线 $y = x$ 上方，因此可以提高输入图像的通道灰度值，同时保证像素值仍然处于［0，1］之间. 由于不同图像和通道所需要增强的程度不同，即需要不同的增强曲线，神经网络模型可以通过调整 $k$ 的值得到不同的增强曲线.  

对于局部光照调整系数，NLIS将LIC中得到的10组系数用于计算局部光照矫正，其计算过程和计算公式如式（2）.

$$
\left\{ \begin{array}{l} I _ {\text {o u t 1}} = I _ {\text {g} _ {-} \text {o u t}} \cdot \left(k _ {1} \cdot I _ {\text {g} _ {-} \text {o u t}} + \left(1 - k _ {1}\right)\right) \\ I _ {\text {o u t 2}} = I _ {\text {o u t 1}} \cdot \left(k _ {2} \cdot I _ {\text {o u t 1}} + \left(1 - k _ {2}\right)\right) \\ \dots \dots \\ I _ {\text {o u t n}} = I _ {\text {o u t} (n - 1)} \cdot \left(k _ {n} \cdot I _ {\text {o u t} (n - 1)} + \left(1 - k _ {n}\right)\right) \end{array} , n = 10 \tag {2} \right.
$$

其中， $I _ { \mathrm { o u t } ( i ) }$ 表示 LIC 模块第 $i$ 个局部光照矫正单元的输出；该模块的输入是GIC的输出 $I _ { \mathrm { g \_ o u t } }$ ； $k _ { i }$ 表示由网络参数估计得到的局部光照矫正系数，是与输入图像同维度的矩阵. 局部光照存在需要增强或者减弱两种情况，因此 $k _ { i }$ 的值需要规定化到（ -1, 1）. 本文使用激活函数“2Sigmoid（x）-1”输出，显然 $k _ { i } \in ( - 1 , 1 )$ ，当 $k _ { i } > 0$ 时起到增强局部光照作用，当 $k _ { i } < 0$ 时起到减弱局部光照作用. 由于局部光照较全局光照复杂，使用一个二次曲线拟合局部光照的影响是不够的，因此 MSICN 将 10 组局部光照矫正参数以迭代方式进行堆叠. 此种方式可以拟合出一个高幂次的调整曲线以应对复杂的局部光照矫正.

在目标检测任务中，目标检测的损失函数是衡量网络工作性能的唯一定量指标，因此MSICN中的参数均由检测损失反向传播得到. MSICN保证了输出图像的有效性（即输出值在0~1之间），同时给目标检测网络提供自适应光照调整的能力，有效提高低照度图像特征提取效率，从而提升低照度目标检测的性能.

# 3. 3　特征交互增强模块

尽管MSICN在一定程度上缓解低照度图像特征提取困难的问题，但是低照度图像中提取到的特征仍然不可避免地存在噪声干扰，这会导致目标特征混杂在噪声特征中，最终导致目标检测精度下降.目标特征不够显著还会导致目标分类错误.为解决这一问题，本文提出特征交互增强检测头FIE.受自注意力机制能够建立长程依赖关系的启发，FIE通过建立特征图中通道之间、像素之间的交互注意力关联，将这种关联以注意力机制的形式保存，从而获得低照度图像各个区域中目标特征之间的空间关联和语义关联. FIE结构如图5所示.为捕获特征图不同像素之间的特征关联以及同一像素内部的语义关联，FIE具有空间特征交互增强和通道特征交互增强两个分支.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/bade02c33dae07c64f013dc3b94b298b4e164970297814961903d5768985f772.jpg)

图5　FIE结构图


特征图上的每一个像素都是对邻近区域中目标特征的一种表述，在低照度环境下，目标特征、噪声特征和背景特征混淆在一起，对目标检测器的最终输出造成不利影响.空间特征交互增强分支对特征图上的像素进行建模，通过建立特征图像素之间的相关性，增强有用特征并弱化背景和噪声特征，从而实现特征增强. 在空间特征交互增强分支中，通过计算特征图中像素之间的关联性，得到描述像素之间关联的特征图，使隐藏在背景和噪声中的特征得到增强，从而提高目标检测的性能.如图4所示，空间特征交互增强分支将输入
特征图经过 3 个 $1 \times 1$ 的卷积得到输出 ${ \pmb b } , { \pmb c } , { \pmb d } . $ 其中 ${ \pmb b } 、 { \pmb c }的输出通道数为输入通道数的一半，{ \pmb d }$  的通道数与输入一致.  接下来，将 $ { \pmb b } $ 逐像素展平得到 $q , c$ 展平后并转置得到 $\pmb { c } ^ { \mathrm { T } } , \pmb { q }$ 与 $c ^ { \mathrm { T } }$ 相乘并经过Softmax函数得到特征图 $\pmb { u }$ . 特征图 $\pmb { u } \in \pmb { R } ^ { W \cdot H \times W \cdot H }$ 对输入特征中每个像素之间的关系进行描述，通过 Softmax 过滤掉背景和噪声对目标特征造成的干扰. 为将空间特征关联编码进特征图中，将 $\pmb { u }$ 与 $\pmb { d }$ 逐像素展开后的矩阵相乘并调整维度得到 $\pmb { f } _ { s } \in \pmb { R } ^ { W \times H \times C }$ 该过程如式（3）示.


$$
\boldsymbol {f} _ {s} = \operatorname {R e} \left(\text {Soft max} (\boldsymbol {q} \otimes \boldsymbol {c} ^ {\mathrm {T}}) \otimes \text {F l a t t e n i n g} (\boldsymbol {d})\right) \tag {3}
$$

其中， $\otimes$ 表示矩阵乘法，$\operatorname { F l a t t e n i n g } ( \cdot )$表示逐像素展开，$\operatorname { R e } ( \cdot )$ 表示将维度调整到与输入特征图一致. 特征图 $f _ { s }$ 对原始特征中的空间上下文信息进行编码，能够对重要目标特征区域进行增强，并且抑制背景和噪声带来的干扰特征.

特征图中每个通道都描述了目标特征的一个方面，通道中含有越多的有效特征用于描述目标，那么目标检测器就能够达到越好的性能.通道特征交互增强分支通过对特征图中的通道之间进行建模，从而建立特征通道之间的语义关联，以此达到特征增强的目的. 该分支的工作原理与空间特征增强分支类似，首先利用3个并行的3×3卷积对特征图尺寸进行压缩，得到特征图 $x , y , z$ ；然后将 $y$ 逐像素展开并转置与 $ x $ 逐像素展开的矩阵 $\pmb { p }$ 相乘，再经过 Softmax 函数得到特征图 $\pmb { n } .$ 特征图 $\pmb { n } \in \pmb { R } ^ { C \times C }$ 包含了输入特征图中 $C$ 个通道之间的关联，然后将此关联通过矩阵乘法的形式编码到输出特征中，该过程如公式（4）所述

$$
\boldsymbol {f} _ {c} = \operatorname {R e} \left(\text {Flattening} (\boldsymbol {z}) \otimes \text {Softmax}  \left(\boldsymbol {y} ^ {\mathrm {T}} \otimes \boldsymbol {p}\right)\right) \tag {4}
$$

由于 $\pmb { b }$ 和 $\pmb { c }$ 的通道数决定了参与计算像素关联程度的向量长度，参与计算的向量长度越小则该向量中每个值对像素关联的影响越大，因此通过通道数减半的形式筛选出输入特征中较为重要的部分，得到更为显著的空间像素关联描述.在空间交互特征增强分支中，计算空间交互特征的特征图 $\pmb { b }$ 和 $\pmb { c }$ 的通道数为输入特征通道数的一半，其目的是强制对输入特征进行像素上的筛选，从而达到抑制背景和噪声特征的作用.同样， $ x $ 和 $y$ 的长度和宽度决定参与计算通道关联程度的向量长度，为了实现特征通道的筛选，抑制语义信息较弱的无效特征，通道交互特征增强分支中的特征图 $ x $ 和 $y$ 的长宽均为输入特征图的1/2. 为合并空间交互特征 $f _ { s }$ 和通道交互特征 $f _ { c }$ ，FIE最终将输入特征 $f$ 与 $f _ { s } 、
  { f _ { c } }$ 沿通道维度拼接得到交互增强特征.

# 3. 4　基于检测损失的对齐训练

为保证 MSICN 与目标检测器优化方向一致，应当使用检测损失作为其参数调整依据，因此提出使用基于检测损失的对齐训练方法对 进行训练 该方法通过正常照度数据集和低照度数据集图像（不要求成对）在目标检测器上的检测损失进行同方向优化训练，使MSICN学习到两者之间的光照矫正关系，如图 6 所示

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/91d92103915d82a8d43767ffbbb95e470697206afcfdb190640d2672c421053b.jpg)

图6 基于检测损失的对齐训练示意图



基于检测损失的对齐训练过程如下. 首先在正常照度数据集上（例如 MS COCO）训练得到一个目标检测器. 该检测器能够很好地提取正常照度图像 $\mathrm { I M G } _ { \mathrm { H } }$ 的特征并将特征解码为目标检测输出，记该检测器在$\mathrm { I M G } _ { \mathrm { H } }$ 上的损失值为 $\mathrm { L o s s } _ { \mathrm { \scriptscriptstyle H } }$ . 由图6可知，若将低照度图像 $\mathrm { I M G } _ { \mathrm { L } }$ 送入该目标检测器，由于低照度图像低质特性，在正常照度目标检测器上得到的损失值 $\mathrm { L o s s } _ { \mathrm { L } }$ 将会变大 . 将低照度图像 $\mathrm { I M G } _ { \mathrm { L } }$ 经过 MSICN 处理后得到光照矫正图像 $\mathrm { I M G } _ { \mathrm { c } }$ ，将 $\mathrm { I M G } _ { \mathrm { c } }$ 送入目标检测器后，其损失值应当也大于 $\mathrm { L o s s } _ { \mathrm { _ H } }$ ，但是 MSICN 中存在可学习参数，若将目标检测器部分的参数固定与 进行联合训练，最终三者损失值关系应当为 $\mathrm { L o s s _ { \mathrm { L } } } { > } \mathrm { L o s s _ { \mathrm { C } } } { > } \mathrm { L o s s _ { \mathrm { H } } }$ . 由于对应正常照度图像 $\mathrm { I M G } _ { \mathrm { H } }$ 无法获取，在实际计算中无法得到 $\mathrm { L o s s } _ { \mathrm { _ H } }$ ，因此无法直接通过优化损失值的距离对进行优化 但是上述损失均向检测损失趋于的方向进行优化，此时固定目标检测器的参数仅仅使用 $\mathrm { L o s s } _ { \mathrm { c } }$ 优化MSICN参数会使MSICN将低照度图像向有利于正常照度目标检测器特征提取和解码的方向进行优化，从而提升低照度目标检测的精度.这是
MSICN与一般图像增强网络最大的不同，它以目标检测为优化目标，并非提升图像质量.


MSICN与目标检测网络作为一个整体共同训练， 由于低照度目标检测数据集中图像数量、实例数量较少，因此目标检测的主干网络一般使用在大数据集上预训练好的参数.在MSICN学习到从低照度图像到正常照度图像的潜在光照矫正关系后，再解冻目标检测器部分的参数，对全模型参数进行微调训练.

# 4　实验分析

# 4. 1　实验环境基本参数

本文实验在通用低照度目标检测数据集Ex⁃ Dark［12］和低照度人脸检测数据集DarkFace［30］上进行. ExDark数据集有12个类别的7 363张低光照图像.实验中将数据集以8∶2的比例划分成训练集和测试集， 即5 890张图像作为训练集，1 473张图像作为测试集； DarkFace数据集包含了6 000张图像，由于图像总量较小，为保证训练效果，将其中5 400张作为训练集， 600张图像作为测试集.实验模型均使用Tensorflow深度学习框架搭建，运行在具有11 GB显存的GTX1080Ti 图形处理器上.

为验证本文方法的有效性，将本文与目前主流的目标检测方法以及低照度目标检测方法进行对比.由于数据集训练图像较少，因此所有方法的特征提取网络均使用MS COCO数据集上预训练得到的参数作为初
始化参数.在训练过程中，先加载预训练参数并冻结特征提取部分网络参数，所有模型训练均使用Adam优化器，学习率初始值为
 $1 0 ^ { - 2 }$ ，学习率最低为 $1 0 ^ { - 5 }$ .

# 4. 2 ICFIE-YOLO的检测性能

本文在ExDark数据集上实验验证ICFIE-YOLO目标检测方法的有效性，将其与主流的正常光照目标检测方法、低照度目标检测方法以及结合了新近优秀图像增强算法的YOLOv7检测器3类方法进行对比.其中，Retinexformer［31］、HVI-CIDNet［32］、NeRCo［33］都是近年来在图像增强任务上获得较好效果的图像增强方法，按照先增强再检测的策略得到Retinexformer+YO⁃ LOv7、HVICIDNet+YOLOv7、NeRCo+YOLOv7方法.由于上述方法都需要明暗成对的数据集，因此无法在目标检测数据集上以联合检测和增强的形式进行组合. Zero-DCE［28］是近年来较为优秀的无参考低光图像检测方法，将其与YOLOv7检测器分别以先增强再检测和联合增强与检测的方法进行组合得到Zero-DCE+YOLOv7 和Zero-DCE-YOLOv7方法.除EfficientDET使用512像素×512像素图像作为输入，其余方法输入图像大小均为416像素×416像素.表1所示为本文方法与上述其他方法在Exdark数据集上各类别平均精度（AP）和mAP 的比较结果.


表1　在ExDark数据集上不同目标检测算法精度比较  
单位： $\%$


<table><tr><td rowspan="2">算法</td><td colspan="13">类别</td></tr><tr><td>单车</td><td>船</td><td>瓶子</td><td>巴士</td><td>汽车</td><td>猫</td><td>椅子</td><td>杯子</td><td>狗</td><td>摩托</td><td>人</td><td>桌子</td><td>mAP</td></tr><tr><td>YOLOv4[8]</td><td>78.8</td><td>67.3</td><td>71.3</td><td>88.6</td><td>75.1</td><td>60.2</td><td>65.7</td><td>60.9</td><td>72.7</td><td>71.2</td><td>73.5</td><td>54.2</td><td>70.0</td></tr><tr><td>YOLOv5_m[34]</td><td>82.0</td><td>70.1</td><td>72.3</td><td>81.7</td><td>80.8</td><td>67.4</td><td>75.7</td><td>82.8</td><td>77.0</td><td>78.5</td><td>83.7</td><td>59.5</td><td>76.0</td></tr><tr><td>YOLOX_m[7]</td><td>82.3</td><td>71.5</td><td>70.5</td><td>90.1</td><td>77.7</td><td>74.8</td><td>70.5</td><td>72.0</td><td>75.8</td><td>75.8</td><td>77.2</td><td>64.5</td><td>75.2</td></tr><tr><td>YOLOv7[35]</td><td>84.7</td><td>76.5</td><td>74.4</td><td>93.2</td><td>81.6</td><td>81.4</td><td>75.5</td><td>79.8</td><td>87.5</td><td>82.3</td><td>81.1</td><td>64.4</td><td>80.2</td></tr><tr><td>EfficientDET_d0[36]</td><td>77.2</td><td>71.9</td><td>57.9</td><td>90.1</td><td>76.8</td><td>72.7</td><td>64.2</td><td>65.5</td><td>74.7</td><td>74.1</td><td>70.1</td><td>52.5</td><td>70.6</td></tr><tr><td>SSD[9]</td><td>77.7</td><td>66.2</td><td>52.8</td><td>87.4</td><td>72.6</td><td>70.7</td><td>56.8</td><td>56.1</td><td>76.9</td><td>75.0</td><td>63.6</td><td>50.7</td><td>67.2</td></tr><tr><td>Centernet[37]</td><td>77.9</td><td>61.7</td><td>53.6</td><td>84.1</td><td>64.5</td><td>67.2</td><td>56.5</td><td>43.3</td><td>75.6</td><td>67.7</td><td>59.5</td><td>43.2</td><td>62.9</td></tr><tr><td>FCOS[38]</td><td>76.5</td><td>68.0</td><td>66.1</td><td>88.7</td><td>69.5</td><td>71.9</td><td>67.0</td><td>61.5</td><td>75.2</td><td>68.9</td><td>69.0</td><td>54.6</td><td>69.7</td></tr><tr><td>PE-YOLO[16]</td><td>84.7</td><td>79.2</td><td>79.3</td><td>92.5</td><td>83.9</td><td>71.5</td><td>71.7</td><td>79.7</td><td>79.7</td><td>77.3</td><td>81.8</td><td>55.3</td><td>78.1</td></tr><tr><td>MAET[17]</td><td>81.3</td><td>71.6</td><td>74.5</td><td>89.7</td><td>82.1</td><td>69.5</td><td>65.5</td><td>72.6</td><td>75.4</td><td>72.7</td><td>77.4</td><td>53.3</td><td>73.8</td></tr><tr><td>文献[18](YOLOv5_m)</td><td>86.0</td><td>75.0</td><td>76.0</td><td>88.0</td><td>84.0</td><td>75.0</td><td>73.0</td><td>71.0</td><td>76.0</td><td>82.0</td><td>85.0</td><td>62.0</td><td>77.8</td></tr><tr><td>SAM-MSFF[19]</td><td>82.7</td><td>77.8</td><td>65.1</td><td>92.8</td><td>85.2</td><td>77.4</td><td>69.0</td><td>70.2</td><td>78.7</td><td>81.1</td><td>82.5</td><td>62.1</td><td>77.1</td></tr><tr><td>RetinexFormer+YOLOv7</td><td>85.0</td><td>75.7</td><td>74.9</td><td>95.2</td><td>80.6</td><td>79.0</td><td>77.7</td><td>82.4</td><td>83.2</td><td>83.4</td><td>81.3</td><td>64.4</td><td>80.2</td></tr><tr><td>HVICIDNet+YOLOv7</td><td>85.6</td><td>74.3</td><td>74.8</td><td>93.5</td><td>81.8</td><td>74.7</td><td>76.4</td><td>78.8</td><td>86.0</td><td>80.8</td><td>82.1</td><td>66.3</td><td>79.7</td></tr><tr><td>NeRCo+YOLOv7</td><td>73.5</td><td>63.3</td><td>67.1</td><td>85.3</td><td>70.4</td><td>69.1</td><td>66.7</td><td>69.8</td><td>73.4</td><td>68.6</td><td>70.5</td><td>56.6</td><td>69.52</td></tr><tr><td>Zero-DCE+YOLOv7</td><td>85.4</td><td>73.6</td><td>75.6</td><td>93.8</td><td>81.2</td><td>78.1</td><td>76.0</td><td>81.6</td><td>85.1</td><td>79.8</td><td>82.0</td><td>64.2</td><td>79.7</td></tr><tr><td>Zero-DCE-YOLOv7</td><td>84.4</td><td>76.5</td><td>76.6</td><td>92.9</td><td>80.3</td><td>80.7</td><td>76.7</td><td>77.4</td><td>85.6</td><td>81.5</td><td>81.2</td><td>65.5</td><td>79.9</td></tr><tr><td>ICFIE-YOLO</td><td>87.5</td><td>77.6</td><td>77.2</td><td>94.4</td><td>83.8</td><td>81.1</td><td>78.4</td><td>85.1</td><td>88.0</td><td>83.2</td><td>83.1</td><td>68.2</td><td>82.3</td></tr></table>

由表1可以看出，主流正常照度下的目标检测方法在低照度数据集上难以获得理想性能，尽管YOLOv5、 YOLOv7和YOLOX检测器通过优化特征提取网络和检测头的形式取得了一定程度上的精度提升，但效果与低照度目标检测方法还是存在差距.本文所提ICFIE YOLO通过优化检测图像质量和特征增强，在大多数类别的检测精度上都具有优势，多数类别平均精度在所有比较的方法中都取得最优或次优结果，各类别平均精度达到82. 3%.文献［17］、文献［18］方法将图像矫正网络与目标检测网络进行联合训练，但是文献［17］方法建立在假设的图像退化模型上，这与真实低照度图像目标分布不一致，导致其性能提升有限；文献［18］使用像素高阶映射模块来解决低照度图像特征不显著的问题，但是该模块仍然使用独立的损失函数，因此尽管将损失函数加入到目标检测的损失函数一同进行优化，缓解了优化方向不一致的问题，但仍然不能完全解决优化方向不一致的问题，在低照度目标检测数据集上的效果提升有限. PE-YOLO［16］方法通过一个级联训练的图像增强网络提升低照度图像检测精度，增强网络中使用了拉普拉斯分解和重建，而重建过程可能导致图像信息损失. RetinexFormer方法在低光图像增强数据集LOL上获得的PSNR和SSIM指标（越高越好）分别是25.16和0.845，而NeRCo和Zero-DCE在同样的指标上取得的值分别是19.00和0.536、14.08和0.561，因此在图像增强任务上，前者显著强于后面两者.但Ret⁃ inexFormer+YOLOv7的检测性能与Zero-DCE+YOLOv7 差距并不大，NeRCo+YOLOv7却取得了相对最差的结果.这是因为“先增强，后检测”的策略中，图像增强的结果不是以目标检测精度为导向的，这种增强对于低照度目标检测而言具有更大的不确定性. NeRCo+YO⁃ LOv7的结果也证明了不适合目标检测的图像增强会导致图像中的目标特征畸形，导致后续检测性能下降.从上述实验结果可以看出，一方面，检测结果及增强方法与低照度图像的增强结果好坏没有直接关系；另一方面，增强方法的性能很大一部分取决于训练数据集，但是目标检测环境下无法保证被检测的低照度图像分布与增强模型的训练集一致，因此这种策略效果十分有限.“联合增强和检测”的策略一定程度上使增强方法对图像的增强有利于后续目标检测，因此Zero-DCE YOLOv7的性能略优于Zero-DCE+YOLOv7，但是损失函数的优化方向仍然有部分倾向于图像增强，这就导致了检测效果受限.

为更加直观对比本文方法与主流目标检测方法检测效果的差异，图7展示了在ExDark数据集上检测结果的可视化对比.可以看出，由于低照度图像质量退化的影响，一般目标检测方法对于较暗区域的目标无法检出.具体来说，由于一些目标在图像中较暗的区域， 这些目标与昏暗的背景几乎融合在一起，导致特征提取存在困难，检测器难以区分背景和目标特征，因此存在较多的目标漏检.本文所提出的方法首先通过光照矫正使目标从背景中凸显出来，背景中的目标特征提取更加顺利，使得许多在暗处原本无法检出的目标被检出；然后通过特征交互增强强化提取到的目标特征并滤除背景和噪声特征，提供给检测头的特征更加清晰、有效，使目标分类更加准确.图7中最后一行右下角的桌子与背景几乎融为一体，仅有ICFIE-YOLO能够将其检测出来，本文方法确实能够更加高效地检出黑暗中的目标.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/a6c1067cb8963397ed25af535cff9261528ea0cec23ccd79c96f184cb8776a9e.jpg)

图7 本文方法与其他主流目标检测方法在ExDark数据集上可视化检测结果


为验证ICFIE-YOLO的泛化性，将所提方法在 DarkFace数据集上与性能较好的目标检测方法的平均精度（AP）、50%目标阈值下的准确率（P）以及召回率 （R）进行对比，结果如表2所示.


表2　在Darkface数据集上不同目标检测算法性能比较 单位 $\%$


<table><tr><td>方法</td><td>AP</td><td>P</td><td>R</td></tr><tr><td>YOLOv7</td><td>52.19</td><td>89.79</td><td>33.48</td></tr><tr><td>HVICIDNet+YOLOv7</td><td>50.02</td><td>86.56</td><td>28.56</td></tr><tr><td>RetinexFormer+YOLOv7</td><td>50.82</td><td>88.96</td><td>27.57</td></tr><tr><td>Zero-DCE+YOLOv7</td><td>43.27</td><td>82.72</td><td>29.05</td></tr><tr><td>Zero-DCE-YOLOv7</td><td>48.79</td><td>91.17</td><td>28.12</td></tr><tr><td>文献[18](YOLOv5_m)</td><td>51.25</td><td>84.70</td><td>37.47</td></tr><tr><td>ICFIE-YOLO</td><td>55.94</td><td>91.53</td><td>37.41</td></tr></table>

DarkFace数据集中的图像光照较ExDark更弱，并且具有更多的小尺寸目标.由表2可见，所提ICFIE-YOLO在DarkFace数据集上各指标均取得较好效果，平均精度达到55.94%，显著优于其他方法.由于Dark⁃Face数据集图像更加劣质，其数据分布与HVICIDNet和RetinexFormer方法的训练数据差异更大，因此采用“先增强后检测”的策略组合检测性能下降较ExDark数据集更加严重，Zero-DCE同样难以处理照度更低的图像，因此检测效果也出现了较大下降. Zero-DCE-YOLOv7采用“联合增强和检测”的策略，增强结果对于目标检测而言相对有利，因此其性能稍有提升.文献［18］也采用“联合增强和检测”的策略，其中的长距离特征捕获模块使其在召回率上取得了不错的效果，但是由于增强网络对目标特征造成了不利影响，因此准确率并不理想.


# 4. 3　消融实验

为验证ICFIE-YOLO方法中所提MSICN和FIE结构对低照度图像目标检测的作用和有效性，对模块进行消融实验，除了针对所提MSICN和FIE进行消融外， 还对MSICN内部的全局光照矫正GIC和局部光照矫正LIC进行了消融，结果如表3所示.

表3 MSICN和FIE的有效性验证


<table><tr><td rowspan="2">基线模型</td><td colspan="2">MSICN</td><td rowspan="2">FIE</td><td rowspan="2">mAP/%</td><td rowspan="2">Recall@0.5/%</td></tr><tr><td>GIC</td><td>LIC</td></tr><tr><td>√</td><td></td><td></td><td></td><td>80.2</td><td>67.3</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>81.4</td><td>69.8</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>82.3</td><td>69.9</td></tr><tr><td>√</td><td></td><td></td><td>√</td><td>81.8</td><td>66.4</td></tr><tr><td>√</td><td></td><td>√</td><td>√</td><td>81.9</td><td>69.0</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>81.5</td><td>69.2</td></tr></table>

表3中基线模型使用YOLOv7-l，在加入MSICN后检测mAP增加1.2个百分点，引入FIE后mAP再提高 0.9个百分点.在加入MSICN后，检测器在50%置信度下的召回率提升到69.8%.这表明在单独加入MSICN 后，模型对于目标的检出率和分类精度都有所提高，本来难以检出的目标在经过MSICN调整后被检出，低照度图像经过MSICN调整后检测器也能够提取到更加丰富的特征，因此mAP也得到提升.再加入FIE模块mAP 得到进一步提升，但召回率仅有小幅度提升，这是因为 FIE的目的是过滤主干网络提取的特征中的噪声，使得检测头能够获取到高质量特征并且进行有效表达，从而增加检测头对目标类别预测的准确率，因此mAP得到进一步提升而召回率小幅提升.当基线模型仅仅加入FIE时也能够提升mAP，但是召回率却略有下降.本文认为造成此现象的原因是低照度图像未经过光照调整，导致部分隐藏在背景中的目标特征提取不足，尽管特征经过了FIE进行增强，但仅仅是增强了图像中那些被成功提取到特征的目标的特征，而特征提取不足的目标被漏检，因而单独使用FIE时召回率没有提升. MSICN能够解决FIE计算过程中目标特征质量不佳的问题，MSICN内部的GIC和LIC对低照度图像光照进行调整，有效帮助后续特征提取网络更好地对目标特征进行提取，因此单独加入GIC和LIC都能够提升目标检测的召回率.

另外，为验证MSICN对图像光照调整的处理效果， 将原图像与MSICN、仅使用LIC、仅使用GIC以及Ret⁃ inexFormer输出的图像进行对比，结果如图8所示. MSICN的目的是根据检测网络的需求调整被检测图像中的光照，使图像的特征更容易被检测网络提取，尽可能减少目标特征信息丢失，因此对图像的调整并不如 RetinexFormer等低照度图像增强网络那样明显.但是由图8可以看出，MSICN确实在一定程度上提升了图像整体亮度和对比度，对于图像目标存在的区域光照调整更为自然，并且没有造成图像失真. RetinexFormer增强后的图像在视觉效果上好于MSICN，但是图像部分区域存在失真和大量噪声，这会导致目标特征变形，影响检测效果.因此，MSICN确实提高了目标检测器的检测性能，达到了其设计目的.

最后，为验证FIE对特征的增强作用，将FIE模块输出的特征图与不带有FIE模块输出的特征图使用热力图的形式进行可视化.图9展示了特征图的可视化效果，图中颜色越红的区域表示特征图中的目标置信度越高.可见，FIE模块可以使输出特征更加准确地聚焦在目标上，使目标定位更加准确，原本由于置信度较低导致漏检的目标也得以检出，最终提升检测模型的准确率.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/8206cf5d8d70b7a88b6b49a92296901f50f9a619534a90c2deda15df8c1a534d.jpg)

图8　MSICN、LIC、GIC和RetinexFormer输出图像与原图像



![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/e1bcf2cdc009b3c02c7966efb8750a681582ee1d1d6e2669ef85f27a52ef3177.jpg)

图9　有无FIE模块的检测器输出特征可视化对比



# 4. 4　性能代价分析

ICFIE-YOLO结合了MSICN和FIE能够更进一步提升检测方法的检测性能，但会造成一定的内存和计算开销，为量化分析参数量和推理速度代价，将ICFIE YOLO的参数量和推理时间与基线模型进行对比，结果如表4所示.


表4 ICFIE-YOLO时间、空间代价分析


<table><tr><td>方法/模块</td><td>输入尺寸</td><td>推理时间/ms</td><td>可训练参数量</td></tr><tr><td>YOLOv7</td><td>640</td><td>51</td><td>37.3×10^6</td></tr><tr><td>YOLOv7</td><td>416</td><td>29</td><td>37.3×10^6</td></tr><tr><td>ICFIE-YOLO</td><td>640</td><td>68</td><td>75.7×10^6</td></tr><tr><td>ICFIE-YOLO</td><td>416</td><td>40</td><td>75.7×10^6</td></tr><tr><td>MSICN</td><td>416/640</td><td>—</td><td>0.85×10^6</td></tr><tr><td>FIE</td><td>416/640</td><td>—</td><td>38.2×10^6</td></tr></table>

以上结果均在GTX1080Ti图形处理器上计算得到，检测方法均使用12分类输出.由于MSICN和FIE 模块的参数量均由卷积层产生，因此参数量仅受输入通道数影响，其中MSICN输入通道数为3，FIE接受YOLOv7的3个特征输出层作为输入，通道数分别为256、 512和1 024.由表4中的实验结果可见，FIE中需要依赖卷积生成交互特征，因此导致参数量较大，达到了$3 8 . 2 \times 1 0 ^ { 6 }$ ，而 仅需要 $0 . 8 5 \times 1 0 ^ { 6 }$ 参数. ICFIE-YOLO 在416像素×416像素的图片输入和640像素×640像素的图片输入时检测推理时间较YOLOv7分别增加了 11 ms和17 ms，这对于整个目标检测模型的计算速度而言是可以接受的.虽然MSICN和IFE导致模型的参数量和推理时间有所增加，但是模型的mAP、准确率和召回率均有明显特征.

# 5　结论

为解决低照度图像中全局光照不足或局部光照不均匀带来的特征难以提取、特征空间噪声大导致目标检测困难的问题，本文提出基于多尺度光照矫正和特征交互增强的ICFIE-YOLO低照度目标检测方法.所提方法首先利用一个多尺度光照矫正网络（MSICN）结合基于检测损失的对齐训练对低照度图像光照进行矫正，使低照度图像中的目标特征能更容易被主干网络提取；然后，提出特征交互增强（FIE）结构，建立特征在空间和通道上的关联关系，从而对特征空间中的目标特征进行增强并过滤背景和噪声特征，最终提高低照度目标检测精度.在ExDark和DarkFace数据集上的实验结果表明，相较于主流目标检测方法和现有低照度目标检测方法，本文方法在低照度环境下的目标检测任务上具有更好的检测性能.考虑到目前低照度目标检测数据集数据量少，成对的正常照度-低照度图像目标检测数据集难以获取，本文所提ICFIE-YOLO在无成对低照度图像数据集的情况下实现了较高精度的目标检测，因此具有较好的潜在应用价值.下一步将进一步优化所提算法，尝试提升更低照度图像的目标检测精度，在提升检测精度的同时实现检测模型轻量化，提升检测速度.

# 参考文献
## ps: 以下参考文献有需要到再看, 可能有错误


［1］ REN S Q, HE K M, GIRSHICK R, et al. Faster R-CNN:
Towards real-time object detection with region proposal networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017, 39(6): 1137-1149.





［2］ KIM H, LEE E C, SEO Y, et al. Character detection in ani⁃
mated movies using multi-style adaptation and visual atten⁃ tion[J]. IEEE Transactions on Multimedia, 2020, 23: 1990- 2004.




［3］ HE K M, GKIOXARI G, DOLLÁR P, et al. Mask R-
CNN[EB/OL]. (2018-01-24)[2024-07-09]. https://arxiv.org/ abs/1703.06870.





［4］ ZHU Y S, ZHAO C Y, WANG J Q, et al. CoupleNet: Coupling global structure with local parts for object de⁃ tection[C]//2017 IEEE International Conference on Com⁃ 





puter Vision (ICCV). Piscataway: IEEE, 2017: 4124-4134. 





［5］ WANG X L, SHRIVASTAVA A, GUPTA A. A-fast-RCNN: Hard positive generation via adversary for object detection[C]//2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway: IEEE, 2017: - . 





［6］ REDMON J, FARHADI A. YOLOv3: An incremental im⁃ provement[EB/OL]. (2018-04-08)[2024-07-09]. https://arx⁃ iv.org/abs/1804.02767v1. 





［7］ GE Z, LIU S T, WANG F, et al. YOLOX: Exceeding YO⁃ LO series in 2021[EB/OL]. (2021-08-06) [2024-07-09]. https://arxiv.org/abs/2107.08430v2. 





［8］ BOCHKOVSKIY A, WANG C Y, LIAO H M. YOLOv4: Optimal speed and accuracy of object detection[EB/OL]. (2020-04-23)[2024-07-09]. https://arxiv.org/abs/2004.10934v1. 





［9］ PAN H D, JIANG J, CHEN G F. TDFSSD: Top-down fea⁃ ture fusion single shot MultiBox detector[J]. Signal Pro⁃ cessing: Image Communication, 2020, 89: 115987. 





［10］ CUI L S, MA R, LV P, et al. MDSSD: Multi-scale decon⁃ volutional single shot detector for small objects[J]. Sci⁃ ence China Information Sciences, , ( ): . 





［11］ VAIDWAN H, SETH N, PARIHAR A S, et al. A study on transformer-based object detection[C]// Interna⁃ tional Conference on Intelligent Technologies (CONIT). Piscataway: IEEE, : - . 





［12］ LOH Y P, CHAN C S. Getting to know low-light images with the Exclusively Dark dataset[J]. Computer Vision and Image Understanding, 2019, 178: 30-42. 





［13］ YUAN J J, HU Y L, SUN Y F, et al. A plug-and-play im⁃ age enhancement model for end-to-end object detec⁃ tion in low-light condition[J]. Multimedia Systems, 2024, 30(1): 27. 





［14］ XUE R, DUAN J L, DU Z W. MPE-DETR: A multiscale pyramid enhancement network for object detection in low-light images[J]. Image and Vision Computing, 2024, : . 





［15］ CHEN C, CHEN Q F, XU J, et al. Learning to see in the dark[C]//2018 IEEE/CVF Conference on Computer Vi⁃ sion and Pattern Recognition. Piscataway: IEEE, 2018: - . 





［16］ YIN X C, YU Z D, FEI Z T, et al. PE-YOLO: Pyramid enhancement network for dark object detection[M]//Arti⁃ ficial Neural Networks and Machine Learning - ICANN 2023. Cham: Springer, 2023: 163-174. 





［17］ CUI Z T, QI G J, GU L, et al. Multitask AET with orthog⁃ onal tangent regularity for dark object detection[C]// 





2021 IEEE/CVF International Conference on Computer Vision (ICCV). Piscataway: IEEE, 2021: 2533-2542. 





［18］江泽涛, 翟丰硕, 钱艺, 等. 结合特征增强和多尺度感受野的低照度目标检测[J]. 计算机研究与发展, 2023,60(4): 903-915.JIANG Z T, ZHAI F S, QIAN Y, et al. Low illuminationobject detection combined with feature enhancement andmulti-scale receptive field[J]. Journal of Computer Re⁃search and Development, 2023, 60(4): 903-915. (in Chi⁃nese)





［19］江泽涛, 李慧, 雷晓春, 等. 一种基于SAM-MSFF网络的低照度目标检测方法[J]. 电子学报, 2024, 52(1): 81-93.JIANG Z T, LI H, LEI X C, et al. A low-light object de⁃tection method based on SAM-MSFF network[J]. ActaElectronica Sinica, 2024, 52(1): 81-93. (in Chinese)





［ ］ PARTHASARATHY S, SANKARAN P. Fusion based multi scale RETINEX with color restoration for image en⁃ hancement[C]//2012 International Conference on Com⁃ puter Communication and Informatics. Piscataway: IEEE, 2012: 1-7. 





［21］ DONG X, WANG G, PANG Y, et al. Fast efficient algo⁃ rithm for enhancement of low lighting video[C]//2011 IEEE International Conference on Multimedia and Expo. Piscataway: IEEE, 2011: 1-6. 





［ ］江泽涛, 覃露露. 一种基于U-Net生成对抗网络的低照度图像增强方法[J]. 电子学报, , ( ): - .JIANG Z T, QIN L L. Low-light image enhancement methodbased on U-Net generative adversarial network[J]. ActaElectronica Sinica, 2020, 48(2): 258-264. (in Chinese)





［23］ LU B B, PANG Z B, GU Y N, et al. Channel splitting at⁃ tention network for low-light image enhancement[J]. IET Image Processing, 2022, 16(5): 1403-1414. 





［ ］江泽涛, 钱艺, 伍旭, 等. 一种基于ARD-GAN的低照度图像增强方法[J]. 电子学报, , ( ): - .JIANG Z T, QIAN Y, WU X, et al. Low-light image en⁃hancement method based on ARD-GAN[J]. Acta Elec⁃tronica Sinica, 2021, 49(11): 2160-2165. (in Chinese)





［ ］江泽涛, 覃露露, 秦嘉奇, 等. 一种基于MDARNet的低照度图像增强方法[J]. 软件学报, , ( ): -3991.JIANG Z T, QIN L L, QIN J Q, et al. Low-light image en⁃hancement method based on MDARNet[J]. Journal ofSoftware, 2021, 32(12): 3977-3991. (in Chinese)





［26］ SHANG X K, AN N, ZHANG S M, et al. Toward robust and efficient low-light image enhancement: Progressive attentive retinex architecture search[J]. Tsinghua Science 





and Technology, 2023, 28(3): 580-594. 





［27］ JIANG Y F, GONG X Y, LIU D, et al. EnlightenGAN: Deep light enhancement without paired supervision[J]. IEEE Transactions on Image Processing, 2021, 30: 2340- 2349. 





［28］ GUO C L, LI C Y, GUO J C, et al. Zero-reference deep curve estimation for low-light image enhancement[C]// 2020 IEEE/CVF Conference on Computer Vision and Pat⁃ tern Recognition (CVPR). Piscataway: IEEE, 2020: 1777- 1786. 





［ ］陈科圻, 朱志亮, 邓小明, 等. 多尺度目标检测的深度学习研究综述[J]. 软件学报, 2021, 32(4): 1201-1227.CHEN K Q, ZHU Z L, DENG X M, et al. Deep learningfor multi-scale object detection: A survey[J]. Journal ofSoftware, 2021, 32(4): 1201-1227. (in Chinese)





［30］ YANG W H, YUAN Y, REN W Q, et al. Advancing im⁃ age understanding in poor visibility environments: A col⁃ lective benchmark study[J]. IEEE Transactions on Image Processing, 2020, 29: 5737-5752. 





［31］ CAI Y H, BIAN H, LIN J, et al. Retinexformer: One-stage ret⁃ inex-based transformer for low-light image enhancement[C]// 2023 IEEE/CVF International Conference on Computer Vi⁃ sion (ICCV). Piscataway: IEEE, 2023: 12470-12479. 





［32］ YAN Q S, FENG Y X, ZHANG C, et al. You only need one color space: An efficient network for low-light image 



# 作者简介：

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/d6f9eed8b137ef92c8b702169ea0c837e71203ec0f3172ea8bc84e71eee521f1.jpg)


秦嘉奇 男，博士研究生，讲师，系统架构设计师 主要研究领域为计算机视觉

E-mail: 18878396109@163.com 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-26/c693685b-d346-4308-8fff-806dbbef1d0e/1cb6cb8a7934000f9c93c5406686c08d1016c1b86d9035e08ff2a57432389f3c.jpg)


江泽涛 男，博士，教授，博士生导师. 主要研究领域为深度学习、计算机视觉

E-mail: zetaojiang@guet.com 



enhancement[EB/OL]. (2024-06-17)[2024-07-09]. https:// arxiv.org/abs/2402.05809v3. 





［33］ YANG S Z, DING M X, WU Y M, et al. Implicit neural rep⁃ resentation for cooperative low-light image enhancement[C]// 2023 IEEE/CVF International Conference on Computer Vision (ICCV). Piscataway: IEEE, 2023: 12872-12881. 





［34］ JOCHER G, NISHIMURA K, MINEEVA T. YOLOv5[EB/ OL]. (2022-11-12) [2024-07-09]. https://github. com/ultra⁃ lytics/yolov5. 





［35］ WANG C Y, BOCHKOVSKIY A, LIAO H M. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for re⁃ al-time object detectors[EB/OL]. (2022-07-06) [2024-07- 09]. https://arxiv.org/abs/2207.02696v1. 





［36］ TAN M X, PANG R M, LE Q V. EfficientDet: Scalable and efficient object detection[C]//2020 IEEE/CVF Confer⁃ ence on Computer Vision and Pattern Recognition (CVPR). Piscataway: IEEE, 2020: 10781-10790. 





［37］ DUAN K W, BAI S, XIE L X, et al. CenterNet: Keypoint triplets for object detection[C]//Proceedings of the IEEE/ CVF International Conference on Computer Vision. Pis⁃ cataway: IEEE, 2019: 6568-6577. 





［38］ TIAN Z, SHEN C H, CHEN H, et al. FCOS: Fully convo⁃ lutional one-stage object detection[C]//2019 IEEE/CVF International Conference on Computer Vision (ICCV). Piscataway: IEEE, 2019: 9627-9636. 
