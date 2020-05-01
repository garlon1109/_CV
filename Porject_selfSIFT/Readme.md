## 特征描述算法

### What is a good feature point?

#### Harris Corner

- Very informational (Harris Corner Detector)
- Rotation/Brightness resistance (Harris Corner Detector)
- Scale resistance (Harris Corner Detector)

### What is the form of a feature point?

- Physical in location
- Abstract in formation (usually a vector) -> Feature Descriptor

### How to get a feature point/descriptor?

#### SIFT

[SIFT](https://www.cnblogs.com/hepc/p/9636474.html)，即尺度不变特征变换（Scale-invariant feature transform，SIFT），是用于 [图像处理](https://baike.baidu.com/item/图像处理/294902) 领域的一种描述。这种描述具有尺度不变性，可在图像中检测出关键点，是一种局部特征描述子。

SIFT 特征是基于物体上的一些局部外观的兴趣点而与影像的大小和旋转无关。对于光线、噪声、微视角改变的容忍度也相当高。基于这些特性，它们是高度显著而且相对容易撷取，在母数庞大的特征数据库中，很容易辨识物体而且鲜有误认。使用 SIFT 特征描述对于部分物体遮蔽的侦测率也相当高，甚至只需要 3 个以上的 SIFT 物体特征就足以计算出位置与方位。在现今的电脑硬件速度下和小型的特征数据库条件下，辨识速度可接近即时运算。SIFT 特征的信息量大，适合在海量数据库中快速准确匹配。

1. Generate Scale-space: DoG (Difference of Gaussian)

    - 采用高斯函数对图像进行模糊以及降采样处理得到高斯金字塔
    - 采用 DOG 在高斯金子塔中的每组中相邻两层相减（“下一层减上一层”）生成高斯差分金字塔

2. Scale-space Extrema Detection

    找到局部极值点

3. Accurate Keypoint Localization

    通过 Taylor 展开式（插值函数）精确定位关键点

4. Eliminating Edge Responses

    通过 [Hessian 矩阵](http://www.docin.com/p-940982857.html) 消除边缘响应点

5. Orientation Assignment

    对关键点进行梯度计算生成梯度直方图，统计领域内像素的梯度和方向，从而确定主（辅）方向，计算梯度大小和方向时使用了 `prewitt` 算子

    $$ 𝑟 =3*1.5𝜎 $$

    $$ 𝑚(𝑥, 𝑦) = \sqrt{(𝐿(𝑥 + 1, 𝑦) − 𝐿(𝑥 − 1, 𝑦))^2 + (𝐿(𝑥, 𝑦 + 1) − 𝐿(𝑥, 𝑦 − 1))^ 2} = \sqrt{G_x^2-G_y^2} $$

    $$ 𝜃(𝑥, 𝑦) = arctan(\frac{𝐿(𝑥, 𝑦 + 1) − 𝐿(𝑥, 𝑦 − 1)}{𝐿(𝑥 + 1, 𝑦) − 𝐿(𝑥 − 1, 𝑦)}) = arctan(\frac{G_y}{G_x}) $$

6. Keypoint Descriptor

    取特征点周围 `4*4` 个区域块，统计每小块内 8 个梯度方向的幅值，用这 `4*4*8=128` 维向量作为 SOFT 特征的描述子

有了关键点描述子之后，就可以用相关算法对不同图片中的关键点描述子进行匹配。

#### HoG

梯度方向直方图 (Histogram of Oriented Gradient, HOG) 是图像处理经典的特征提取算法。HOG 特征是直接将图像像素点的方向梯度作为图像特征，包括梯度大小和方向。通过计算图像局部区域的梯度直方图特征，然后将局部的特征串联起来，构成整幅图像的 HOG 特征。

- 主要思想

    在一幅图像中，局部目标的表象和形状（appearance and shape）能够被梯度或边缘的方向密度分布很好地描述。（本质：梯度的统计信息，而梯度主要存在于边缘的地方）。

- 实现方法

    首先将图像分成小的连通区域，我们把它叫细胞单元。然后采集细胞单元中各像素点的梯度的或边缘的方向直方图。最后把这些直方图组合起来就可以构成特征描述器。

- 提高性能

    把这些局部直方图在图像的更大的范围内（我们把它叫区间或 block）进行对比度归一化（contrast-normalized），所采用的方法是：先计算各直方图在这个区间（block）中的密度，然后根据这个密度对区间中的各个细胞单元做归一化。通过这个归一化后，能对光照变化和阴影获得更好的效果。

- 优点

    与其他的特征描述方法相比，HOG 有很多优点。首先，由于 HOG 是在图像的局部方格单元上操作，所以它对图像几何的和光学的形变都能保持很好的不变性，这两种形变只会出现在更大的空间领域上。其次，在粗的空域抽样、精细的方向抽样以及较强的局部光学归一化等条件下，只要行人大体上能够保持直立的姿势，可以容许行人有一些细微的肢体动作，这些细微的动作可以被忽略而不影响检测效果。因此 HOG 特征是特别适合于做图像中的人体检测的。

#### RANSAC

随机抽样一致算法（RANdom SAmple Consensus, RANSAC）,采用迭代的方式从一组包含离群的被观测数据中估算出数学模型的参数。RANSAC 算法假设数据中包含正确数据和异常数据（或称为噪声）。正确数据记为内点（inliers），异常数据记为外点（outliers）。同时RANSAC也假设，给定一组正确的数据，存在可以计算出符合这些数据的模型参数的方法。该算法核心思想就是随机性和假设性，随机性是根据正确数据出现概率去随机选取抽样数据，根据大数定律，随机性模拟可以近似得到正确结果。假设性是假设选取出的抽样数据都是正确数据，然后用这些正确数据通过问题满足的模型，去计算其他点，然后对这次结果进行一个评分。

RANSAC 算法被广泛应用在计算机视觉领域和数学领域，例如直线拟合、平面拟合、计算图像或点云间的变换矩阵、计算基础矩阵等方面，使用的非常多。

例如拟合一个直线模型，使用 RANSAC 算法可以遵循如下步骤：

1. 需要两个点唯一确定一个直线方程，所以第一步随机选择两个点
2. 通过这两个点，可以计算出这两个点所表示的模型方程 y=ax+b，计算其系数和截距
3. 将所有的数据点套到这个模型中计算误差
4. 找到所有满足误差阈值的点的个数
5. 重复前 4 步迭代过程，直到达到一定迭代次数后，选出那个满足指定误差点数最多的模型，作为问题的解

#### 小结

SelfSift为主文件 依次实现了SIFT差分金字塔图像预处理 harris角点检测 hessian边沿检测  描述子生成 特征匹配 自己根据原理实现了SIFT 
但是无奈暴力循环太多 图像生成不理想 
Test为图像变换应用
Test2为使用opencv完成的通过RANSAC过滤badpoint的 SIFT算法