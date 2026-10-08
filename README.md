# 实验二：图像增强

## 一、实验目的

学会 OpenCV 的基本使用方法，利用 OpenCV 等计算机库对图像进行平滑、滤波等操作，实现图像增强。

## 二、实验内容

### 2.1 导入图像滤波相关的依赖包

- OpenCV 库：计算机视觉与图像处理
- scikit-image 库：图像增强与分析，random_noise 用于给图像添加噪声
- NumPy 库：科学计算的基础库
- Matplotlib 库：绘图与数据可视化

导入依赖库：

```python
import cv2
from skimage.util import random_noise
import numpy as np
from matplotlib import pyplot as plt
```

### 2.2 读取原始图像并进行色彩空间转换
读取计算机本地图像文件，获取并输入【100，100】处像素点的 RBG 参数并输出，通过下面的输出结果可以看到，【100，100】像素点的 RGB 为【72, 210, 239】，表现为偏蓝色，通过图像也可以看到这个像素点属于猫猫的衣服处的深蓝色部位。

```
img = cv2.imread('p1.jpg')
# 获取图像中【100，100】这个像素的 rgb 三色
(b, g, r) = img[100, 100]
# 打印这个像素点的 rgb 参数
print(b, g, r)
# 输出原始图像
plt.imshow(img)
plt.title('Original Image')
plt.savefig('output_images/original_bgr.jpg', dpi=300)
plt.show()
```

<img width="1180" height="1194" alt="ab99d83850f00c643ab8d5f32d481ad" src="https://github.com/user-attachments/assets/b6a2b84b-67c4-4fb8-9162-c745f3c5bbd2" />

<img width="861" height="740" alt="b9a419894438ad06ac77a96fca76d32" src="https://github.com/user-attachments/assets/c2d16cdf-f7bc-457f-9960-003d206e0f40" />
<img width="960" height="547" alt="46b98914af44d45510f37ba5fd2dc64" src="https://github.com/user-attachments/assets/35f7c7be-4f03-49bc-a6be-b2652e21b001" />

输出：
82 86 241

#灰度图片
<img width="922" height="764" alt="5745e2edd05761c97ba7706def88fe1" src="https://github.com/user-attachments/assets/23b4e119-951e-4186-8441-33f370fc9a2f" />


### 2.3 添加噪声
这里在原始图像的基础上添加噪声，引入了两个 API 方法，椒盐噪声和高斯噪声，通过对比可以发现，椒盐噪声和高斯噪声的本质不同，椒盐噪声表现为像素会随机替换为白色或者黑色像素（灰度通道），在 RGB 通道表现为像素变成随机彩色点，而高斯噪声会在每个像素上添加随机偏差，服从高斯分布。
```
# ====================== 添加噪声 ======================
# mode='s&p' 代表椒盐噪声，s 代表白色，p 代表黑色，amount=0.4 代表会有 40% 的像素被替换
sp_noise_img = random_noise(rgb_img, mode='s&p', amount=0.4)

# mode='gaussian' 代表高斯噪声，会在每个像素上添加随机偏差，服从高斯分布
gaus_noise_img = random_noise(rgb_img, mode='gaussian', mean=0.2, var=0.03)

# 原图
plt.subplot(1, 3, 1)
plt.imshow(rgb_img, cmap='gray')
plt.title('Original Image')

# 椒盐噪声
plt.subplot(1, 3, 2)
plt.imshow(sp_noise_img, cmap='gray')
plt.title('S&P Noise')

# 高斯噪声
plt.subplot(1, 3, 3)
plt.imshow(gaus_noise_img, cmap='gray')
plt.title('Gus Noise')

plt.tight_layout()
plt.savefig('output_images/noise_comparison.jpg', dpi=300)
plt.show()
```

<img width="1170" height="395" alt="55806584c4b32d80d2c908185d122b8" src="https://github.com/user-attachments/assets/97208ee3-7140-4707-955a-9a1d9d9de0fc" />

### 2.4 图像滤波
将图像认为产生噪声后，用 OpenCV 的三个 API 滤波方式进行对比，分别对椒盐滤波和高斯滤波使用【均值滤波】，【中值滤波】，【高斯滤波】，对比每个最适合的滤波方式。
```
# 均值滤波
mean_sp = cv2.blur(sp_noise_img, (5, 5))
mean_gus = cv2.blur(gaus_noise_img, (5, 5))

# 中值滤波（需要 uint8）
mid_sp = cv2.medianBlur((sp_noise_img * 255).astype(np.uint8), 5)
mid_gus = cv2.medianBlur((gaus_noise_img * 255).astype(np.uint8), 5)

# 高斯滤波
gauss_sp = cv2.GaussianBlur((sp_noise_img * 255).astype(np.uint8), (5, 5), 0)
gauss_gus = cv2.GaussianBlur((gaus_noise_img * 255).astype(np.uint8), (5, 5), 0)

# 图像显示
plt.figure(figsize=(13, 9))

# 第 1 行：椒盐噪声
plt.subplot(2, 3, 1)
plt.imshow(mean_sp)
plt.title("S&P noise with Mean Filter")

plt.subplot(2, 3, 2)
plt.imshow(mid_sp)
plt.title("S&P Noise with Median Filter")

plt.subplot(2, 3, 3)
plt.imshow(gauss_sp)
plt.title("S&P Noise with Gaussian Filter")

# 第 2 行：高斯噪声
plt.subplot(2, 3, 4)
plt.imshow(mean_gus)
plt.title("Gaussian noise with Mean Filter")

plt.subplot(2, 3, 5)
plt.imshow(mid_gus)
plt.title("Gaussian noise with Median Filter")

plt.subplot(2, 3, 6)
plt.imshow(gauss_gus)
plt.title("Gaussian noise with Gaussian Filter")

plt.tight_layout()
plt.savefig('result/filter_results_2x3.jpg', dpi=300)
plt.show()
```

<img width="1226" height="705" alt="476d4817c1da1f225295c55dbcd8ea6" src="https://github.com/user-attachments/assets/7363a9ef-5bfe-4024-b262-ce4e86e7abe0" />

### 2.5 手动实现一个滤波方式（中值滤波）
```
# ==================== 完整独立可运行版本 ====================
import cv2
import numpy as np
from skimage.util import random_noise
from matplotlib import pyplot as plt
import os

# 1. 读取图片（请确保文件名正确，如果图片名字不一样请修改这里）
img = cv2.imread('p1.jpg')  # 如果图片叫别的名字，改这里

if img is None:
    print("❌ 错误：找不到图片 p1.jpg，请检查文件名和路径！")
else:
    print("✅ 图片读取成功，尺寸：", img.shape)

    # 2. 转换为 RGB 并缩小尺寸（手动滤波很慢，缩小后几秒就能跑完）
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_small = cv2.resize(rgb_img, (200, 200))  # 缩小到 200x200
    print("✅ 图片已缩小到 200x200，用于加速演示")

    # 3. 添加椒盐噪声
    sp_noise_img = random_noise(rgb_small, mode='s&p', amount=0.4)
    print("✅ 椒盐噪声添加完成")

    # 4. 定义手动中值滤波函数
    def manual_median_filter_color(image, kernel_size=5):
        """手动实现彩色图像的中值滤波"""
        pad = kernel_size // 2
        filtered_img = np.zeros_like(image)
        for c in range(3):  # 遍历 R,G,B 三个通道
            channel = image[:, :, c]
            padded_channel = np.pad(channel, pad_width=pad, mode='edge')
            for i in range(channel.shape[0]):
                for j in range(channel.shape[1]):
                    region = padded_channel[i:i + kernel_size, j:j + kernel_size]
                    filtered_img[i, j, c] = np.median(region)
        return filtered_img

    # 5. 执行手动中值滤波（注意：转换数据类型为 uint8）
    print("⏳ 正在执行手动中值滤波，请稍候...")
    manual_mid = manual_median_filter_color((sp_noise_img * 255).astype(np.uint8), kernel_size=5)
    print("✅ 手动中值滤波完成！")

    # 6. 显示对比结果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(sp_noise_img)
    plt.title("S&P Noise (200x200)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(manual_mid)
    plt.title("Manual Median Filter")
    plt.axis('off')

    plt.tight_layout()

    # 7. 保存结果（自动创建 result 文件夹，防止报错）
    if not os.path.exists('result'):
        os.makedirs('result')
        print("📁 已自动创建 result 文件夹")

    plt.savefig('result/manual_median.jpg', dpi=300)
    print("💾 图片已保存到 result/manual_median.jpg")

    plt.show()
    print("🎉 全部完成！")
```

<img width="1191" height="616" alt="a0c95781c4120613540c0f43ebae975" src="https://github.com/user-attachments/assets/a8b9f939-f518-48f9-aeff-b665c662d5f9" />

## 三、实验结果分析
# 1. 原始图像与颜色空间转换
通过 OpenCV 读取图像并显示，可以清楚看到原始彩色图像的细节。

将 BGR 格式转换为 RGB 格式后，颜色显示更符合人眼认知，红、绿、蓝通道正确对应。

灰度图显示则突出图像的亮度信息，有助于后续图像处理分析。

# 2. 噪声添加效果
椒盐噪声（s&p noise）：在图像中随机出现黑白点，使图像部分区域明显破坏。高斯噪声（Gaussian noise）：整个图像亮度轻微抖动，更加均匀，整体细节略模糊。对比实验显示，椒盐噪声的局部破坏更明显，高斯噪声则影响全局视觉效果。

# 3. 滤波去噪效果
均值滤波（Mean Filter）：对高斯噪声的去除效果较好，能够平滑整幅图像，但对椒盐噪声中的尖锐黑白点去除不彻底，且容易造成边缘模糊。

中值滤波（cv2.medianBlur）：对椒盐噪声的去除效果显著，能有效保留图像边缘和细节；对高斯噪声也有一定的抑制作用，但整体去噪效果略逊于均值滤波。

高斯滤波（cv2.GaussianBlur）：对高斯噪声的去除效果最佳，平滑自然，且保留了一定边缘信息；但对椒盐噪声的处理能力有限，因为其噪声是突变的而非连续分布。

手动实现中值滤波（彩色）：效果与 OpenCV 自带中值滤波基本一致，能有效去除彩色图像中的椒盐噪声，并保持颜色真实和边缘清晰。实验验证了对彩色图像应分别对 R、G、B 三通道独立滤波的合理性。通过对比显示，手动中值滤波在保持彩色信息、边缘清晰度方面表现良好，说明对彩色图像处理时，需要对每个通道分别滤波。

# 总体分析
不同类型的噪声适合采用不同的滤波方法进行去噪处理：

椒盐噪声（Salt & Pepper Noise）：由于其属于突变型噪声，中值滤波（Median Filter）能有效去除孤立的黑白噪点，同时较好地保留图像边缘细节。

高斯噪声（Gaussian Noise）：属于连续型噪声，适合采用均值滤波（Mean Filter）或高斯滤波（Gaussian Filter）进行平滑处理，能够在抑制噪声的同时保持较自然的视觉效果。

手动实现的彩色中值滤波不仅复现了 OpenCV 中的滤波效果，还加深了对滤波原理的理解。在实现过程中，通过分别对 R、G、B 三个通道独立处理，能够灵活调整滤波核大小和算法逻辑，为后续针对不同噪声类型的自定义滤波与优化提供了良好的基础。

## 四、实验小结
本实验完成了彩色图像的读取、BGR → RGB 转换以及灰度图显示，并在此基础上为图像添加了椒盐噪声与高斯噪声。

通过分别采用均值滤波、中值滤波以及手动实现的中值滤波对噪声图像进行去噪处理，对比分析了不同滤波方法的效果。

实验结果表明：

中值滤波对椒盐噪声的去除效果最为显著，能够有效保留图像边缘和细节；

均值滤波更适用于高斯噪声的平滑去除；

手动实现的中值滤波在处理彩色图像时同样表现良好，既能有效抑制噪声，又能保持图像的真实色彩与结构信息。

通过本次实验，进一步加深了对图像噪声类型与滤波原理的理解，掌握了手动实现彩色图像中值滤波的基本方法，为后续图像去噪与滤波算法的改进与优化奠定了基础。
