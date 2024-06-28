视觉大模型Llava微调效果报告
本报告主要介绍微调视觉大模型Llava-v1.5-7B模型的效果和方法，使用的数据为裕彤项目的苹果盒子数据约10张，使用的算力资源为公司服务器RTX 4090-24G * 4。本报告中主要分为以下四个部分：Llava的简单介绍及选择原因，Llava微调前后效果对比，Llava微调的主要步骤和注意事项，Llava微调问题点与总结。
Llava Github: https://github.com/haotian-liu/LLaVA 
Llava Demo on Huggingface: https://huggingface.co/spaces/badayvedat/LLaVA  
Llava paper links: https://arxiv.org/pdf/2310.03744v2 




一．Llava微调前后效果对比
由于微调步骤需要注意的点较多，先将按照第三点中的设置对llava进行微调后的效果展示如下，所有图片是未经过训练的验证图片。所有验证图片的prompt与训练集数据略有不同，统一如下：

而用于微调的训练集数据的prompt较为简单统一，如下图所示：

训练集图片如下图所示，验证集的图片和训练集的相比分布还是不太一样：

下面的一组数据，分别是OK品和NG品的验证集照片，在训练集中，如下图所示，仅有一张图片与这对图片的面相似，且缺陷不同的NG图片。但微调后的Llava仍然可以正确分别ok和ng品，由此可见微调后的 Llava仍然具有一定的泛化能力。



原模型输出：
The box in the image appears to be in good condition, with no visible defects. It is a white box with a label on it, and it is placed on a black background. The box is large and rectangular, and it seems to be well-constructed and suitable for its intended purpose.
微调后输出:
There are some defects on the box. There is a small mark on the top left corner of the box and a small mark on the top right corner of the box. These defects are located on the top edge of the box.



其他的结果：
原模型输出：
The box in the image appears to be in good condition, with no visible defects or damage. It is a white box, possibly made of cardboard, and it is sitting on a table. The box's contents are not visible in the image, but it seems to be a well-maintained and undamaged item.
微调后输出：
The box in the image is an OK product. There are no defects detected.




原模型输出：
Based on the image, it is not possible to determine if the box is an OK product or not. However, there is a small defect visible in the box. A small piece of the box is missing, which could be a concern for the user. This missing piece might affect the functionality or aesthetics of the box, depending on its purpose.
微调后输出：
No defects were found on the box.

原模型输出：
The box in the image appears to be in good condition, with no visible defects. The box is white and features an Apple logo, indicating that it is an Apple product. The logo is located on the top left corner of the box, and the overall appearance of the box suggests that it is in good condition and ready for use.
微调后输出：
There are some defects on the box. There are two small black spots on the box. One spot is located on the top left corner and the other spot is located on the top right corner.


由上可知，微调后的模型对于OK/NG判别有一定的效果，对于训练集中没有的样式也可以进行识别，但是由于训练集没有较为详细的对缺陷的描述，微调后llava对缺陷的描述和识别还有一定欠缺，这点可以在后面增加训练的对话对去尝试改善。
二．Llava微调的主要步骤和注意事项
1.环境安装
大致和Llava的github：https://github.com/haotian-liu/LLaVA 上的一致，这里只复制代码没有其他太多关注的地方。如果遇到连接超时的错误挂个vpn或者换个镜像可以解决。
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
git pull
pip install -e .
# if you see some import errors when you upgrade,# please try running the command below (without #)# pip install flash-attn --no-build-isolation --no-cache-dir
2.预训练模型下载
这里选择的是开源里最小的模型llava-v1.5-7b模型进行微调。
# 如果不能访问Hugging Face，可以执行下面这一行设置使用hf-mirror镜像站下载 HF_ENDPOINT=https://hf-mirror.com# export HF_ENDPOINT=https://hf-mirror.com
# 下载 llava-v1.5-7b 模型权重
huggingface-cli download "liuhaotian/llava-v1.5-7b" --local-dir "./checkpoints/llava-v1.5-7b"
# 下载 clip-vit-large-patch14-336 模型权重
huggingface-cli download "openai/clip-vit-large-patch14-336" --local-dir "./checkpoints/clip-vit-large-patch14-336"

注意事项：出现连接错误的话挂个vpn或者转个镜像站，如果出现下载超时之类的报错，可以多下载几次。（huggingface连接不是很稳定）

3.准备训练数据
如图生成一个json文件，内容如下图所示，可以有多个对话对，注意路径和json文件存放位置的关系。该微调是将数据和json文件都放在 ‘./playground/data/’下面。这次用的微调训练数据集大概有10张，6张ng, 4张OK。在data里面有我用于生成对话对的脚本data.py，可以根据分类文件夹生成对应的图文对。

4.模型微调
官方提供了几个shell脚本进行微调，由于我们的显卡资源有限，推荐使用lora微调来进行llava微调。根据实测，如果使用全局微调，我们的内存和显卡都很吃紧，无法开始训练。因此推荐使用’./scripts/finetune_lora.sh’进行微调，对应选择的超参如下图所示：其中注意的是，每张卡的 batch要设置为1才不会造成显存溢出。

另外要注意的是，框架会自动选择所有的gpu进行训练，如果需要指定gpu进行训练，不能通过在控制台输入CUDA_VISIABLE_DEVICES=1这种方式，需要和上图一样，在deepspeed后面增加 --include locahost: 指定显卡号，否则会报错无法开始训练。
5.开始训练
运行shell脚本就可以开始训练了，训练开始前会问是否需要可视化wandb,直接选3跳过就可以。另外如果遇到报被kill但是没有说是显存溢出的情况，大概率是因为内存不够用，可以对应调整一下各个超参数。
训练完成后模型会保存在‘./checkpoints/llava-v1.5-7b-lora’里。
6.模型融合
通过lora微调的模型需要与原模型进行融合，具体原因可以查阅大模型lora微调。在命令行运行项目脚本：
python scripts/merge_lora_weights.py --model-path './checkpoints/llava-v1.5-7b-lora' --model-base './checkpoints/llava-v1.5-7b' --save-model-path './checkpoints/llava-v1.5-7b-merged'

7.运行模型
运行代码如下，分别对比原模型和微调模型的运行效果
运行代码run_llava.py
