# "Logit Mixing Training for More Reliable and Accurate Prediction"

> ** Logit Mixing Training for More Reliable and Accurate Prediction ** <br>
> Duhyeon Bang*, Kyungjune Baek*, Jiwoo Kim, Yunho Jeon, Jin-Hwa Kim, Jiwon Kim, Jongwuk Lee, Hyunjung Shim <br>
> (* indicates equal contribution)
> *International Joint Conference on Artificial Intelligence **IJCAI 2022***
[[Paper](https://www.ijcai.org/proceedings/2022/0390.pdf)]
> The official code is implemented using Pytorch

When a person solves the multi-choice problem, she considers not only what is the answer but also what is not the answer. Knowing what choice is not the answer and utilizing the relationships between choices, she can improve the prediction accuracy. Inspired by this human reasoning process, we pro-pose a new training strategy to fully utilize inter-class relationships, namely LogitMix. Our strategy is combined with recent data augmentation tech-niques, e.g., Mixup, Manifold Mixup, CutMix, and PuzzleMix. Then, we suggest using a mixed logit, i.e., a mixture of two logits, as an auxiliary training objective. Since the logit can preserve both positive and negative inter-class relationships, it can impose a network to learn the probability of wrong answers correctly. Our extensive experimental results on the image- and language-based tasks demonstrate that LogitMix achieves state-of-the-art performance among recent data augmentation techniques regard-ing calibration error and prediction accuracy.

## Setting requirements
>>  pip install -r requirements.txt  
>>  Need to set the dataset directory  
    The default dir is  
    $(pwd)$/Databases/tiny-imagenet-200   for tiny-imagenet dataset  
    $(pwd)$/Databases/cifar/cifar-100-python  for cifar100 dataset  
    
## Run
>> python main.py \  
--network resnet50 \  
--dataset cifar100 \  
--batch_size [default for logitmix_M: 128] \  
--mixmethod logitmix_M \  
--weights [default for logitmix_M: 1 1 1] \  
--dist [default for logitmix_M: beta] \  
--alpha [default for logitmix_M: 3] \  
--loss [loss, we recommend to use mse_mixed_logsoftmax] \  
--gpu [gpu number]  \
--dataset_dir [dataset directory]  

## Etc  
Warmup mode makes the training more stable, but it is not necessary.  

## Reference code  
github.com/weiaicunzai/pytorch-cifar100  
github.com/pytorch/vision  
github.com/clovaai/CutMix-PyTorch  
