# Official code for "Logit Mixing Training for More Reliable and Accurate Prediction"
The official code is implemented using Pytorch

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

## Warmup mode makes the training more stable, but it is not necessary.  

## Reference code  
github.com/weiaicunzai/pytorch-cifar100  
github.com/pytorch/vision  
github.com/clovaai/CutMix-PyTorch  
