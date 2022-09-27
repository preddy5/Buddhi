# [Buddhi](https://en.wikipedia.org/wiki/Buddhi)(bddi) Activation for training accurate Spiking Networks
*Accurate, Efficient, Shallow and Spiking*

Author \- [Pradyumna Reddy](https://preddy5.github.io/)

In science we have taken inspiration from nature many times to solve problems. One of the big problems we are yet to comprehensive figure out is intelligence. So naturally understanding/simulating the brain in parts or whole is the most important step in our civilizational journey to understand and replicate higher-order intelligence. 

While the the current neural networks are loosely inspired by our brain, they are different in many explicitly fundamental ways. A couple of them are
* Neocortex the area of the brain that is responsible for higher-order function is relatively shallow compared to the current day "deep neural networks" that achieve impressive results.
*  Neurons in our brain communicate using spikes compared to the floating point numbers in current neural networks. 

These two properties are also interesting to us from a computation efficiency perspective. First, shallow networks are more parallel compared to deep networks since each layer in a feed forward step is conditioned on fewer previous layers. Second, the most fundamental operation in floating point neural networks or ANN's is matrix multiplication of floating point numbers. However this does not have to be the case when layers communicate in spikes i.e 0's and 1's. Since 0 and 1 are special numbers, 0 is multiplicative null and 1 is multiplicative identity. So, in a network that communicates in spikes we can reformulate all matrix multiplications as just memory retrieval operations making inference extremely effective.    

Apart from these direct benefits, SNN's are also interesting because of various other reasons, a few being:
* The difficult problem of training a highly parameterized hierarchical network that communicates with discrete values. 
* Ability to explore ideas like polychronization for the problem of feature binding. 
* Interest from academic heavy weights like Prof. Hinton.

Despite their advantages and biological plausibility SNN research could not gather similar amount of interest as ANN's from the larger scientific community. A major reason for this is that leading SNN models biologically plausible or otherwise are less accurate than their ANN counterparts. In this repository I present BITnets shallow SNN models trained using Buddhi activation that outperforms many deep ANN's on Cifar10, Cifar100 and Tiny-Imagenet classification task. Interestingly the networks presented in this repository are shallower than AlexNet and same depth(loosely) as the neocortex.


# Results
From now I will refer to the activation function as "bddi" and the networks trained with bddi activation as BITnet(Binary InTeraction network). 

In the table below, I compare the classification accuracy of famous ANN's and networks trained with bddi activation. I present the classification accuracy results on Cifar10, Cifar100 and Tiny-imagenet. BITnets ourperform the ANN networks by a margin in Cifar10 while being competitive in Cifar100 classification performance. The results also show that networks trained with bddi activation are able to achieve this while being shallow and in some cases parameter efficient. To the best of my knowledge this is the first example of shallow spiking networks that outperform deep ANN's. In the table below the integer in each BITnet model name refers to the number of layers in the network, more details about the architecture of the network are available at the bottom of the section.  

Baseline - Baseline has the same architecture as BITnet6 but with ReLU activation function and a softmax layer in the end.*

() - In the brackets are the validation accuracy's of architecturally closest networks as reported in the original paper or in a follow-up paper by the same authors.

| No. |     Model    | Cifar10 ValAcc    | Cifar100 ValAcc        | No. Params Cifar 10/100 |   
|:---:|:-------------|------------------:|-----------------------:|------------------------:|
| 0   | Baseline    |   93.13%          |        70.26%                |   25.7 / 28.7 M |
| 1   | Diet-SNN VGG16     |   92.70%          |         69.67%         |   33.6 / 34.0 M |
| 2   | SNN Sengupta et al(2019) VGG-16    |   91.5%          |         62.7%         |   - / - M |
| 3   | SNN STBP_tdBN ResNET-19    |   93.16%          |         71.69%         |   - / - M |
| 4   | SNN IDE-FSNN CIFARNET-F    |   92.69%          |         73.29%         |   - / - M |
| 5   | SNN Dspike_tdBN ResNet-18(Code unavailable)    |   94.32%          |         74.34%         |   - / - M |
| 6   | VGG11_BN     |   92.39%          |         68.64%         |   28.6 / 28.50 M | 
| 7   | VGG13_BN     |   94.22%          |         72.00%         |   28.3 / 28.7 M |
| 8   | VGG16_BN     |   94.00%          |         72.93%         |   33.6 / 34.0 M |
| 9   | VGG19_BN     |   93.95%          |         72.23%         |   39.0 / 39.0 M | 
| 10   | inception_v3 |   93.74%          |         77.19          |   21.6 / 22.8 M | 
| 11   | ResNet18     |   93.07 (91.25)%  |         75.61 (68.88)% |   11.2 / 11.2 M |
| 12   | ResNet34     |   93.34 (92.49)%  |         76.76 (70.26)% |   21.3 / 21.3 M |
| 13   | ResNet50     |   93.65 (93.03)%  |         77.39 (71.54)% |   23.5 / 23.7 M |
| 14   | ResNet101    |   93.75 (93.39)%  |		   77.78 (72.59)% |   42.5 / 42.7 M |  
| 15   | BITnet6[Ours]|   94.37%         |              75.93%          |   25.7 / 28.7 M | 
| 16   | BITnet6 Max[Ours]|   94.62%         |           77.15%           |   40.1 / 43.7 M |

Among models in the table above, all the ANN's use BatchNormalization(BN) and some SNN's us Time-dependant BatchNormalization(tdBN) both assist with the network convergence.

| No. |     Model    | Tiny-Imagenet ValAcc    | 
|:---:|:-------------|------------------:|
| 0   | VGG16    |   51.9%          | 
| 1   | BITnet6[Ours]     |   59.10%          |


Adding an extra layer dramatically improves the performance on the tiny-imagenet dataset but I'll leave that to a later update. 


Architecture: [64\*fs, 64\*fs, M, 128\*fs, M, 256\*fs, 512\*fs, M, FC]

where, **fs** refers to feature scaling parameter, fs is 4 for the standard models and 5 for the max models, **integer\*fs** - represents a convolution layer with integer\*fs output channels, all convolution layers have a filter size of 3x3 and stride 1, **M** - referes to a maxpool layer with kernel size 2 and stride 2 and **FC** - represents a fully connected layer.
From architecture details above, notice that BITnets have fewer maxpool operations than ANN's. Emperically using a similar sub-sampling strategy as ANN's is not ideal. Further research should be done to find the optimal architectural recipe of SNN with bddi activation.

Spiking networks which are based on integrate and fire models arrive at an output after running the same model multiple times. Let us refer to the number of times each model is run as "steps". The outputs of all the steps are aggregated to form the final output. We train different versions of the model with different number of steps because of VRAM limitations. The standard version and max versions of the model are trained with 50 steps. Number of steps during training effect the accuracy of the model. During training the standard model increasing the steps from 36 to 50 improved the models performance by 0.38% and 1.79% on Cifar10 and Cifar100 respectively. Training models with more than 50 steps would be an interesting experiment, I am not prioritizing this because of hardware requirement. Interestingly increasing the steps during inference also increases the validation accuracy. The Cifar100, Tiny-imagenet accuracies reported above are with 200 steps rather than the original steps value during training.

To reproduce the results from the above table first clone this repo. Then download the network weights from [here](https://drive.google.com/drive/folders/1YirxrcvKxI_q0i9PIMKOHtu5ZayUAuR3?usp=sharing) and place them in the root directory.  Make appropriate changes to the commands below and run them. 

    CUDA_VISIBLE_DEVICES=0  python scripts/trainer_pl.py --b 32 --model BITnet6 --dataset cifar10 --n_gpu 1 --checkpoint_dir ./checkpoints/cifar10/ --fs 4 --steps 50
    # BITnet6 Cifar10 Expected output: 0.9437000155448914
    
    CUDA_VISIBLE_DEVICES=0  python scripts/trainer_pl.py --b 32 --model BITnet6 --dataset cifar10 --n_gpu 1 --checkpoint_dir ./checkpoints/cifar10_max/ --fs 5 --steps 50
    # BITnet6_max Cifar10 Expected output: 0.9462000131607056
    
    CUDA_VISIBLE_DEVICES=0  python scripts/trainer_pl.py --b 32 --model BITnet6 --dataset cifar100 --n_gpu 1 --checkpoint_dir ./checkpoints/cifar100/ --fs 4 --steps 200
    # BITnet6 Cifar100 Expected output: 0.7592999935150146
    
    CUDA_VISIBLE_DEVICES=0  python scripts/trainer_pl.py --b 32 --model BITnet6 --dataset cifar100 --n_gpu 1 --checkpoint_dir ./checkpoints/cifar100_max/ --fs 5 --steps 210
    # BITnet6_max Cifar100 Expected output: 0.7714999914169312
    
    # Download Tiny-Imagenet to '../data/tiny-imagenet-200/' folder or
    # Change folder name in ln183 in src/io.py to Tiny-Imagenet location
    CUDA_VISIBLE_DEVICES=0  python scripts/trainer_pl.py --b 16 --model BITnet6 --dataset tiny_imagenet --n_gpu 1 --checkpoint_dir ./checkpoints/tiny/ --fs 4 --steps 200
    # BITnet6 Tiny-Imagenet Expected output: 0.5910000205039978

    
 **Sparsity analysis**: In the introduction I discuss how spikes can make multiplications in neural networks efficient by reformulating multiplication as memory retrieval. The gain in efficiency can be measured as a factor number of zeros i.e sparsity of each layer. In the table below I present, the avg percentage of zeros in each layer output.
 
| Dataset - Model | Layer1 | Layer2    | Layer3 | Layer4 | Layer5 | Layer6 |   
|:---------------:|:-------|----------:|-------:|-------:|-------:|-------:|
| Cifar10 - BITnet6 | 90.35% | 92.38%    | 96.48% | 94.11% | 98.56% | 90.78% | 
| Cifar10 - BITnet6 Max | 93.74% | 94.94%    | 97.72% | 94.31% | 98.89% | 90.77% | 
| Cifar100 - BITnet6 | 91.75% | 90.29%    | 96.32% | 88.35% | 98.39% | 99.33% | 
| Cifar100 - BITnet6 Max | 92.74% | 92.69%    | 97.14% | 90.39% | 98.98% | 99.34% | 
| TinyImagenet - BITnet6 | 95.34% | 91.26%    | 96.77% | 87.89% | 98.76% | 99.71% | 
    

# Future work and Conclusion
Adopting autograd for bddi activation to make training both compute and memory efficient is an important next step. Current deep-learning accelerators are majorly optimized for matrix multiplication, however these optimizations do take advantage of bddi activation and its properties making inference and training of SNN's on current frameworks and hardware computationally slower. Optimizing matrix multiplication for spikes instead of floating point numbers should make inference of BITnets faster. During training deep learning frameworks remember every layer output of all the steps making training of BITnets memory intensive, we can exploit the recurrent nature of BITnets to make training using backpropogation more memory efficient. These improvements will make training on larger resolution images for more steps tractable. 
As bddi activation can naturally be extended to sequential data, exploring architectures for sequential data like text and video would be interesting. 
Spiking communication also naturally lends itself to boolean logic operations enabling us to explore neuro-symbolic ideas to address the shortcomings of ANN's. Infact the maxpooling operation in the BITnet models can be thought of as a logical-or operation. 
<!---
Another important next step would be to explore appropriate architectures for Spiking networks to outperform ANN architectures like ResNext, and VIT. 

-->


This is a proof of concept of what is plausible with bddi activation and shallow spiking networks and we are only getting started. Interest from the community is a big part of how resources are allocated in research, so please consider sharing this repository or dropping me an email, if you have found this stream of research interesting. If you have ideas on how I can improve the project I would love to hear them you can reach me at [email](mailto:reddy.pradyumna5@gmail.com) or [@me](https://twitter.com/preddy_v1) with you suggestions. 
