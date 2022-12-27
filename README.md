# Image Fusion
This repository contains code reproductions of existing state-of-the-art image fusion models, our work is still in progress.    





## 多模图像融合(Multi-Modal Image Fusion)
### 红外和可见光图像融合(Infrared and visible image fusion)
<table>
    <thead>
      <tr>
        <th>方法</th>
        <th>标题</th>
        <th>论文</th>
        <th>代码</th>
        <th>发表期刊或会议</th>
        <th>基础框架</th>
        <th>监督范式</th>
        <th>发表年份</th>
        <th>引用量</th>
        <th>复现情况</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>DenseFuse</td>
        <td>DenseFuse: A Fusion Approach to Infrared and Visible Images</td>
        <td><a href="https://ieeexplore.ieee.org/abstract/document/8580578/">Paper</a></td>
        <td><a href="https://github.com/hli1221/imagefusion_densefuse">Code</a></td>
        <td>TIP</td>
        <td>AE</td>
        <td>自监督</td>
        <td>2019</td>
        <td>569</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>FusionGAN</td>
        <td>FusionGAN: A generative adversarial network for infrared and&nbsp;&nbsp;&nbsp;visible image fusion</td>
        <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S1566253518301143%3Fvia%253Dihub">Paper</a></td>
        <td><a href="https://github.com/jiayi-ma/FusionGAN">Code</a></td>
        <td>InfFus</td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2019</td>
        <td>711</td>
        <td> </td>
      </tr>
      <tr>
        <td>DDcGAN</td>
        <td>Learning a Generative Model for Fusing Infrared and Visible&nbsp;&nbsp;&nbsp;Images via Conditional Generative Adversarial Network with Dual&nbsp;&nbsp;&nbsp;Discriminators</td>
        <td><a href="https://www.ijcai.org/proceedings/2019/0549.pdf">Paper</a></td>
        <td><a href="https://github.com/hanna-xu/DDcGAN">Code</a></td>
        <td>IJCAI</td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2019</td>
        <td>51</td>
        <td> </td>
      </tr>
      <tr>
        <td>NestFuse</td>
        <td>NestFuse: An Infrared and Visible Image Fusion Architecture&nbsp;&nbsp;&nbsp;Based on Nest Connection and Spatial/Channel Attention Models</td>
        <td><a href="https://ieeexplore.ieee.org/document/9127964/">Paper</a></td>
        <td><a href="https://github.com/hli1221/imagefusion-nestfuse">Code</a></td>
        <td>TIM</td>
        <td>AE</td>
        <td>自监督</td>
        <td>2020</td>
        <td>125</td>
        <td> </td>
      </tr>
      <tr>
        <td>DDcGAN</td>
        <td>DDcGAN: A dual-discriminator conditional generative&nbsp;&nbsp;&nbsp;adversarial network for multi-resolution image fusion</td>
        <td><a href="https://ieeexplore.ieee.org/abstract/document/9031751/">Paper</a></td>
        <td><a href="https://github.com/hanna-xu/DDcGAN">Code</a></td>
        <td>TIP </td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2020</td>
        <td>307</td>
        <td> </td>
      </tr>
      <tr>
        <td>DIDFuse</td>
        <td>DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion</td>
        <td><a href="https://arxiv.org/abs/2003.09210">Paper</a></td>
        <td><a href="https://github.com/Zhaozixiang1228/IVIF-DIDFuse">Code</a></td>
        <td>IJCAI </td>
        <td>AE</td>
        <td>自监督</td>
        <td>2020</td>
        <td>40</td>
        <td> </td>
      </tr>
      <tr>
        <td>RFN-Nest</td>
        <td>RFN-Nest: An end-to-end residual fusion network for infrared&nbsp;&nbsp;&nbsp;and visible images</td>
        <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S1566253521000440%3Fvia%253Dihub">Paper</a></td>
        <td><a href="https://github.com/hli1221/imagefusion-rfn-nest">Code</a></td>
        <td>InfFus</td>
        <td>AE</td>
        <td>自监督</td>
        <td>2021</td>
        <td>112</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>CSF</td>
        <td>Classification Saliency-Based Rule for Visible and Infrared&nbsp;&nbsp;&nbsp;Image Fusion</td>
        <td><a href="https://ieeexplore.ieee.org/abstract/document/9502544">Paper</a></td>
        <td><a href="https://github.com/hanna-xu/CSF">Code</a></td>
        <td>TCI</td>
        <td>AE</td>
        <td>自监督</td>
        <td>2021</td>
        <td>19</td>
        <td> </td>
      </tr>
      <tr>
        <td>DRF</td>
        <td>DRF: Disentangled Representation for Visible and Infrared&nbsp;&nbsp;&nbsp;Image Fusion</td>
        <td><a href="https://ieeexplore.ieee.org/document/9345717/">Paper</a></td>
        <td><a href="https://github.com/hanna-xu/DRF">Code</a></td>
        <td>TIM</td>
        <td>AE</td>
        <td>自监督</td>
        <td>2021</td>
        <td>41</td>
        <td> </td>
      </tr>
      <tr>
        <td>SEDRFuse</td>
        <td>SEDRFuse: A Symmetric Encoder–Decoder With Residual Block&nbsp;&nbsp;&nbsp;Network for Infrared and Visible Image Fusion</td>
        <td><a href="https://ieeexplore.ieee.org/document/9187663">Paper</a></td>
        <td><a href="https://github.com/jianlihua123/SEDRFuse">Code</a></td>
        <td>TIM</td>
        <td>AE</td>
        <td>自监督</td>
        <td>2021</td>
        <td>43</td>
        <td> </td>
      </tr>
      <tr>
        <td>MFEIF</td>
        <td>Learning a Deep Multi-Scale Feature Ensemble and an&nbsp;&nbsp;&nbsp;Edge-Attention Guidance for Image Fusion</td>
        <td><a href="https://ieeexplore.ieee.org/document/9349250">Paper</a></td>
        <td></td>
        <td>TCSVT</td>
        <td>AE</td>
        <td>自监督</td>
        <td>2021</td>
        <td>38</td>
        <td> </td>
      </tr>
      <tr>
        <td>Meta-Learning</td>
        <td>Different Input Resolutions and Arbitrary Output Resolution: A&nbsp;&nbsp;&nbsp;Meta Learning-Based Deep Framework for Infrared and Visible Image Fusion</td>
        <td><a href="https://ieeexplore.ieee.org/document/9394791/">Paper</a></td>
        <td></td>
        <td>TIP</td>
        <td>CNN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>15</td>
        <td> </td>
      </tr>
      <tr>
        <td>RXDNFuse</td>
        <td>RXDNFuse: A aggregated residual dense network for infrared and&nbsp;&nbsp;&nbsp;visible image fusion</td>
        <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S1566253520304152%3Fvia%253Dihub">Paper</a></td>
        <td><a href="https://github.com/JinyuanLiu-CV/MFEIF">Code</a></td>
        <td>InfFus</td>
        <td>CNN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>27</td>
        <td> </td>
      </tr>
      <tr>
        <td>STDFusionNet</td>
        <td>STDFusionNet: An Infrared and Visible Image Fusion Network&nbsp;&nbsp;&nbsp;Based on Salient Target Detection</td>
        <td><a href="https://ieeexplore.ieee.org/document/9416507">Paper</a></td>
        <td><a href="https://github.com/Linfeng-Tang/STDFusionNet">Code</a></td>
        <td>TIM</td>
        <td>CNN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>67</td>
        <td> </td>
      </tr>
      <tr>
        <td>D2LE</td>
        <td>A Bilevel Integrated Model With Data-Driven Layer Ensemble for&nbsp;&nbsp;&nbsp;Multi-Modality Image Fusion</td>
        <td><a href="https://ieeexplore.ieee.org/document/9293146">Paper</a></td>
        <td></td>
        <td>TIP</td>
        <td>CNN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>33</td>
        <td> </td>
      </tr>
      <tr>
        <td>HAF</td>
        <td>Searching a Hierarchically Aggregated Fusion Architecture for&nbsp;&nbsp;&nbsp;Fast Multi-Modality Image Fusion</td>
        <td><a href="https://dl.acm.org/doi/abs/10.1145/3474085.3475299%3Fcasa_token%3DtT48gSwVWjkAAAAA%3AQaDUAB7nLzWcByiAESzOTAgFjdh5kLxS8J612shuDn3RLLIOcU1AX7AhcvYT9UlGTub1mi85Tws">Paper</a></td>
        <td><a href="https://github.com/LiuzhuForFun/Hierarchical-NAS-Image-Fusion">Code</a></td>
        <td>ACM MM</td>
        <td>CNN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>5</td>
        <td> </td>
      </tr>
      <tr>
        <td>SDDGAN</td>
        <td>Semantic-supervised Infrared and Visible Image Fusion via a&nbsp;&nbsp;&nbsp;Dual-discriminator Generative Adversarial Network</td>
        <td><a href="https://ieeexplore.ieee.org/document/9623476">Paper</a></td>
        <td><a href="https://github.com/WeiWu-WIT/SDDGAN">Code</a></td>
        <td>TMM</td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>7</td>
        <td> </td>
      </tr>
      <tr>
        <td>Detail-GAN</td>
        <td>Infrared and visible image fusion via detail preserving&nbsp;&nbsp;&nbsp;adversarial learning</td>
        <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S1566253519300314%3Fvia%253Dihub">Paper</a></td>
        <td><a href="https://github.com/jiayi-ma/ResNetFusion">Code</a></td>
        <td>InfFus</td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>206</td>
        <td> </td>
      </tr>
      <tr>
        <td>Perception-GAN</td>
        <td>Image fusion based on&nbsp;&nbsp;&nbsp;generative adversarial network consistent with perception</td>
        <td><a href="https://www.sciencedirect.com/science/article/pii/S1566253521000439%3Fcasa_token%3D-LrGkXT12IMAAAAA%3A35WH5dIx-2zWPZ3sTFL3cgjjCK_17nkc0xYVkDXL8Pp51k3DIPtaNT8NWvuKoXlIOoRkKi87AZo0">Paper</a></td>
        <td><a href="https://github.com/thfylsty/imagefusion_Perceptual_FusionGan">Code</a></td>
        <td>InfFus</td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>28</td>
        <td> </td>
      </tr>
      <tr>
        <td>GAN-FM</td>
        <td>GAN-FM: Infrared and Visible&nbsp;&nbsp;&nbsp;Image Fusion Using GAN With Full-Scale Skip Connection and Dual Markovian&nbsp;&nbsp;&nbsp;Discriminators</td>
        <td><a href="https://ieeexplore.ieee.org/document/9573457">Paper</a></td>
        <td><a href="https://github.com/yuanjiteng/GAN-FM">Code</a></td>
        <td>TCI</td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>19</td>
        <td> </td>
      </tr>
      <tr>
        <td>AttentionFGAN</td>
        <td>AttentionFGAN: Infrared and Visible Image Fusion Using&nbsp;&nbsp;&nbsp;Attention-Based Generative Adversarial Networks</td>
        <td><a href="https:////ieeexplore.ieee.org/document/9103116/">Paper</a></td>
        <td></td>
        <td>TMM</td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>71</td>
        <td> </td>
      </tr>
      <tr>
        <td>GANMcC</td>
        <td>GANMcC: A Generative&nbsp;&nbsp;&nbsp;Adversarial Network With Multiclassification Constraints for Infrared and&nbsp;&nbsp;&nbsp;Visible Image Fusion</td>
        <td><a href="https://ieeexplore.ieee.org/abstract/document/9274337">Paper</a></td>
        <td><a href="https://github.com/jiayi-ma/GANMcC">Code</a></td>
        <td>TIM</td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>109</td>
        <td> </td>
      </tr>
      <tr>
        <td>MgAN-Fuse</td>
        <td>Multigrained Attention Network for Infrared and Visible Image&nbsp;&nbsp;&nbsp;Fusion</td>
        <td><a href="https://ieeexplore.ieee.org/document/9216075">Paper</a></td>
        <td></td>
        <td>TIM</td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>30</td>
        <td> </td>
      </tr>
      <tr>
        <td>TC-GAN</td>
        <td>Infrared and Visible Image&nbsp;&nbsp;&nbsp;Fusion via Texture Conditional Generative Adversarial Network</td>
        <td><a href="https://ieeexplore.ieee.org/abstract/document/9335976">Paper</a></td>
        <td></td>
        <td>TCSVT</td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2021</td>
        <td>23</td>
        <td> </td>
      </tr>
      <tr>
        <td>AUIF</td>
        <td>Efficient and model-based infrared and visible image fusion via algorithm unrolling</td>
        <td><a href="https://ieeexplore.ieee.org/abstract/document/9416456/">Paper</a></td>
        <td><a href="https://github.com/Zhaozixiang1228/IVIF-AUIF-Net">Code</a></td>
        <td>TCSVT</td>
        <td>AE</td>
        <td>自监督</td>
        <td>2021</td>
        <td>14</td>
        <td> </td>
      </tr>
      <tr>
        <td>TarDAL</td>
        <td>Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection</td>
        <td><a href="https://arxiv.org/abs/2203.16220v1">Paper</a></td>
        <td><a href="https://github.com/JinyuanLiu-CV/TarDAL">Code</a></td>
        <td>CVPR</td>
        <td>GAN</td>
        <td>无监督</td>
        <td>2022</td>
        <td>21</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>RFNet</td>
        <td>RFNet: Unsupervised Network for Mutually Reinforcing Multi-modal Image Registration and Fusion</td>
        <td><a href="https://www.researchgate.net/publication/360238557_RFNet_Unsupervised_Network_for_Mutually_Reinforcing_Multi-modal_Image_Registration_and_Fusion">Paper</a></td>
        <td><a href="https://github.com/hanna-xu">Code</a></td>
        <td>CVPR</td>
        <td>CNN</td>
        <td>无监督</td>
        <td>2022</td>
        <td>7</td>
        <td> </td>
      </tr>
      <tr>
        <td>SeAFusion</td>
        <td>Image fusion in the loop of&nbsp;&nbsp;&nbsp;high-level vision tasks: A semantic-aware real-time infrared and visible&nbsp;&nbsp;&nbsp;image fusion network</td>
        <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S1566253521002542%3Fvia%253Dihub">Paper</a></td>
        <td><a href="https://github.com/Linfeng-Tang/SeAFusion">Code</a></td>
        <td>InfFus</td>
        <td>CNN</td>
        <td>无监督</td>
        <td>2022</td>
        <td>52</td>
        <td> </td>
      </tr>
      <tr>
        <td>PIAFusion</td>
        <td>PIAFusion: A progressive infrared and visible image fusion&nbsp;&nbsp;&nbsp;network based on illumination aware</td>
        <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S156625352200032X">Paper</a></td>
        <td><a href="https://github.com/Linfeng-Tang/PIAFusion" target="_blank" rel="noopener noreferrer">Code</a></td>        
        <td>InfFus</td>
        <td>CNN</td>
        <td>无监督</td>
        <td>2022</td>
        <td>20</td>
        <td> </td>
      </tr>      
      <tr>
        <td>UMF-CMGR</td>
        <td>Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration</td>
        <td><a href="https://arxiv.org/pdf/2205.11876.pdf">Paper</a></td>
        <td><a href="https://github.com/wdhudiekou/UMF-CMGR" target="_blank" rel="noopener noreferrer">Code</a></td>        
        <td>IJCAI</td>
        <td>CNN</td>
        <td>无监督</td>
        <td>2022</td>
        <td>11</td>
        <td> </td>
      </tr>
      <tr>
        <td>DetFusion</td>
        <td>DetFusion: A Detection-driven Infrared and Visible Image Fusion Network</td>
        <td><a href="https://dl.acm.org/doi/abs/10.1145/3503161.3547902?casa_token=YmZKEkd1zVcAAAAA:wb0HbfopS60BwSh0_QAdHHjwNQ5ZoWxduwHfBd1NzLbMr32AOPcInC4TCbZ5KM9Ly1PbVIGkmg">Paper</a></td>
        <td><a href="https://github.com/SunYM2020/DetFusion" target="_blank" rel="noopener noreferrer">Code</a></td>        
        <td>ACM MM</td>
        <td>CNN</td>
        <td>无监督</td>
        <td>2022</td>
        <td>2</td>
        <td> </td>
      </tr>
      <tr>
        <td>DIVFusion</td>
        <td>DIVFusion: Darkness-free infrared and visible image fusion</td>
        <td><a href="https://authors.elsevier.com/c/1g4EB_ZdCkUn0I">Paper</a></td>
        <td><a href="https://github.com/Xinyu-Xiang/DIVFusion" target="_blank" rel="noopener noreferrer">Code</a></td>        
        <td>InfFus</td>
        <td>CNN</td>
        <td>无监督</td>
        <td>2023</td>
        <td>1</td>
        <td> </td>
      </tr>
    </tbody>
    </table>

## Citation
我们的总结参考了该仓库(https://github.com/Linfeng-Tang/Image-Fusion)：
