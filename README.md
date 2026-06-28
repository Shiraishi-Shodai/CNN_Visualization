
# VGG16を実装する
---
<img src="public\img\archtecher.png" alt="" height="600" />


## 参考元論文
---
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)  


## 参考書籍
---
- [ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装](https://www.oreilly.co.jp/books/9784873117584/)
- [ゼロから作るDeep Learning ❷ ―自然言語処理編](https://www.oreilly.co.jp/books/9784873118369/)
- [ゼロから作るDeep Learning ❸ ―フレームワーク編](https://www.oreilly.co.jp/books/9784873119069/)
- [これならわかる深層学習入門](https://www.kspub.co.jp/book/detail/1538283.html)

## Convolution
---
- kernel : (out_channels, in_channel, 3, 3)
- bias : (out_channels)
- stride : 1
- padding : 1

## MaxPooling
---
- pool_w : 2
- pool_h : 2
- stride : 2
- padding : 0
---

## パラメータ数
>重みをもつレイヤーが16層あるからVGG16  
>MaxPoolingやReLU, Dropoutはカウントしない  
>合計パラメータ数 : 134,301,514   
>他にも各レイヤーで逆伝搬のために保持する特徴マップが存在するため、  
>メモリ上に持つデータは膨大になる。(むしろ各レイヤー内部で保持する行列の方が巨大)  

| 層 | パラメータ | パラメータ数
| ---- | ---- | ---- |
| Conv3-64 | (64, 3, 3, 3) + (64, ) | 1,792 |
| Conv3-64 | (64, 64, 3, 3) + (64, ) | 36,928 |
| MaxPool | (0, )  | 0 |
| Conv3-128 | (128, 64, 3, 3) + (128, ) | 73,856 |
| Conv3-128 | (128, 128, 3, 3) + (128, ) | 147,584 |
| MaxPool | (0, )  | 0 |
| Conv3-256 | (256, 128, 3, 3) + (256, ) | 295,168 |
| Conv3-256 | (256, 256, 3, 3) + (256, ) | 590,080 |
| Conv3-256 | (256, 256, 3, 3) + (256, ) | 590,080 |
| MaxPool | (0, )  | 0 |
| Conv3-512 | (512, 256, 3, 3) + (512, ) | 1,180,160 |
| Conv3-512 | (512, 512, 3, 3) + (512, ) | 2,359,808 |
| Conv3-512 | (512, 512, 3, 3) + (512, ) | 2,359,808 |
| MaxPool | (0, )  | 0 |
| Conv3-512 | (512, 512, 3, 3) + (512, ) | 2,359,808 |
| Conv3-512 | (512, 512, 3, 3) + (512, ) | 2,359,808 |
| Conv3-512 | (512, 512, 3, 3) + (512, ) | 2,359,808 |
| MaxPool | (0, ) | 0 |
| Flatten | (0, ) | 0 |
| Dense | (25,088, 4,096) + (4,096) | 102,764,544 |
| ReLU | (0, ) | 0 |
| Dropout | (0, ) | 0 |
| Dense | (4096, 4,096) + (4,096) | 16,781,312 |
| ReLU | (0, ) | 0 |
| Dropout | (0, ) | 0 |
| Dense | (4096, 10) + (10) | 40,970 |


## AIモデル構築設計

```
YAML
   │
   ▼
ConfigLoad
   │
   ▼
DataLoader
   │
   ▼
ModelBuilder
   │
   ▼
InitializerRegistry
   │
   ▼
LayerRegistry
   │
   ▼
List[Layer] を生成
   │
   ▼
Sequential
   │
   ▼
Model
   │
   ▼
Criterion
   │
   ▼
Optimizer
   │
   ▼
Trainer
   │
   ▼
Train / Evaluate / Test
   │
   ▼
{Train : Params Update(backward → Optimizer.update(params, grads))}
   │
   ▼
Score / Loss
```

### モデルビルダーの役割
```
ModelBuilder
│
├── YAMLを受け取る
├── レイヤーを順番に読む
├── fan_inを管理する
├── Initializerを取得する
├── Layerを生成する
└── Modelを返す
```