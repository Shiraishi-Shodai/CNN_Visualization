
# VGG16を実装する(カーネルサイズがすべて3 × 3のバージョン)
<img src="public\img\archtecher.png" alt="" height="600" />

## パラメータ数(重みをもつレイヤーが16層あるからVGG16, MaxPoolingやReLU, Dropoutはカウントしない)
**合計パラメータ数 : 134,301,514**  
他にも各レイヤーで逆伝搬のために保持する特徴マップが存在するため、
パラメータは膨大になる。(むしろ各レイヤー内部で保持する行列の方が巨大)

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


## 設計方針

### モデルの読み込み
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