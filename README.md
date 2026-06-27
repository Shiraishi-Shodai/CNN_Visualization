

## 設計方針

### モデルの読み込み
```
YAML
   │
   ▼
ConfigLoader
   │
   ▼
ModelBuilder
   │
   ├── Layer Registry
   ├── Initializer Registry
   └── Loss Registry
   │
   ▼
Model

    model.yaml
        │
        ▼
ModelBuilder
        │
List[Layer] を生成
        │
        ▼
Sequential
        │
forward / backward
        │
        ▼
    Optimizer
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