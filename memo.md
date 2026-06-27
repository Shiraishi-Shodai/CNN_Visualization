この設計なら、**いきなりモデルを生成しようとせず、「Builderパターン」を段階的に作る**のがおすすめです。

今後VGG16まで拡張することを考えると、依存関係は次のようになります。

```text
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
```

この依存関係を崩さないように、一つずつ実装していきます。

---

## ステップ1: YAMLを読み込めるようにする（最優先）

まずはPythonでこのYAMLを辞書として取得します。

```python
config = load_config("config/model.yaml")

print(config["model"]["layers"])
```

ここで期待する出力は

```python
[
    {
        "type": "Affine",
        "out_features": 3,
        "initializer": {
            "weight": "he",
            "bias": "zeros"
        }
    },
    {
        "type": "ReLU"
    },
    ...
]
```

この段階ではモデル生成はまだ行いません。

---

## ステップ2: Layer Registryを作る

初期化器と同じようにレイヤーも登録します。

```python
LAYER_REGISTRY = {}
```

```python
@register_layer("Affine")
class Affine:
    ...
```

```python
@register_layer("ReLU")
class ReLU:
    ...
```

すると

```python
layer_cls = get_layer("Affine")
```

でクラスを取得できます。

---

## ステップ3: ModelBuilderを書く

ここで初めて

```python
builder = ModelBuilder(config)
```

を書きます。

Builderは

```python
for layer_cfg in config["model"]["layers"]:
```

を回すだけです。

つまり

```
Affine
ReLU
Affine
ReLU
Affine
```

を順番に生成していきます。

---

## ステップ4: BuilderはAffineだけ対応する

最初から全部対応する必要はありません。

まずは

```yaml
- type: Affine
```

だけ対応します。

Builderは

```python
if layer_type == "Affine":
```

でも構いません。

動作確認できてからRegistryへ置き換えれば十分です。

---

## ステップ5: fan_inをBuilderが計算する

Affineを生成するには

```
fan_in
fan_out
```

が必要です。

YAMLには

```yaml
input_size: 10
```

しかありません。

Builderは

```
現在の出力数
        ↓
次の入力数
```

を覚えておきます。

例えば

```
input_size = 10

Affine(out=3)

↓

fan_in = 10
fan_out = 3
```

次は

```
Affine(out=4)

↓

fan_in = 3
fan_out = 4
```

さらに

```
Affine(out=5)

↓

fan_in = 4
fan_out = 5
```

このようにBuilderが管理します。

---

## ステップ6: Initializerを呼ぶ

Builderが

```python
weight_init = get_initializer("he")
```

を取得し、

```python
W = weight_init(
    shape=(fan_in, fan_out),
    fan_in=fan_in,
    fan_out=fan_out,
)
```

を実行します。

Biasも同様です。

---

## 私なら、この順番で進めます

1. ✅ YAMLを読み込む
2. ✅ `ModelBuilder` の骨組みを作る
3. ✅ `Affine` レイヤーだけ生成できるようにする
4. ✅ `ReLU` を追加する
5. ✅ `Loss` を追加する
6. ✅ Layer Registryに置き換える
7. ✅ ConvやPoolingを追加してVGG16へ拡張する

---

### 次の一歩（おすすめ）

**最初に実装するべきクラスは `ModelBuilder`** です。

このクラスの役割はシンプルで、

* YAMLを1行ずつ読む
* `fan_in` を管理する
* レイヤーを順番に生成する

だけに限定します。

これが完成すると、VGG16になってもBuilderの処理の流れはほとんど変わらず、レイヤーの種類を追加するだけで拡張できる設計になります。
