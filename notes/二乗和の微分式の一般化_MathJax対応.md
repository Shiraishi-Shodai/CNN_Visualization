# 二乗和の微分式の一般化

求める式は

$$
f=\sum_{i=1}^{N}(x_i-\mu)^2
$$

ただし

$$
\mu=\frac1N\sum_{i=1}^{N}x_i
$$

とします。

## ステップ1：微分したい変数を決める

$x_k$ について微分します。

$$
\frac{\partial f}{\partial x_k}
=
\sum_{i=1}^{N}
\frac{\partial}{\partial x_k}(x_i-\mu)^2
$$

## ステップ2：各項を微分する

チェーンルールより

$$
\frac{\partial}{\partial x_k}(x_i-\mu)^2
=
2(x_i-\mu)
\frac{\partial(x_i-\mu)}{\partial x_k}
$$

### $i=k$ のとき

$$
\frac{\partial(x_k-\mu)}{\partial x_k}
=
1-\frac1N
=
\frac{N-1}{N}
$$

したがって

$$
2(x_k-\mu)\frac{N-1}{N}
$$

### $i\neq k$ のとき

$$
\frac{\partial(x_i-\mu)}{\partial x_k}
=
-\frac1N
$$

したがって

$$
-\frac2N(x_i-\mu)
$$

## ステップ3：全部足す

$$
\frac{\partial f}{\partial x_k}
=
\frac{2(N-1)}{N}(x_k-\mu)
-\frac2N
\sum_{i\neq k}(x_i-\mu)
$$

## ステップ4：平均との差の和を利用する

平均との差の和は

$$
\sum_{i=1}^{N}(x_i-\mu)=0
$$

なので

$$
\sum_{i\neq k}(x_i-\mu)
=
-(x_k-\mu)
$$

これを代入すると

$$
\frac{\partial f}{\partial x_k}
=
\frac{2(N-1)}{N}(x_k-\mu)
+\frac2N(x_k-\mu)
$$

係数をまとめると

$$
\frac{2(N-1)+2}{N}
=
\frac{2N}{N}
=
2
$$

したがって

$$
\boxed{
\frac{\partial}{\partial x_k}
\sum_{i=1}^{N}(x_i-\mu)^2
=
2(x_k-\mu)
}
$$

## 分散の場合

分散は

$$
\sigma^2
=
\frac1N
\sum_{i=1}^{N}(x_i-\mu)^2
$$

なので

$$
\boxed{
\frac{\partial\sigma^2}{\partial x_k}
=
\frac{2}{N}(x_k-\mu)
}
$$

## まとめ

### 二乗和

$$
\boxed{
\frac{\partial}{\partial x_k}
\sum_{i=1}^{N}(x_i-\mu)^2
=
2(x_k-\mu)
}
$$

### 分散

$$
\boxed{
\frac{\partial\sigma^2}{\partial x_k}
=
\frac{2}{N}(x_k-\mu)
}
$$

この2つはBatch Normalizationの逆伝播を導出する際の重要な公式です。
