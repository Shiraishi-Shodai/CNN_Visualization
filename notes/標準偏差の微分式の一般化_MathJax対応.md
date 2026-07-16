# 標準偏差の微分式の一般化

標準偏差を

$$
\sigma=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2}
$$

ただし

$$
\mu=\frac1N\sum_{i=1}^{N}x_i
$$

とします。

## ステップ1：中身を置き換える

中身を

$$
A=\frac1N\sum_{i=1}^{N}(x_i-\mu)^2
$$

と置くと、

$$
\sigma=\sqrt{A}=A^{1/2}
$$

となります。

## ステップ2：平方根を微分する

チェーンルールより

$$
\frac{\partial\sigma}{\partial x_k}
=
\frac12A^{-1/2}
\frac{\partial A}{\partial x_k}
$$

となります。

## ステップ3：分散を微分する

前に導出した結果より、

$$
\frac{\partial A}{\partial x_k}
=
\frac{2}{N}(x_k-\mu)
$$

です。

これを代入すると

$$
\frac{\partial\sigma}{\partial x_k}
=
\frac12A^{-1/2}
\cdot
\frac{2}{N}(x_k-\mu)
$$

## ステップ4：整理する

係数をまとめると

$$
\frac{\partial\sigma}{\partial x_k}
=
\frac1N
A^{-1/2}
(x_k-\mu)
$$

さらに

$$
A^{-1/2}
=
\frac1{\sqrt{A}}
$$

なので

$$
\frac{\partial\sigma}{\partial x_k}
=
\frac{x_k-\mu}{N\sqrt{A}}
$$

最後に

$$
\sqrt{A}=\sigma
$$

を代入すると

$$
\boxed{
\frac{\partial\sigma}{\partial x_k}
=
\frac{x_k-\mu}{N\sigma}
}
$$

## Batch Normalizationで用いる形

Batch Normalizationでは

$$
\sqrt{\sigma^2+\varepsilon}
$$

を用いるため、

$$
\boxed{
\frac{\partial}{\partial x_k}
\sqrt{\sigma^2+\varepsilon}
=
\frac{x_k-\mu}
{N\sqrt{\sigma^2+\varepsilon}}
}
$$

となります。

## まとめ

$$
\boxed{
\frac{\partial\sigma}{\partial x_k}
=
\frac{x_k-\mu}{N\sigma}
}
$$

これはBatch Normalizationの逆伝播で重要な公式です。
