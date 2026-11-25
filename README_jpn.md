# embedfit

English: [README.md](README.md)

参照量で規格化されたスペクトルを、計測器の補正をモデルに組み込んで解析するパッケージである。

多くの分光装置は、補正した値を出力する。記録された計数から暗計数の水準を差し引き、検出器の補正 G（不感時間、効率）を施し、参照量 Phi（光子数、I0）で割った値である。この補正値に物理モデルを当てはめると、バックグラウンドとノイズの形が変わる。`embedfit` は手順を逆にし、補正をモデルの側に置いたうえで、モデルと記録された計数をポアソン尤度で比べる（モデル組込み法）。順モデルは次のとおりである。

```
lambda = d + c Phi + k Phi f(x; theta)        true rate
mu_i   = T_i G^{-1}(lambda_i)                 expected recorded counts
N_i    ~ Poisson(mu_i)
```

1本のスペクトルから、推定値（たとえば光電子放出のしきい値）、その標準誤差、モデルがデータを説明しているかの判定（正常・要注意・異常）が得られる。この方法には、記録された計数（または計数を復元できる出力）と、計測器の補正式とその定数が必要である。

文献: 柳生進二郎，長田貴弘，中島嘉之「参照量で規格化された計数データの解析：補正式をモデルに組み込む方法と光電子収量分光への適用」（Journal of Surface Analysis に投稿中、プレプリントは Jxiv）。このパッケージを使ったときは、この論文を引用してほしい。

## インストール

```
pip install .            # numpy, scipy
pip install ".[plot]"      # + matplotlib for the figures
```

## 使い方（理研計器 AC シリーズ）

```python
from embedfit import Analysis, ScanSet

an = Analysis.from_file("examples/data/Au_AC5_20240229/Au 240229034207_1.dat").run()
print(an.report_text())         # recovery of counts, fit, verdict, display yield
an.plot("overview")             # 2 x 3 overview figure
an.plot("display", power=0.5)   # display yield; Y^(1/2) applied to the model curve only
an.to_csv("out")                # the values behind the figures, and a JSON report

scans = ScanSet.from_folder("examples/data/Au_AC5_20240229").run()
for row in scans.table():
    print(row["scan"], row["estimate"], row["se"], row["verdict"])
```

`report_text()` は計数の復元、当てはめ、判定、表示用の収量をまとめて返す。`plot("overview")` は 2 × 3 の概観図、`plot("display", power=0.5)` は表示用の収量の図である（Y^(1/2) はモデルの曲線にだけ施す）。`to_csv` は図の元になった数値と、報告する値の JSON を書き出す。

`examples/au_ac5.py` は、論文の Au の 10 スキャンを再現する。

## `run()` が行うこと

| 手順 | 関数 | 論文 |
|---|---|---|
| ファイルを読み、計測器が行った処理（G、Phi、暗計数の差し引き）を記録する | `readers.read_ac` | 4.2 節 |
| 暗計数を差し引いた計数率から整数の計数を復元する（経路 B） | `recover_counts` | 3.4.1 節、付録 B.1 |
| 順モデルをポアソン尤度で当てはめる。標準誤差は Fisher 情報量から求める | `fit` | 3.3 節、付録 B.3 |
| chi2/dof と z_low、対照モデル（c を自由、n を自由）、判定 | `verdict` | 3.5 節 |
| 表示用の収量 (G(N/T) - d_hat)/Phi と sigma_y | `display_yield` | 4.3 節、付録 B.4 |

## 設計

モデルは、手法ごとのサブクラスではなく、手法の間で独立に変わる部品から組み立てる。

| 部品 | モジュール | 用意しているもの |
|---|---|---|
| データと、計測器が行った処理 | `data.Spectrum` | 信号の種類 `counts`, `rate`, `rate_dark_subtracted`, `current` |
| 読み込み（ファイル形式ごとに1つ） | `readers` | 理研計器 AC シリーズ、汎用の表・配列 |
| 検出器の補正 G | `detector` | `RikenAC`, `NonParalyzable`, `Identity` |
| 信号の形 f | `model` | `PowerThreshold` |
| バックグラウンドの項 | `model` | `Constant`（d）, `FluxProportional`（c Phi）, `ExpDecay`, `PowerDecay`, `LinearDrift` |
| ノイズモデル | `noise` | `Poisson`, `Gaussian` |
| プリセット（手法 = 部品の選び方） | `presets` | `pys_counting`, `pys_current`（試験的） |

新しい手法は新しいプリセットとして、新しい計測器は新しい読み込み関数として加える（`readers.register` で登録する）。たとえば逆光電子分光は、計数のノイズと G を計数法の PYS と共有し、信号の形だけが異なる。電流法の PYS は信号の形を共有し、ガウス分布のノイズモデルを使う。

## 状況

* **計数法の PYS（AC シリーズ）**: JSA 論文の結果を再現する（テストは `tests/test_au.py`）。
* **電流法の PYS**: *試験的*。シミュレーションデータでのみ確かめている。エレクトロメータのフィルタはモデル化していない。ノイズのパラメータ sigma0 と rel は別に測定する必要がある。
* その他の手法（透過、逆光電子分光、XPS/UPS、NEXAFS）は実装していない。上の部品でこれらを扱えるように設計している。
* 従来法を自動化した方法（1/n-Scan、n-pys で公開）は含まない。

## テスト

```
pip install ".[test]"
pytest
```

## ライセンス

BSD 3-Clause（`LICENSE` を参照）。`examples/data` のサンプルデータも BSD 3-Clause（公開元の n-pys と同じ。元の著作権表示は `examples/data/Au_AC5_20240229/README.md`）である。
