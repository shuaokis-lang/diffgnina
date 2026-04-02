
# DiffDock + GNINA 実行パイプライン

タンパク質とリガンドのドッキング予測（DiffDock）と、その結果の物理的妥当性の評価（GNINA）を行うスクリプトです。

## ⚠️ 前提条件
* OS: Ubuntu
* GPU: NVIDIA製 GPU (VRAM 8GB以上推奨)
* ドライバ: `nvidia-smi` コマンドで CUDA Version が 12.1 以上であること
* Python: Python 3.10 または 3.12 (および `python3-venv`)

## 1. 環境構築の手順
ターミナルを開き、この README ファイルがあるディレクトリに移動して以下を実行してください。

### ① 仮想環境の作成と有効化
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
② 必要なライブラリのインストール
Bash
￼
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f [https://data.pyg.org/whl/torch-2.3.0+cu121.html](https://data.pyg.org/whl/torch-2.3.0+cu121.html)
pip install torch-geometric
pip install e3nn fair-esm prody biopython rdkit scipy networkx pyyaml pandas tabulate
2. 実行方法
ご自身で前処理（水分子や不要なリガンドの除去）を行った標的タンパク質のPDBファイルと、リガンドのSMILESリストを用意し、以下のコマンドで実行します。

Bash
￼
python run_diffgina.py -p receptor.pdb -l ligand.txt -dp 10 -mc -2.0 -cp 0.3
-p: 標的タンパク質のファイル（例: receptor.pdb）

-l: リガンドのSMILESが縦に並んだテキストファイル（例: ligand.txt）

-dp: 1リガンドあたりの生成ポーズ数（デフォルト 10）

-mc: DiffDockの信頼度スコアの足切り閾値（デフォルト -2.0）

-cp: GNINAのCNN Pose Scoreの足切り閾値（デフォルト 0.3）

※ 初回実行時のみ、AIモデル（数GB）のダウンロードが行われます。
※ RuntimeError: CUDA out of memory. が出た場合は、run_diffgina.py 内の --batch_size を 1 に変更して再実行してください。


---

**2. パッケージ化のための実行コマンド**
`README.md` を保存した後、カレントディレクトリ（`~/diffgnina`）で以下を実行してください。

```bash
# 過去の実行結果をクリーンアップ
rm -rf results filtered_results.csv input.csv

# 一つ上の階層に移動
cd ..

# 仮想環境(.venv)を除外して圧縮
tar --exclude='diffgnina/.venv' -czvf diffgnina_package.tar.gz diffgnina/
以上で、相手に渡すための diffgnina_package.tar.gz が完成します。
