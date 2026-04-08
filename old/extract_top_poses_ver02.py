#!/usr/bin/env python3
import os
import pandas as pd
from rdkit import Chem

def main():
    csv_file = "filtered_results.csv"
    
    # 1. CSVファイルの存在確認と読み込み
    if not os.path.exists(csv_file):
        print(f"❌ エラー: '{csv_file}' が見つかりません。")
        print("先にスクリーニングスクリプトを実行して結果を生成してください。")
        return

    df = pd.read_csv(csv_file)

    if df.empty:
        print("⚠️ 警告: CSVファイルにデータがありません。")
        return

    # 最新版(ver10)に合わせてカラム名を "Saved SDF" に変更
    if 'Saved SDF' not in df.columns:
        print("❌ エラー: CSVファイルに 'Saved SDF' カラムが見つかりません。最新のパイプライン(ver10)の出力を使用してください。")
        return

    # 2. 各化合物の「GNINA Affinity (pK) が最大」のポーズを抽出
    best_idx = df.groupby('Compound ID')['Affinity (pK)'].idxmax()
    df_best_affinity = df.loc[best_idx].copy()

    if df_best_affinity.empty:
        print("⚠️ 警告: 条件を満たすポーズが一つも残っていません。")
        return

    # 3. 抽出したトップポーズ同士を Affinity (pK) で降順（結合力が強い順）にソート
    df_sorted = df_best_affinity.sort_values(by="Affinity (pK)", ascending=False).reset_index(drop=True)

    # 4. 結果を表形式でコンソールに出力
    print("\n🏆 各化合物の GNINA Affinity (pK) トップポーズ")
    print("=" * 90)
    print(df_sorted[['Compound ID', 'SMILES', 'Orig Rank', 'Affinity (pK)', 'CNN Pose Score']].to_markdown(index=False, floatfmt=".4f"))
    print("=" * 90)

    # 5. 構造ファイル(Saved SDF)をPDBとSDFに変換して別ディレクトリに保存
    out_dir = "top_affinity_poses_pdb"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n📂 解析用構造を '{out_dir}' ディレクトリに PDB/SDF 形式で保存します...")

    for i, row in df_sorted.iterrows():
        # 最新版のカラム名を参照
        src_file = row['Saved SDF']
        complex_id = row['Compound ID']
        affinity = row['Affinity (pK)']
        orig_rank = int(row['Orig Rank'])
        
        if pd.isna(src_file) or not os.path.exists(str(src_file)):
            print(f"  ⚠️ ファイルが見つかりません: {src_file}")
            continue
        
        overall_rank = i + 1
        
        # 新しいファイル名の作成
        base_filename = f"{overall_rank:02d}_{complex_id}_OrigRank{orig_rank}_pKd_{affinity:.2f}"
        dest_pdb = os.path.join(out_dir, f"{base_filename}.pdb")
        dest_sdf = os.path.join(out_dir, f"{base_filename}.sdf")
        
        # OpenBabelの代わりにRDKitを使用 (安全のため sanitize=False で単純なフォーマット変換を行う)
        mol = Chem.MolFromMolFile(str(src_file), removeHs=False, sanitize=False)
        
        if mol is not None:
            Chem.MolToPDBFile(mol, dest_pdb)
            Chem.MolToMolFile(mol, dest_sdf)
            print(f"  ✅ [Rank {overall_rank:02d}] {complex_id} (DiffDock Rank: {orig_rank}) -> 保存完了")
        else:
            print(f"  ❌ [Rank {overall_rank:02d}] {complex_id} の変換に失敗しました。")

    print("\n🎉 すべての処理が完了しました！")

if __name__ == "__main__":
    main()
