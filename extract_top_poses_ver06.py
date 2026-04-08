#!/usr/bin/env python3
import os
import pandas as pd
from rdkit import Chem

def main():
    csv_file = "filtered_results.csv"
    top_n = 100 # 抽出する上位化合物の数
    out_dir = f"top{top_n}_affinity_poses_pdb"
    
    # 1. CSVファイルの存在確認と読み込み
    if not os.path.exists(csv_file):
        print(f"❌ エラー: '{csv_file}' が見つかりません。")
        print("先にスクリーニングスクリプトを実行して結果を生成してください。")
        return

    df = pd.read_csv(csv_file)

    if df.empty:
        print("⚠️ 警告: CSVファイルにデータがありません。")
        return

    if 'Saved SDF' not in df.columns:
        print("❌ エラー: CSVファイルに 'Saved SDF' カラムが見つかりません。")
        return

    # 2. 各化合物の「GNINA Affinity (pK) が最大」のポーズを抽出
    best_idx = df.groupby('Compound ID')['Affinity (pK)'].idxmax()
    df_best_affinity = df.loc[best_idx].copy()

    if df_best_affinity.empty:
        print("⚠️ 警告: 条件を満たすポーズが一つも残っていません。")
        return

    # 3. 抽出したトップポーズ同士を Affinity (pK) で降順（結合力が強い順）にソートし、上位100件を取得
    df_sorted = df_best_affinity.sort_values(by="Affinity (pK)", ascending=False).reset_index(drop=True)
    df_top_n = df_sorted.head(top_n)

    # 4. 結果を表形式でコンソールに出力
    print(f"\n🏆 各化合物の GNINA Affinity (pK) トップ {len(df_top_n)} リスト")
    print("=" * 90)
    print(df_top_n[['Compound ID', 'SMILES', 'Orig Rank', 'Affinity (pK)', 'CNN Pose Score']].to_markdown(index=False, floatfmt=".4f"))
    print("=" * 90)

    # 5. 構造ファイル(Saved SDF)をPDBとSDFに変換して別ディレクトリに保存
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n📂 解析用構造を '{out_dir}' ディレクトリに PDB/SDF 形式で保存します...")

    for i, row in df_top_n.iterrows():
        src_file = row['Saved SDF']
        complex_id = row['Compound ID']
        affinity = row['Affinity (pK)']
        orig_rank = int(row['Orig Rank'])
        
        if pd.isna(src_file) or not os.path.exists(str(src_file)):
            print(f"  ⚠️ ファイルが見つかりません: {src_file}")
            continue
        
        overall_rank = i + 1
        
        # 新しいファイル名の作成 (100位まであるため 03d で3桁ゼロ埋めに変更)
        base_filename = f"{overall_rank:03d}_{complex_id}_OrigRank{orig_rank}_pKd_{affinity:.2f}"
        dest_pdb = os.path.join(out_dir, f"{base_filename}.pdb")
        dest_sdf = os.path.join(out_dir, f"{base_filename}.sdf")
        
        # RDKitを使用して単純なフォーマット変換を行う
        mol = Chem.MolFromMolFile(str(src_file), removeHs=False, sanitize=False)
        
        if mol is not None:
            Chem.MolToPDBFile(mol, dest_pdb)
            Chem.MolToMolFile(mol, dest_sdf)
            print(f"  ✅ [Rank {overall_rank:03d}] {complex_id} (DiffDock Rank: {orig_rank}) -> 保存完了")
        else:
            print(f"  ❌ [Rank {overall_rank:03d}] {complex_id} の変換に失敗しました。")

    print("\n🎉 すべての処理が完了しました！")

if __name__ == "__main__":
    main()
