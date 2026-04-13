#!/usr/bin/env python3
import os
import argparse
import shutil
import pandas as pd
from rdkit import Chem

def main():
    parser = argparse.ArgumentParser(
        description="GNINAスクリーニング結果からAffinity上位のポーズを抽出し、PDB/SDF形式で保存します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", default="filtered_results.csv", help="入力となるCSVファイルのパス")
    parser.add_argument("-n", "--top_n", type=int, default=100, help="抽出する上位化合物の数")
    parser.add_argument("-o", "--out_dir", default=None, help="出力ディレクトリ（未指定時は自動生成）")
    
    args = parser.parse_args()

    csv_file = args.input
    top_n = args.top_n
    out_dir = args.out_dir if args.out_dir else f"top{top_n}_affinity_poses_pdb"
    
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

    # 3. 抽出したトップポーズ同士を Affinity (pK) で降順（結合力が強い順）にソートし、上位N件を取得
    df_sorted = df_best_affinity.sort_values(by="Affinity (pK)", ascending=False).reset_index(drop=True)
    df_top_n = df_sorted.head(top_n)

    # 4. 結果を表形式でコンソールに出力
    print(f"\n🏆 各化合物の GNINA Affinity (pK) トップ {len(df_top_n)} リスト")
    print("=" * 90)
    try:
        # tabulateがインストールされていない環境への配慮
        print(df_top_n[['Compound ID', 'SMILES', 'Orig Rank', 'Affinity (pK)', 'CNN Pose Score']].to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        print("💡 ヒント: 'pip install tabulate' を実行すると綺麗なマークダウン表が出力されます。")
        print(df_top_n[['Compound ID', 'SMILES', 'Orig Rank', 'Affinity (pK)', 'CNN Pose Score']].to_string(index=False))
    print("=" * 90)

    # 5. 構造ファイル(Saved SDF)をPDBとSDFに変換して別ディレクトリに保存
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n📂 解析用構造を '{out_dir}' ディレクトリに PDB/SDF 形式で保存します...")

    for i, row in df_top_n.iterrows():
        src_file = str(row['Saved SDF'])
        complex_id = str(row['Compound ID'])
        affinity = float(row['Affinity (pK)'])
        orig_rank = int(row['Orig Rank'])
        
        if pd.isna(src_file) or not os.path.exists(src_file):
            print(f"  ⚠️ ファイルが見つかりません: {src_file}")
            continue
        
        overall_rank = i + 1
        base_filename = f"{overall_rank:03d}_{complex_id}_OrigRank{orig_rank}_pKd_{affinity:.2f}"
        dest_pdb = os.path.join(out_dir, f"{base_filename}.pdb")
        dest_sdf = os.path.join(out_dir, f"{base_filename}.sdf")
        
        # SDFはGNINAのメタデータを残すために単純コピー
        try:
            shutil.copy2(src_file, dest_sdf)
            
            # PDBへの変換のみRDKitを使用
            mol = Chem.MolFromMolFile(src_file, removeHs=False, sanitize=False)
            if mol is not None:
                Chem.MolToPDBFile(mol, dest_pdb)
                print(f"  ✅ [Rank {overall_rank:03d}] {complex_id} (DiffDock: {orig_rank}) -> 保存完了")
            else:
                print(f"  ⚠️ [Rank {overall_rank:03d}] {complex_id} SDFのコピーは成功しましたが、PDB変換に失敗しました。")
                
        except Exception as e:
             print(f"  ❌ [Rank {overall_rank:03d}] {complex_id} の処理中にエラーが発生しました: {e}")

    print("\n🎉 すべての処理が完了しました！")

if __name__ == "__main__":
    main()
