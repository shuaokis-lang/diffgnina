#!/usr/bin/env python3
import os
import subprocess
import pandas as pd

def main():
    csv_file = "filtered_results.csv"
    top_n = 100  # 抽出する上位の件数
    
    # 1. CSVファイルの存在確認と読み込み
    if not os.path.exists(csv_file):
        print(f"❌ エラー: '{csv_file}' が見つかりません。")
        print("先にスクリーニングスクリプトを実行して結果を生成してください。")
        return

    df = pd.read_csv(csv_file)

    if df.empty:
        print("⚠️ 警告: CSVファイルにデータがありません。")
        return

    # 最新版(ver06)に合わせてカラム名を "Minimized SDF" に変更
    if 'Minimized SDF' not in df.columns:
        print("❌ エラー: CSVファイルに 'Minimized SDF' カラムが見つかりません。最新のパイプラインの出力を使用してください。")
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
    print(f"\n🏆 各化合物の GNINA Affinity (pK) トップ {len(df_top_n)} ポーズ")
    print("=" * 90)
    print(df_top_n[['Compound ID', 'SMILES', 'Orig Rank', 'Affinity (pK)', 'CNN Pose Score']].to_markdown(index=False, floatfmt=".4f"))
    print("=" * 90)

    # 5. 構造ファイル(SDF)をPDBに変換して別ディレクトリに保存
    out_dir = "top100_affinity_poses_pdb"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n📂 解析用構造を '{out_dir}' ディレクトリに PDB 形式で保存します...")

    for i, row in df_top_n.iterrows():
        # 最新版のカラム名を参照
        src_file = row['Minimized SDF']
        complex_id = row['Compound ID']
        affinity = row['Affinity (pK)']
        orig_rank = int(row['Orig Rank'])
        
        if pd.isna(src_file) or not os.path.exists(str(src_file)):
            print(f"  ⚠️ ファイルが見つかりません: {src_file}")
            continue
        
        overall_rank = i + 1
        
        # 新しいファイル名の作成 (100位まで綺麗に並ぶよう 03d に変更)
        new_filename = f"{overall_rank:03d}_{complex_id}_OrigRank{orig_rank}_pKd_{affinity:.2f}.pdb"
        dest_file = os.path.join(out_dir, new_filename)
        
        # OpenBabel: 入力をSDF (-isdf) としてPDBに変換 (すべての水素が維持されます)
        obabel_cmd = ["obabel", "-isdf", str(src_file), "-opdb", "-O", dest_file]
        
        try:
            proc = subprocess.run(obabel_cmd, capture_output=True, text=True)
            if os.path.exists(dest_file):
                print(f"  ✅ [Rank {overall_rank:03d}] {complex_id} (DiffDock Rank: {orig_rank}) -> {new_filename}")
            else:
                print(f"  ❌ {complex_id} のPDB変換に失敗しました。")
        except Exception as e:
            print(f"  ❌ {complex_id} の処理中にエラーが発生しました: {e}")

    print("\n🎉 すべての処理が完了しました！")

if __name__ == "__main__":
    main()
