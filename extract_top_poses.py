#!/usr/bin/env python3
import os
import shutil
import pandas as pd

def main():
    csv_file = "filtered_results.csv"
    
    # 1. CSVファイルの存在確認と読み込み
    if not os.path.exists(csv_file):
        print(f"❌ エラー: '{csv_file}' が見つかりません。")
        print("先に 'run_diffgnina_01.py' を実行して結果を生成してください。")
        return

    df = pd.read_csv(csv_file)

    if df.empty:
        print("⚠️ 警告: CSVファイルにデータがありません。")
        return

    # 2. 各化合物の「GNINA Affinity (pK) が最大」のポーズを抽出
    # groupbyでComplex IDごとにグループ化し、Affinity (pK)が最大の行のインデックス(idxmax)を取得
    best_idx = df.groupby('Complex ID')['Affinity (pK)'].idxmax()
    df_best_affinity = df.loc[best_idx].copy()

    if df_best_affinity.empty:
        print("⚠️ 警告: 条件を満たすポーズが一つも残っていません。")
        return

    # 3. 抽出したトップポーズ同士を Affinity (pK) で降順（結合力が強い順）にソート
    df_sorted = df_best_affinity.sort_values(by="Affinity (pK)", ascending=False).reset_index(drop=True)

    # 4. 結果を表形式でコンソールに出力
    # Orig Rank（DiffDockの順位）も一緒に表示して比較できるようにしています
    print("\n🏆 各化合物の GNINA Affinity (pK) トップポーズ")
    print("=" * 90)
    print(df_sorted[['Complex ID', 'SMILES', 'Orig Rank', 'Affinity (pK)', 'CNN Pose Score']].to_markdown(index=False, floatfmt=".4f"))
    print("=" * 90)

    # 5. 構造ファイル(SDF)を別ディレクトリにコピー＆リネーム
    out_dir = "top_affinity_poses"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n📂 構造ファイルを '{out_dir}' ディレクトリに保存します...")

    for i, row in df_sorted.iterrows():
        src_file = row['File Path']
        complex_id = row['Complex ID']
        affinity = row['Affinity (pK)']
        orig_rank = int(row['Orig Rank'])
        
        # 元のファイルが存在するかチェック
        if pd.isna(src_file) or not os.path.exists(src_file):
            print(f"  ⚠️ ファイルが見つかりません: {src_file}")
            continue
        
        # 全体での順位 (1始まり)
        overall_rank = i + 1
        
        # 新しいファイル名の作成 (例: 01_complex_1_OrigRank4_pKd_3.66.sdf)
        # どのDiffDock順位のものがGNINAの1位に選ばれたか分かるようにOrigRankを含める
        new_filename = f"{overall_rank:02d}_{complex_id}_OrigRank{orig_rank}_pKd_{affinity:.2f}.sdf"
        dest_file = os.path.join(out_dir, new_filename)
        
        try:
            shutil.copy2(src_file, dest_file)
            print(f"  ✅ [Rank {overall_rank}] {complex_id} (DiffDock Rank: {orig_rank}) -> {new_filename}")
        except Exception as e:
            print(f"  ❌ {complex_id} のコピー中にエラーが発生しました: {e}")

    print("\n🎉 すべての処理が完了しました！")

if __name__ == "__main__":
    main()
