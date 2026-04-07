#!/usr/bin/env python3
import argparse
import sys
import subprocess
import os
import glob
import pandas as pd
import yaml
import tempfile
import concurrent.futures
import time

# ==========================================
# 1. DiffDock 実行 (チャンク対応・ID管理対応)
# ==========================================
def run_diffdock(protein_path, ligand_chunk, start_idx, n_poses, out_dir="results"):
    print(f"\n🤖 DiffDock 実行中 (リガンド Index {start_idx} 〜 {start_idx + len(ligand_chunk) - 1}, 各 {n_poses} ポーズ)...")
    
    # 大規模実行では毎回resultsを削除しない（続きから保存するため）
    os.makedirs(out_dir, exist_ok=True)

    input_csv = "input.csv"
    with open(input_csv, "w") as f:
        f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
        # ligand_chunk には (ID, SMILES) のタプルが入る
        for lig_id, smi in ligand_chunk:
            f.write(f"{lig_id},{os.path.abspath(protein_path)},{smi},\n")

    repo_path = os.path.abspath("DiffDock")
    if not os.path.exists(repo_path):
        print("❌ エラー: カレントディレクトリに 'DiffDock' フォルダが見つかりません。")
        sys.exit(1)

    yaml_path = os.path.join(repo_path, "default_inference_args.yaml")
    with open(yaml_path, 'r') as f: 
        config = yaml.safe_load(f)
        
    config['samples_per_complex'] = n_poses
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_yaml:
        yaml.dump(config, tmp_yaml)
        tmp_yaml_path = tmp_yaml.name

    cmd = [
        sys.executable, os.path.join(repo_path, "inference.py"),
        "--config", tmp_yaml_path, "--protein_ligand_csv", input_csv,
        "--out_dir", out_dir, "--inference_steps", "20",
        "--batch_size", "10", "--no_final_step_noise"
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = repo_path + ":" + env.get('PYTHONPATH', '')
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout: 
            print(line, end="")
        process.wait()
        
        if process.returncode != 0:
            print(f"❌ DiffDockの実行中にエラーが発生しました (Index {start_idx}付近)。")
            sys.exit(1)
    except KeyboardInterrupt:
        # Ctrl+Cで止められた時に、子プロセスのDiffDockを確実にキルする
        print("\n🛑 ユーザーによって処理が中断されました。バックグラウンドのDiffDockを強制終了します...")
        process.kill()
        process.wait()
        sys.exit(1)
    finally:
        if os.path.exists(tmp_yaml_path):
            os.remove(tmp_yaml_path)

# ==========================================
# 2. GNINA ワーカー関数 (完全修正版)
# ==========================================
def _evaluate_single_pose(args):
    sdf, protein_pdb, compound_id, smi, min_conf, min_cnn = args
    filename = os.path.basename(sdf)
    
    # 1. DiffDockのrank1.sdf (confidenceの記述がないファイル) への対応
    try:
        # "rank"と"_"の間、または"rank"と".sdf"の間の数字を安全に取得
        orig_rank_str = filename.split('rank')[1].split('_')[0].replace('.sdf', '')
        orig_rank = int(orig_rank_str)
        
        if "confidence" in filename:
            conf = float(filename.split('confidence')[1].replace('.sdf', ''))
        else:
            conf = 0.0  # rank1などconfidenceがない場合は0.0として扱う（足切り回避）
    except Exception as e: 
        return None

    if conf < min_conf:
        return None

    cmd = ["./gnina", "-r", protein_pdb, "-l", sdf, "--score_only"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    # 2. returncodeの厳格なチェックを削除！
    # OpenBabelの警告でエラー終了扱いになっても、スコアが取れていればOKとする
    
    cnn_pose_score = None
    cnn_affinity = None
    vina_score = None
    
    for line in proc.stdout.splitlines():
        if "CNNscore:" in line: cnn_pose_score = float(line.split()[1])
        elif "CNNaffinity:" in line: cnn_affinity = float(line.split()[1])
        elif "Affinity:" in line: vina_score = float(line.split()[1])

    # 警告ではなく本当にエラーで落ちて、スコアが1つも取れなかった場合は除外
    if cnn_pose_score is None:
        return None

    if cnn_pose_score < min_cnn:
        return None

    return {
        "Compound ID": compound_id,
        "SMILES": smi,
        "Orig Rank": orig_rank,
        "Model Conf": conf,
        "CNN Pose Score": cnn_pose_score,
        "Affinity (pK)": cnn_affinity,
        "Vina Score": vina_score,
        "File Path": sdf
    }

# ==========================================
# 3. GNINA オーケストレーション (フォルダ探索修正版)
# ==========================================
def evaluate_chunk_results(protein_pdb, ligand_chunk, start_idx, min_conf, min_cnn, num_workers, out_dir="results"):
    print(f"\n⚖️ チャンク結果を評価中 (Index {start_idx} 〜)...")
    
    tasks = []
    for lig_id, smi in ligand_chunk:
        complex_dir = None
        # 修正ポイント: DiffDock特有の "index0_lig_id" のようなフォルダ名に対応する
        if os.path.exists(out_dir):
            for dirname in os.listdir(out_dir):
                if dirname == lig_id or dirname.endswith(f"_{lig_id}"):
                    complex_dir = os.path.join(out_dir, dirname)
                    break
        
        if complex_dir is None:
            print(f"⚠️ 警告: DiffDockの出力フォルダ内に {lig_id} の結果が見つかりません。")
            continue
            
        sdf_files = glob.glob(os.path.join(complex_dir, "rank*.sdf"))
        for sdf in sdf_files:
            tasks.append((sdf, protein_pdb, lig_id, smi, min_conf, min_cnn))

    results = []
    actual_workers = min(num_workers, len(tasks)) if tasks else 1
    
    if actual_workers > 0 and len(tasks) > 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
            for res in executor.map(_evaluate_single_pose, tasks):
                if res is not None:
                    results.append(res)

    df = pd.DataFrame(results)
    
    if df.empty:
        print(f"⚠️ このチャンク (Index {start_idx} 〜) では閾値(Conf>={min_conf}, CNN>={min_cnn})を通過したポーズはありませんでした。")
        return

    # CNN Pose Score が高い順にソート
    df_sorted = df.sort_values(by=["Compound ID", "CNN Pose Score"], ascending=[True, False]).reset_index(drop=True)
    
    # CSVファイルへの「追記」保存
    output_csv = "filtered_results.csv"
    write_header = not os.path.exists(output_csv) # ファイルがない初回だけヘッダーを書く
    
    df_sorted.to_csv(output_csv, mode='a', header=write_header, index=False)
    print(f"✅ {len(df_sorted)} 個のポーズを '{output_csv}' に追記保存しました。")

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffDock + GNINA 10k Large Scale Pipeline")
    parser.add_argument("-p", required=True, help="標的タンパク質のPDBファイルパス")
    parser.add_argument("-l", required=True, help="化合物IDとSMILESが並んだテキストファイルパス (.smi)")
    parser.add_argument("-dp", type=int, default=20, help="DiffDockでの1リガンドあたりのポーズ生成数 (デフォルト: 20)")
    parser.add_argument("-mc", type=float, default=-2.0, help="min_model_confの閾値 (デフォルト: -2.0)")
    parser.add_argument("-cp", type=float, default=0.3, help="min_cnn_pose_scoreの閾値 (デフォルト: 0.3)")
    parser.add_argument("-w", "--workers", type=int, default=1, help="GNINA並列実行時のワーカー数 (デフォルト: 1)")
    
    # 大規模用の追加引数
    parser.add_argument("-c", "--chunk_size", type=int, default=100, help="1度に処理するリガンドの数 (デフォルト: 100)")
    parser.add_argument("-s", "--start_idx", type=int, default=0, help="処理を開始するリガンドのインデックス(0始まり)。再開時に使用 (デフォルト: 0)")
    
    args = parser.parse_args()

    if not os.path.exists(args.p) or not os.path.exists(args.l):
        print("❌ エラー: 入力ファイルが見つかりません。")
        sys.exit(1)

    # ファイルから ID と SMILES を読み込む (タブ・スペース両対応)
    ligand_list = []
    with open(args.l, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split() # スペースやタブで分割
            if len(parts) >= 2:
                # ID SMILES の順になっていることを想定
                lig_id, smi = parts[0], parts[1]
            else:
                # 万が一IDがない行があった場合のフォールバック
                lig_id = f"complex_{len(ligand_list)}"
                smi = parts[0]
            ligand_list.append((lig_id, smi))

    total_ligands = len(ligand_list)
    if total_ligands == 0:
        print("❌ エラー: リガンドリストが空です。")
        sys.exit(1)

    print(f"🚀 大規模スクリーニングを開始します (全 {total_ligands} リガンド)")
    print(f"📦 チャンクサイズ: {args.chunk_size} / 開始インデックス: {args.start_idx}")
    print("=" * 60)

    start_time = time.time()

    # チャンクごとにループ処理
    for chunk_start in range(args.start_idx, total_ligands, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, total_ligands)
        chunk_ligands = ligand_list[chunk_start:chunk_end]
        
        print(f"\n" + "="*60)
        print(f"🔄 チャンク処理開始: Index {chunk_start} から {chunk_end - 1} ({len(chunk_ligands)} 個)")
        print("="*60)

        # 1. DiffDock (チャンク分だけ)
        run_diffdock(args.p, chunk_ligands, chunk_start, args.dp, out_dir="results")
        
        # 2. GNINA評価とCSV追記 (チャンク分だけ)
        evaluate_chunk_results(args.p, chunk_ligands, chunk_start, args.mc, args.cp, args.workers, out_dir="results")

    elapsed_time = time.time() - start_time
    print(f"\n🎉🎉 すべての処理が完了しました！ (総所要時間: {elapsed_time/3600:.2f} 時間)")
