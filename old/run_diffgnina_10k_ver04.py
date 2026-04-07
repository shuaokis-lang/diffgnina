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
    
    os.makedirs(out_dir, exist_ok=True)

    input_csv = "input.csv"
    with open(input_csv, "w") as f:
        f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
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
        print("\n🛑 ユーザーによって処理が中断されました。強制終了します...")
        process.kill()
        process.wait()
        sys.exit(1)
    finally:
        if os.path.exists(tmp_yaml_path):
            os.remove(tmp_yaml_path)

# ==========================================
# 2. GNINA ワーカー関数 (Minimize 組み込み版)
# ==========================================
def _evaluate_single_pose(args):
    sdf, protein_pdb, compound_id, smi, min_conf, min_cnn = args
    filename = os.path.basename(sdf)
    
    try:
        # rank情報の抽出
        orig_rank_str = filename.split('rank')[1].split('_')[0].replace('.sdf', '')
        orig_rank = int(orig_rank_str)
        
        if "confidence" in filename:
            conf = float(filename.split('confidence')[1].replace('.sdf', ''))
        else:
            conf = 0.0  # rank1等でconfidence表記がない場合はパスさせる
    except Exception: 
        return None

    # DiffDockの信頼度スコアで事前フィルタリング
    if conf < min_conf:
        return None

    # GNINA実行コマンド
    # --minimize: 物理的な歪みをとるための構造最適化
    # --autobox_ligand: 入力リガンドの周りで最適化を行う
    # --no_gpu: 並列実行時のVRAM枯渇を避けるためCPUを使用
    cmd = [
        "./gnina", 
        "-r", protein_pdb, 
        "-l", sdf, 
        "--score_only", 
        "--minimize", 
        "--autobox_ligand", sdf,
        "--no_gpu"
    ]
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    cnn_pose_score = None
    cnn_affinity = None
    vina_score = None
    
    for line in proc.stdout.splitlines():
        if "CNNscore:" in line: cnn_pose_score = float(line.split()[1])
        elif "CNNaffinity:" in line: cnn_affinity = float(line.split()[1])
        elif "Affinity:" in line: vina_score = float(line.split()[1])

    # スコアが取得できない場合や、CNNスコアが閾値未満の場合は除外
    if cnn_pose_score is None or cnn_pose_score < min_cnn:
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
# 3. GNINA オーケストレーション
# ==========================================
def evaluate_chunk_results(protein_pdb, ligand_chunk, start_idx, min_conf, min_cnn, num_workers, out_dir="results"):
    print(f"\n⚖️ 評価・最適化中 (Index {start_idx} 〜, 並列数: {num_workers})...")
    
    tasks = []
    for lig_id, smi in ligand_chunk:
        complex_dir = None
        if os.path.exists(out_dir):
            for dirname in os.listdir(out_dir):
                if dirname == lig_id or dirname.endswith(f"_{lig_id}"):
                    complex_dir = os.path.join(out_dir, dirname)
                    break
        
        if complex_dir is None:
            continue
            
        sdf_files = glob.glob(os.path.join(complex_dir, "rank*.sdf"))
        for sdf in sdf_files:
            tasks.append((sdf, protein_pdb, lig_id, smi, min_conf, min_cnn))

    results = []
    if tasks:
        actual_workers = min(num_workers, len(tasks))
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
            for res in executor.map(_evaluate_single_pose, tasks):
                if res is not None:
                    results.append(res)

    if not results:
        print(f"⚠️ チャンク (Index {start_idx}) で条件を満たすポーズはありませんでした。")
        return

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by=["Compound ID", "CNN Pose Score"], ascending=[True, False]).reset_index(drop=True)
    
    output_csv = "filtered_results.csv"
    write_header = not os.path.exists(output_csv)
    df_sorted.to_csv(output_csv, mode='a', header=write_header, index=False)
    print(f"✅ {len(df_sorted)} 個の最適化済みポーズを '{output_csv}' に追記しました。")

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffDock + GNINA (with Minimize) Large Scale Pipeline")
    parser.add_argument("-p", required=True, help="標的タンパク質のPDBファイル")
    parser.add_argument("-l", required=True, help="化合物リスト (.smi)")
    parser.add_argument("-dp", type=int, default=20, help="生成ポーズ数")
    parser.add_argument("-mc", type=float, default=-2.0, help="Min DiffDock Conf")
    parser.add_argument("-cp", type=float, default=0.3, help="Min GNINA CNN Score")
    parser.add_argument("-w", "--workers", type=int, default=4, help="GNINA並列数")
    parser.add_argument("-c", "--chunk_size", type=int, default=100, help="チャンクサイズ")
    parser.add_argument("-s", "--start_idx", type=int, default=0, help="開始インデックス")
    
    args = parser.parse_args()

    # SMILES読み込み
    ligand_list = []
    with open(args.l, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                ligand_list.append((parts[0], parts[1]))
            elif len(parts) == 1:
                ligand_list.append((f"lig_{len(ligand_list)}", parts[0]))

    total = len(ligand_list)
    print(f"🚀 スクリーニング開始: 全 {total} リガンド / 構造最適化: ON")
    start_time = time.time()

    for chunk_start in range(args.start_idx, total, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, total)
        chunk_ligands = ligand_list[chunk_start:chunk_end]
        
        print(f"\n" + "="*50)
        print(f"📦 CHUNK: {chunk_start} - {chunk_end-1}")
        print("="*50)

        run_diffdock(args.p, chunk_ligands, chunk_start, args.dp, out_dir="results")
        evaluate_chunk_results(args.p, chunk_ligands, chunk_start, args.mc, args.cp, args.workers, out_dir="results")

    print(f"\n🎉 完了! 総時間: {(time.time()-start_time)/3600:.2f}h")
