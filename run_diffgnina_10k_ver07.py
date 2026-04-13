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
import shutil
from collections import deque
from rdkit import Chem

# ==========================================
# 1. DiffDock 実行
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
    
    log_queue = deque(maxlen=30)
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout: 
            log_queue.append(line.strip())
            
        process.wait()
        if process.returncode != 0:
            print(f"❌ DiffDockエラー (Index {start_idx})。直近のログを出力します:")
            for log_line in log_queue:
                print(log_line)
            sys.exit(1)
    except KeyboardInterrupt:
        process.kill()
        sys.exit(1)
    finally:
        if os.path.exists(tmp_yaml_path): os.remove(tmp_yaml_path)

# ==========================================
# 2. RDKit水素付加 ＆ GNINAスコアリング
# ==========================================
def _evaluate_single_pose(args):
    orig_sdf, protein_pdb, compound_id, smi, min_conf, min_cnn = args
    filename = os.path.basename(orig_sdf)
    
    try:
        orig_rank = int(filename.split('rank')[1].split('_')[0].replace('.sdf', ''))
        conf = float(filename.split('confidence')[1].replace('.sdf', '')) if "confidence" in filename else 0.0
    except Exception: 
        return None

    if conf < min_conf: return None

    # --- 1. RDKitで水素付加 (サニタイズできない破綻構造は弾く) ---
    mol = Chem.MolFromMolFile(orig_sdf, sanitize=True)
    if mol is None: return None
    
    sdf_with_h = orig_sdf.replace(".sdf", "_H.sdf")
    try:
        mol_with_h = Chem.AddHs(mol, addCoords=True)
        Chem.MolToMolFile(mol_with_h, sdf_with_h)
    except Exception:
        return None

    if not os.path.exists(sdf_with_h): return None

    # --- 2. GNINAでスコアのみ計算 (構造は出力しない) ---
    # --score_only と --minimize を併用することで、「最適化した上でのスコア」を計算しつつ、ファイルは出力しません。
    cmd = [
        "./gnina", "-r", protein_pdb, "-l", sdf_with_h, 
        "--score_only", "--minimize", "--autobox_ligand", sdf_with_h,
        "--cpu", "1"
    ]
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    cnn_pose_score, cnn_affinity, vina_score = None, None, None
    for line in proc.stdout.splitlines():
        if "CNNscore:" in line: cnn_pose_score = float(line.split()[1])
        elif "CNNaffinity:" in line: cnn_affinity = float(line.split()[1])
        elif "Affinity:" in line: vina_score = float(line.split()[1])

    # 基準を満たさない場合は、作成した水素付きSDFを削除
    if cnn_pose_score is None or cnn_pose_score < min_cnn:
        if os.path.exists(sdf_with_h): os.remove(sdf_with_h)
        return None

    # 基準を満たした場合は、水素付加済みのSDF（sdf_with_h）をそのまま保存ファイルとして残す
    return {
        "Compound ID": compound_id, "SMILES": smi, "Orig Rank": orig_rank,
        "Model Conf": conf, "CNN Pose Score": cnn_pose_score,
        "Affinity (pK)": cnn_affinity, "Vina Score": vina_score,
        "Saved SDF": sdf_with_h  # RDKitで水素を付加した直後のSDF
    }

# ==========================================
# 3. オーケストレーション
# ==========================================
def evaluate_chunk_results(protein_pdb, ligand_chunk, start_idx, min_conf, min_cnn, num_workers, max_poses, out_dir="results", output_csv="filtered_results.csv"):
    print(f"\n⚖️ 評価中 (Index {start_idx} 〜, 並列数: {num_workers}, 評価ポーズ数: 各上位 {max_poses})...")
    
    tasks = []
    complex_dirs_to_clean = [] 
    
    for lig_id, smi in ligand_chunk:
        complex_dir = None
        if os.path.exists(out_dir):
            for dirname in os.listdir(out_dir):
                if dirname == lig_id or dirname.endswith(f"_{lig_id}") or lig_id in dirname:
                    complex_dir = os.path.join(out_dir, dirname)
                    complex_dirs_to_clean.append(complex_dir)
                    break
        
        if complex_dir is None: continue
        
        sdf_files = glob.glob(os.path.join(complex_dir, "rank*.sdf"))
        parsed_sdfs = []
        
        for sdf in sdf_files:
            if "_H" in sdf: continue
            filename = os.path.basename(sdf)
            try:
                rank = int(filename.split('rank')[1].split('_')[0].replace('.sdf', ''))
                parsed_sdfs.append((rank, sdf))
            except Exception:
                pass
                
        parsed_sdfs.sort(key=lambda x: x[0])
        top_sdfs = [sdf for rank, sdf in parsed_sdfs[:max_poses]]

        for sdf in top_sdfs:
            tasks.append((sdf, protein_pdb, lig_id, smi, min_conf, min_cnn))

    results = []
    if tasks:
        actual_workers = min(num_workers, len(tasks))
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
            for res in executor.map(_evaluate_single_pose, tasks):
                if res is not None: results.append(res)

    # クリーンアップ: 不要なDiffDockの初期出力SDF(水素なし)を削除
    for d in complex_dirs_to_clean:
        for f in glob.glob(os.path.join(d, "rank*.sdf")):
            if "_H" not in f:
                try: os.remove(f)
                except: pass
        
        if not os.listdir(d):
            try: os.rmdir(d)
            except: pass

    if not results: return

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by=["Compound ID", "CNN Pose Score"], ascending=[True, False]).reset_index(drop=True)
    write_header = not os.path.exists(output_csv)
    df_sorted.to_csv(output_csv, mode='a', header=write_header, index=False)
    print(f"✅ {len(df_sorted)} 個のポーズを '{output_csv}' に追記しました。")


# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DiffDock -> RDKit(水素付加) -> GNINA(スコア計算) パイプライン",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-p", "--protein", required=True, help="標的タンパク質のPDBファイルパス")
    parser.add_argument("-l", "--ligand", required=True, help="化合物リスト(.smi) (フォーマット: ID SMILES)")
    parser.add_argument("-dp", "--diffdock_poses", type=int, default=20, help="DiffDockで生成するポーズ数")
    parser.add_argument("-mp", "--max_poses", type=int, default=5, help="GNINAで評価するDiffDockの上位ポーズ数")
    parser.add_argument("-mc", "--min_conf", type=float, default=-2.0, help="DiffDock Confidenceの最小値")
    parser.add_argument("-cp", "--cnn_pose", type=float, default=0.3, help="保存対象とするGNINA CNN Pose Scoreの最小値")
    parser.add_argument("-w", "--workers", type=int, default=4, help="GNINAの並列実行数(VRAMに合わせて調整)")
    parser.add_argument("-c", "--chunk_size", type=int, default=100, help="チャンクサイズ")
    parser.add_argument("-s", "--start_idx", type=int, default=0, help="開始インデックス")
    
    args = parser.parse_args()

    ligand_list = []
    with open(args.ligand, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2: ligand_list.append((parts[0], parts[1]))
            elif len(parts) == 1: ligand_list.append((f"lig_{len(ligand_list)}", parts[0]))

    total = len(ligand_list)
    print(f"🚀 スクリーニング開始: 全 {total} リガンド")
    start_time = time.time()

    output_csv = "filtered_results.csv"
    if args.start_idx == 0 and os.path.exists(output_csv): os.remove(output_csv)

    for chunk_start in range(args.start_idx, total, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, total)
        chunk_ligands = ligand_list[chunk_start:chunk_end]
        
        print(f"\n" + "="*50 + f"\n📦 CHUNK: {chunk_start} - {chunk_end-1}\n" + "="*50)
        run_diffdock(args.protein, chunk_ligands, chunk_start, args.diffdock_poses, out_dir="results")
        evaluate_chunk_results(args.protein, chunk_ligands, chunk_start, args.min_conf, args.cnn_pose, args.workers, args.max_poses, out_dir="results", output_csv=output_csv)

    print(f"\n🎉 完了! (総時間: {(time.time()-start_time)/3600:.2f}h)")
