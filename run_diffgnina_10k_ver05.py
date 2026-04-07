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
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout: 
            # ログ出力が不要な場合は pass にしてください
            pass
        process.wait()
        if process.returncode != 0:
            print(f"❌ DiffDockエラー (Index {start_idx})")
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

    # --- 1. DiffDockの出力に、イオン化状態を保持して水素付加 ---
    sdf_with_h = orig_sdf.replace(".sdf", "_H.sdf")
    mol = Chem.MolFromMolFile(orig_sdf, sanitize=True)
    if mol is None: return None
    
    mol_with_h = Chem.AddHs(mol, addCoords=True)
    Chem.MolToMolFile(mol_with_h, sdf_with_h)

    if not os.path.exists(sdf_with_h): return None

    # --- 2. GNINAでスコアリング (構造は出力させない) ---
    cmd = [
        "./gnina", "-r", protein_pdb, "-l", sdf_with_h, 
        "--score_only", "--minimize", "--autobox_ligand", sdf_with_h,
        "--no_gpu"
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

    return {
        "Compound ID": compound_id, "SMILES": smi, "Orig Rank": orig_rank,
        "Model Conf": conf, "CNN Pose Score": cnn_pose_score,
        "Affinity (pK)": cnn_affinity, "Vina Score": vina_score,
        "Saved SDF": sdf_with_h  # 👈 スコア基準を満たした「DiffDock出力＋水素」のファイルパス
    }

# ==========================================
# 3. オーケストレーション
# ==========================================
def evaluate_chunk_results(protein_pdb, ligand_chunk, start_idx, min_conf, min_cnn, num_workers, out_dir="results", output_csv="filtered_results.csv"):
    print(f"\n⚖️ 評価・最適化中 (Index {start_idx} 〜, 並列数: {num_workers})...")
    
    tasks = []
    for lig_id, smi in ligand_chunk:
        complex_dir = None
        if os.path.exists(out_dir):
            for dirname in os.listdir(out_dir):
                if dirname == lig_id or dirname.endswith(f"_{lig_id}"):
                    complex_dir = os.path.join(out_dir, dirname)
                    break
        if complex_dir is None: continue
            
        for sdf in glob.glob(os.path.join(complex_dir, "rank*.sdf")):
            if "_H" not in sdf: # 水素付加済みのものが混ざるのを防止
                tasks.append((sdf, protein_pdb, lig_id, smi, min_conf, min_cnn))

    results = []
    if tasks:
        actual_workers = min(num_workers, len(tasks))
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
            for res in executor.map(_evaluate_single_pose, tasks):
                if res is not None: results.append(res)

    if not results: return

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by=["Compound ID", "CNN Pose Score"], ascending=[True, False]).reset_index(drop=True)
    write_header = not os.path.exists(output_csv)
    df_sorted.to_csv(output_csv, mode='a', header=write_header, index=False)
    print(f"✅ {len(df_sorted)} 個のポーズを '{output_csv}' に追記しました。")

# ==========================================
# 4. トップ結果の出力 (DiffDock構造＋水素 のまま保存)
# ==========================================
def export_top_results(csv_file="filtered_results.csv", top_n=10, out_dir="top10_results"):
    if not os.path.exists(csv_file): return
    df = pd.read_csv(csv_file)
    if df.empty: return

    df_sorted = df.sort_values(by="CNN Pose Score", ascending=False)
    df_unique = df_sorted.drop_duplicates(subset=["Compound ID"], keep="first").reset_index(drop=True)
    top_df = df_unique.head(top_n)

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n🏆 解析用: トップ {len(top_df)} 化合物の保存を開始します -> '{out_dir}/'")

    for idx, row in top_df.iterrows():
        lig_id = row["Compound ID"]
        score = row["CNN Pose Score"]
        
        # GNINAの出力ではなく、保持しておいた「DiffDock出力＋水素」のパスを使用
        sdf_path = row.get("Saved SDF")
        
        if pd.isna(sdf_path) or not os.path.exists(str(sdf_path)): continue

        out_pdb = os.path.join(out_dir, f"Top{idx+1:02d}_{lig_id}_CNN{score:.3f}.pdb")
        out_sdf = os.path.join(out_dir, f"Top{idx+1:02d}_{lig_id}_CNN{score:.3f}.sdf")
        
        # 保存用の読み込みでは、すでに完全な水素付き構造なので sanitize=False で単純変換する
        mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False, sanitize=False)
        
        if mol is not None:
            Chem.MolToPDBFile(mol, out_pdb)
            Chem.MolToMolFile(mol, out_sdf)
            print(f"  [{idx+1:02d}] ✅ 成功: {lig_id} (DiffDockポーズ + 水素)")
        else:
            print(f"  [{idx+1:02d}] ❌ 失敗: {lig_id}")

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffDock -> RDKit(水素付加) -> GNINA(スコアのみ) Pipeline")
    parser.add_argument("-p", required=True, help="標的タンパク質のPDBファイル")
    parser.add_argument("-l", required=True, help="化合物リスト (.smi)")
    parser.add_argument("-dp", type=int, default=20, help="生成ポーズ数")
    parser.add_argument("-mc", type=float, default=-2.0, help="Min DiffDock Conf")
    parser.add_argument("-cp", type=float, default=0.3, help="Min GNINA CNN Score")
    parser.add_argument("-w", "--workers", type=int, default=4, help="GNINA並列数")
    parser.add_argument("-c", "--chunk_size", type=int, default=100, help="チャンクサイズ")
    parser.add_argument("-s", "--start_idx", type=int, default=0, help="開始インデックス")
    
    args = parser.parse_args()

    ligand_list = []
    with open(args.l, "r") as f:
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
        run_diffdock(args.p, chunk_ligands, chunk_start, args.dp, out_dir="results")
        evaluate_chunk_results(args.p, chunk_ligands, chunk_start, args.mc, args.cp, args.workers, out_dir="results", output_csv=output_csv)

    print(f"\n🎉 完了! (総時間: {(time.time()-start_time)/3600:.2f}h)")
    export_top_results(csv_file=output_csv, top_n=10, out_dir="top10_results")
