#!/usr/bin/env python3
import argparse
import sys
import subprocess
import os
import glob
import pandas as pd
import yaml

# ==========================================
# 1. DiffDock 実行 (複数リガンド対応)
# ==========================================
def run_diffdock(protein_path, smiles_list, n_poses):
    print(f"\n🤖 DiffDock 実行中 ({len(smiles_list)} 個のリガンド, 各 {n_poses} ポーズ)...")
    if os.path.exists("results"):
        import shutil
        shutil.rmtree("results")

    # 複数リガンド用の入力CSVを作成
    input_csv = "input.csv"
    with open(input_csv, "w") as f:
        f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
        for i, smi in enumerate(smiles_list):
            f.write(f"complex_{i},{os.path.abspath(protein_path)},{smi},\n")

    repo_path = os.path.abspath("DiffDock")
    if not os.path.exists(repo_path):
        print("❌ エラー: カレントディレクトリに 'DiffDock' フォルダが見つかりません。")
        print("実行前に 'git clone https://github.com/gcorso/DiffDock.git' を行ってください。")
        sys.exit(1)

    yaml_path = os.path.join(repo_path, "default_inference_args.yaml")
    with open(yaml_path, 'r') as f: 
        config = yaml.safe_load(f)
        
    config['samples_per_complex'] = n_poses
    
    with open(yaml_path, 'w') as f: 
        yaml.dump(config, f)

    cmd = [
        sys.executable, os.path.join(repo_path, "inference.py"),
        "--config", yaml_path, "--protein_ligand_csv", input_csv,
        "--out_dir", "results", "--inference_steps", "20",
        "--batch_size", "10", "--no_final_step_noise"
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = repo_path + ":" + env.get('PYTHONPATH', '')
    
    # 標準出力をリアルタイムで表示
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout: 
        print(line, end="")
    process.wait()
    
    if process.returncode != 0:
        print("❌ DiffDockの実行中にエラーが発生しました。")
        sys.exit(1)

# ==========================================
# 2. GNINAによる評価とフィルタリング
# ==========================================
def evaluate_results(protein_pdb, smiles_list, min_conf, min_cnn):
    print(f"\n⚖️ 結果を評価中 (閾値: Model Conf >= {min_conf}, CNN Pose Score >= {min_cnn})...")
    
    if not os.path.exists("gnina"):
        print("❌ エラー: カレントディレクトリに 'gnina' 実行ファイルが見つかりません。")
        sys.exit(1)

    results = []

    for i, smi in enumerate(smiles_list):
        complex_dir = f"results/complex_{i}"
        if not os.path.exists(complex_dir):
            continue
            
        sdf_files = glob.glob(os.path.join(complex_dir, "rank*.sdf"))

        for sdf in sdf_files:
            filename = os.path.basename(sdf)
            if "confidence" not in filename: continue
                
            try:
                orig_rank = int(filename.split('rank')[1].split('_')[0])
                conf = float(filename.split('confidence')[1].replace('.sdf', ''))
            except: continue 

            # GNINAでの再スコアリング (score_only)
            cmd = ["./gnina", "-r", protein_pdb, "-l", sdf, "--score_only"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            
            cnn_pose_score = 0.0
            cnn_affinity = 0.0
            vina_score = 0.0
            
            for line in proc.stdout.splitlines():
                if "CNNscore:" in line: cnn_pose_score = float(line.split()[1])
                elif "CNNaffinity:" in line: cnn_affinity = float(line.split()[1])
                elif "Affinity:" in line: vina_score = float(line.split()[1])

            # 🛡️ フィルタリング（足切り）処理
            if conf < min_conf or cnn_pose_score < min_cnn:
                continue

            results.append({
                "Complex ID": f"complex_{i}",
                "SMILES": smi,
                "Orig Rank": orig_rank,
                "Model Conf": conf,
                "CNN Pose Score": cnn_pose_score,
                "Affinity (pK)": cnn_affinity,
                "Vina Score": vina_score,
                "File Path": os.path.join(complex_dir, filename)
            })

    df = pd.DataFrame(results)
    
    if df.empty:
        print("\n⚠️ 警告: フィルタリングの条件が厳しすぎたため、すべてのポーズが除外されました。")
        return

    # CNN Pose Score が高い順（最も物理的に妥当な順）にソート
    df_sorted = df.sort_values(by=["Complex ID", "CNN Pose Score"], ascending=[True, False]).reset_index(drop=True)

    print("\n" + "="*95)
    print(f"🏆 評価結果 (フィルタ通過: {len(df_sorted)} poses)")
    print("="*95)
    print(df_sorted[["Complex ID", "SMILES", "Orig Rank", "Model Conf", "CNN Pose Score", "Affinity (pK)", "Vina Score"]].to_markdown(index=False, floatfmt=".4f"))
    
    # 結果をCSVに保存
    output_csv = "filtered_results.csv"
    df_sorted.to_csv(output_csv, index=False)
    print(f"\n✅ すべての処理が完了しました。結果は '{output_csv}' に保存されています。")

# ==========================================
# メイン処理 (引数パーサー)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffDock + GNINA CUI Pipeline")
    parser.add_argument("-p", required=True, help="標的タンパク質のPDBファイルパス")
    parser.add_argument("-l", required=True, help="リガンドのSMILESが縦に並んだテキストファイルパス")
    parser.add_argument("-dp", type=int, default=20, help="DiffDockでの1リガンドあたりのポーズ生成数 (デフォルト: 20)")
    parser.add_argument("-mc", type=float, default=-2.0, help="min_model_confの閾値 (デフォルト: -2.0)")
    parser.add_argument("-cp", type=float, default=0.3, help="min_cnn_pose_scoreの閾値 (デフォルト: 0.3)")
    
    args = parser.parse_args()

    # 入力ファイルの確認
    if not os.path.exists(args.p):
        print(f"❌ エラー: タンパク質ファイル '{args.p}' が見つかりません。")
        sys.exit(1)
    if not os.path.exists(args.l):
        print(f"❌ エラー: SMILESリストファイル '{args.l}' が見つかりません。")
        sys.exit(1)

    # SMILESの読み込み (空行を無視)
    with open(args.l, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    if not smiles_list:
        print("❌ エラー: SMILESリストファイルが空です。")
        sys.exit(1)

    # 実行
    run_diffdock(args.p, smiles_list, args.dp)
    evaluate_results(args.p, smiles_list, args.mc, args.cp)
