import os
import shutil
import sys
import subprocess
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dataset dir')
    parser.add_argument('--model', default='models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--cycle_th', type=float, default=3., help='threshold for cycle consistency error')
    parser.add_argument('--chain', action='store_true', help='if chaining cycle consistent flows (optional)')

    args = parser.parse_args()

    # compute raft optical flows between all pairs
    os.chdir('RAFT')
    subprocess.run(['python', 'exhaustive_raft.py', '--data_dir', args.data_dir, '--model', args.model])

    # compute dino feature maps
    os.chdir('../dino')
    subprocess.run(['python', 'extract_dino_features.py', '--data_dir', args.data_dir])

    # filtering
    os.chdir('../RAFT')
    subprocess.run(['python', 'filter_raft.py', '--data_dir', args.data_dir, '--cycle_th', str(args.cycle_th)])

    # chaining (optional)
    subprocess.run(['python', 'chain_raft.py', '--data_dir', args.data_dir])


