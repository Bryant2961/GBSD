# plot_calibration.py — Figure 5: Error–uncertainty consistency and calibration
"""Usage: python plot_calibration.py --case all"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.calibration import run_from_npz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='all')
    args = parser.parse_args()
    cases = ['Laplace','Burgers_inv','Poisson'] if args.case=='all' else [args.case]
    results = []
    for c in cases:
        m = run_from_npz(c)
        if m: results.append(m)
    if results:
        import pandas as pd
        print('\n  Calibration summary:')
        print(pd.DataFrame(results).to_string(index=False))
