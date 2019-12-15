#!/usr/bin/env python3
import sys
from tqdm import tqdm

pbar = tqdm(total=int(sys.argv[1]))

for line in sys.stdin:
    pbar.update(1)
