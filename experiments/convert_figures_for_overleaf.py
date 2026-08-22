#!/usr/bin/env python3
"""Convert all paper figures to Overleaf-safe format (RGB, max 2200px, compressed)."""
import os
from PIL import Image

ANA = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
PKG = '/hd/liujx/microbiome_llm_project/ProCyon_v2/overleaf_package/figures'
os.makedirs(PKG, exist_ok=True)

MAX_SIZE = 2200
figures = [f for f in os.listdir(ANA) if f.endswith('.png')]

for f in sorted(figures):
    src = os.path.join(ANA, f)
    try:
        im = Image.open(src)
        im.load()  # force full load
    except Exception as e:
        print(f'{f}: READ ERROR {e}')
        continue

    # Convert RGBA to RGB on white background
    if im.mode == 'RGBA':
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    elif im.mode != 'RGB':
        im = im.convert('RGB')

    # Downscale if too large
    w, h = im.size
    longest = max(w, h)
    if longest > MAX_SIZE:
        scale = MAX_SIZE / longest
        im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    out = os.path.join(PKG, f)
    im.save(out, 'PNG', optimize=True)
    print(f'{f}: {w}x{h} -> {im.size[0]}x{im.size[1]} RGB')

print('\nDone. All figures converted to Overleaf-safe format.')
