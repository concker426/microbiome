#!/usr/bin/env python3
"""Convert the regenerated inductive_bias_figure to Overleaf-safe format."""
from PIL import Image

src = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis/inductive_bias_figure.png'
dst = '/hd/liujx/microbiome_llm_project/ProCyon_v2/overleaf_package/figures/inductive_bias_figure.png'

im = Image.open(src)
im.load()
if im.mode == 'RGBA':
    bg = Image.new('RGB', im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[3])
    im = bg
elif im.mode != 'RGB':
    im = im.convert('RGB')
im.save(dst, 'PNG', optimize=True)
print(f'Converted: {im.size}, mode={im.mode}')
print(f'Saved to: {dst}')
