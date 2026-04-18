#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
from wsi_core.WholeSlideImage import WholeSlideImage
import h5py
from PIL import Image
import multiprocessing as mp
from functools import partial

def resolve_wsi_path(source_root, slide_id):
    candidates = [
        os.path.join(source_root, slide_id + '.svs'),
        os.path.join(source_root, 'benign', slide_id + '.svs'),
        os.path.join(source_root, 'non_benign', slide_id + '.svs'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def process_slide_compatible(slide_name, source_root, coords_dir, output_dir, patch_level, patch_size, skip_existing=False):
    """处理单个WSI文件，裁剪5x patches"""
    
    slide_path = resolve_wsi_path(source_root, slide_name)
    if slide_path is None:
        print(f"WSI文件不存在: {slide_name} in {source_root}")
        return
    if not os.path.exists(slide_path):
        print(f"WSI文件不存在: {slide_path}")
        return
    
    # 加载坐标文件（create_patches_fp 输出的目录：patches_<patch_size>）
    patch_file = os.path.join(coords_dir, slide_name + '.h5')
    if not os.path.exists(patch_file):
        print(f"坐标文件不存在: {patch_file}")
        return
    
    # 创建保存目录
    slide_save_dir = os.path.join(output_dir, slide_name)
    os.makedirs(slide_save_dir, exist_ok=True)
    
    # 加载WSI
    wsi = WholeSlideImage(slide_path)
    
    # 读取坐标
    with h5py.File(patch_file, 'r') as f:
        coords = f['coords'][:]
    
    print(f"处理 {slide_name}: {len(coords)} patches")

    # 如果选择跳过已完成的 slide -> 检查 output 是否含有相同数目的 patch
    existing_coords = set()
    if skip_existing and os.path.isdir(slide_save_dir):
        for fn in os.listdir(slide_save_dir):
            # 期望文件名格式: <slide>_<patch_size>_<x>_<y>.png
            if not fn.endswith('.png'):
                continue
            # 从右侧分割出 patch_size, x, y
            parts = fn.rsplit('_', 3)
            if len(parts) < 4:
                continue
            try:
                _ps = int(parts[-3])
                x = int(parts[-2])
                y = int(os.path.splitext(parts[-1])[0])
            except Exception:
                continue
            if _ps == patch_size:
                existing_coords.add((x, y))

    # 增强逻辑：区分三种状态
    coords_total = len(coords)
    if skip_existing:
        if len(existing_coords) > coords_total:
            print(f"裁剪异常: {slide_name} 已存在 {len(existing_coords)} 个 patches，但 coords 文件只有 {coords_total} 个。")
            print("请检查输出目录是否包含异常文件名或重复的 patch。脚本将继续裁剪缺失的 patches。")
        elif len(existing_coords) == coords_total:
            print(f"已跳过 {slide_name}（已存在 {len(existing_coords)} 个 patches — 裁剪已正常完成）")
            return
        else:
            print(f"继续裁剪 {slide_name}，已存在 {len(existing_coords)} 个 patches，剩余 {coords_total - len(existing_coords)} 个将被生成。")
    
    # 裁剪patches
    for i, coord in enumerate(coords):
        x, y = coord
        # 如果已有该 patch, 跳过
        if skip_existing and (int(x), int(y)) in existing_coords:
            continue
        patch = wsi.wsi.read_region((x, y), patch_level, (patch_size, patch_size)).convert('RGB')
        patch_name = f"{slide_name}_{patch_size}_{x}_{y}.png"
        patch_path = os.path.join(slide_save_dir, patch_name)
        patch.save(patch_path)
        
        if (i + 1) % 1000 == 0:
            print(f"{slide_name}: 已处理 {i+1}/{len(coords)}")

def main():
    parser = argparse.ArgumentParser(description='从 create_patches_fp.py 输出的 coords 裁剪 patches')
    parser.add_argument('--source', required=True, help='WSI 根目录')
    parser.add_argument('--csv', required=True, help='包含 slide_id 列的 CSV')
    parser.add_argument('--coords-root', required=True, help='create_patches_fp.py 的 save_dir（例如 /.../patches_coords_xin_10x）')
    parser.add_argument('--patch-size', type=int, default=1024, help='patch size（例如 1024）')
    parser.add_argument('--patch-level', type=int, default=1, help='patch level 对应 create_patches_fp 的 patch_level')
    parser.add_argument('--output-root', default=None, help='输出目录（默认: coords_root/patch_images_<patch_size>）')
    parser.add_argument('--workers', type=int, default=1, help='并行 worker 数')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已经完全裁剪过的 slide（检测输出目录）')
    parser.add_argument('--limit', type=int, default=None, help='仅处理前 N 个 slide')
    args = parser.parse_args()
    
    # 读取数据
    df = pd.read_csv(args.csv)
    
    # 使用 create_patches_fp 的输出路径结构: coords_dir = coords_root/patches_<patch_size>
    coords_dir = os.path.join(args.coords_root, 'patches_' + str(args.patch_size))
    if args.output_root is None:
        output_dir = os.path.join(args.coords_root, 'patch_images_' + str(args.patch_size))
    else:
        output_dir = args.output_root
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始裁剪patches (patch_size={args.patch_size}, level={args.patch_level})...")
    print(f"coords 目录: {coords_dir}")
    print(f"输出目录: {output_dir}")
    
    # 处理每个slide
    slide_ids = df['slide_id'].astype(str).tolist()
    if args.limit is not None:
        slide_ids = slide_ids[:args.limit]

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            pool.starmap(partial(process_slide_compatible, source_root=args.source, coords_dir=coords_dir, output_dir=output_dir, patch_level=args.patch_level, patch_size=args.patch_size, skip_existing=args.skip_existing), [(sid,) for sid in slide_ids])
    else:
        for slide_name in slide_ids:
            # 检查benign和non_benign目录
            # Note: make resolve_wsi_path handle these variations
            process_slide_compatible(slide_name, args.source, coords_dir, output_dir, args.patch_level, args.patch_size, skip_existing=args.skip_existing)
    print("patches 裁剪完成!")
    
    print("5x patches裁剪完成!")

if __name__ == "__main__":
    main()
