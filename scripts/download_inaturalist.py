#!/usr/bin/env python3
"""
从iNaturalist下载植物图片到unlabeled目录的脚本

用法:
    python scripts/download_inaturalist.py --plants "狗尾草, 紫花地丁, 蒲公英" --max-images 50
    
参数:
    --plants: 植物中文名称列表，用逗号分隔
    --max-images: 每种植物最大下载图片数
    --pages: 从iNaturalist获取的页数
"""

import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.inaturalist_api import get_taxon_id, get_image_urls, download_images

def download_plant_images(plants, max_images=30, pages=5):
    """
    下载指定植物的图片到unlabeled目录
    
    Args:
        plants: 植物名称列表
        max_images: 每种植物最大下载图片数
        pages: 从iNaturalist获取的页数
    """
    # 确保unlabeled目录存在
    unlabeled_dir = 'data/unlabeled'
    os.makedirs(unlabeled_dir, exist_ok=True)
    
    total_downloaded = 0
    
    for plant_name in plants:
        plant_name = plant_name.strip()
        if not plant_name:
            continue
        
        print(f"\n开始处理植物: {plant_name}")
        print("=" * 50)
        
        # 获取taxon_id
        print(f"1. 获取 {plant_name} 的taxon_id...")
        taxon_id = get_taxon_id(plant_name)
        
        if not taxon_id:
            print(f"❌ 未找到植物: {plant_name}")
            continue
        
        print(f"✅ 找到taxon_id: {taxon_id}")
        
        # 获取图片URL
        print(f"2. 获取 {plant_name} 的图片URL...")
        image_urls = get_image_urls(taxon_id, pages=pages)
        
        if not image_urls:
            print(f"❌ 未找到 {plant_name} 的图片")
            continue
        
        print(f"✅ 找到 {len(image_urls)} 个图片URL")
        
        # 下载图片到unlabeled目录
        print(f"3. 下载 {plant_name} 的图片到 {unlabeled_dir}...")
        downloaded = download_images(image_urls, unlabeled_dir, max_images=max_images)
        
        if downloaded > 0:
            print(f"✅ 成功下载 {downloaded} 张 {plant_name} 的图片")
            total_downloaded += downloaded
        else:
            print(f"❌ 下载 {plant_name} 的图片失败")
    
    print(f"\n" + "=" * 50)
    print(f"下载完成！")
    print(f"总共下载了 {total_downloaded} 张图片到 {unlabeled_dir}")
    print("=" * 50)

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='从iNaturalist下载植物图片到unlabeled目录')
    parser.add_argument('--plants', type=str, required=True, help='植物中文名称列表，用逗号分隔')
    parser.add_argument('--max-images', type=int, default=30, help='每种植物最大下载图片数')
    parser.add_argument('--pages', type=int, default=5, help='从iNaturalist获取的页数')
    
    args = parser.parse_args()
    
    # 解析植物列表
    plants = [p.strip() for p in args.plants.split(',')]
    
    print(f"准备下载以下植物的图片: {plants}")
    print(f"每种植物最大下载图片数: {args.max_images}")
    print(f"从iNaturalist获取的页数: {args.pages}")
    
    # 开始下载
    download_plant_images(plants, max_images=args.max_images, pages=args.pages)

if __name__ == '__main__':
    # 添加项目根目录到Python路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
