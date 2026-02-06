#!/usr/bin/env python3
"""
从iNaturalist使用学名下载植物图片到unlabeled目录的脚本

用法:
    python scripts/download_inaturalist_scientific.py --max-images 5
    
参数:
    --max-images: 每种植物最大下载图片数
    --pages: 从iNaturalist获取的页数
"""

import os
import sys
import argparse
import requests
from PIL import Image
from io import BytesIO

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_taxon_id(scientific_name, rank="species"):
    """
    通过学名获取iNaturalist的taxon_id
    :param scientific_name: 植物学名
    :param rank: 分类等级，默认为species
    :return: taxon_id或None
    """
    try:
        url = "https://api.inaturalist.org/v1/taxa"
        params = {"q": scientific_name, "rank": rank}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("results"):
            return data["results"][0]["id"]
        return None
    except Exception as e:
        print(f"获取taxon_id失败 {scientific_name}: {e}")
        return None

def get_image_urls(taxon_id, pages=5):
    """
    使用taxon_id从iNaturalist获取图片URL
    :param taxon_id: 物种的taxon_id
    :param pages: 获取的页数
    :return: 图片URL列表
    """
    urls = []
    try:
        for page in range(1, pages + 1):
            r = requests.get(
                "https://api.inaturalist.org/v1/observations",
                params={
                    "taxon_id": taxon_id,
                    "per_page": 20,
                    "page": page,
                    "quality_grade": "research",
                    "photo_license": "any",
                    "has_photo": True
                },
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            for obs in data.get("results", []):
                if "photos" in obs:
                    for photo in obs["photos"]:
                        if "url" in photo:
                            # 获取大图URL
                            photo_url = photo["url"]
                            if photo_url:
                                # 替换URL以获取大图
                                large_url = photo_url.replace("square", "large")
                                urls.append(large_url)
        return urls
    except Exception as e:
        print(f"获取图片URL失败: {e}")
        return urls

def download_images(urls, save_dir, max_images=50, max_retries=3):
    """
    下载图片到指定目录
    :param urls: 图片URL列表
    :param save_dir: 保存目录
    :param max_images: 最大下载图片数
    :param max_retries: 最大重试次数
    :return: 下载成功的图片数
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算现有文件数量，从现有数量+1开始编号
    existing_files = len([f for f in os.listdir(save_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')])
    count = 0
    
    print(f"开始下载图片，现有文件数: {existing_files}")
    
    for i, url in enumerate(urls[:max_images]):
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                print(f"下载第 {i+1} 张图片 (尝试 {retry_count+1}/{max_retries}): {url}")
                # 增加超时时间，添加headers模拟浏览器请求
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, timeout=30, headers=headers)
                response.raise_for_status()
                
                # 检查是否为有效图片
                img = Image.open(BytesIO(response.content))
                img.verify()
                
                # 保存图片，从现有数量+1开始编号
                img_path = os.path.join(save_dir, f"{existing_files + count + 1}.jpg")
                with open(img_path, "wb") as f:
                    f.write(response.content)
                
                print(f"✅ 保存成功: {img_path}")
                count += 1
                success = True
                if count >= max_images:
                    break
            except Exception as e:
                retry_count += 1
                print(f"❌ 下载失败: {e}")
                if retry_count >= max_retries:
                    print(f"❌ 已达到最大重试次数，跳过此图片")
                else:
                    print(f"⏳ 等待2秒后重试...")
                    import time
                    time.sleep(2)
                continue
    
    print(f"下载完成，成功下载 {count} 张图片")
    return count

def download_plant_images(plants, max_images=10, pages=5):
    """
    下载指定植物的图片到unlabeled目录
    
    Args:
        plants: 植物字典列表，包含中文名称和学名
        max_images: 每种植物最大下载图片数
        pages: 从iNaturalist获取的页数
    """
    # 确保unlabeled目录存在
    unlabeled_dir = 'data/unlabeled'
    os.makedirs(unlabeled_dir, exist_ok=True)
    
    total_downloaded = 0
    
    for plant in plants:
        chinese_name = plant["chinese"]
        scientific_name = plant["scientific"]
        rank = plant.get("rank", "species")
        
        print(f"\n开始处理植物: {chinese_name} ({scientific_name})")
        print("=" * 60)
        
        # 获取taxon_id
        print(f"1. 获取 {scientific_name} 的taxon_id...")
        taxon_id = get_taxon_id(scientific_name, rank=rank)
        
        if not taxon_id:
            print(f"❌ 未找到植物: {scientific_name}")
            continue
        
        print(f"✅ 找到taxon_id: {taxon_id}")
        
        # 获取图片URL
        print(f"2. 获取 {chinese_name} 的图片URL...")
        image_urls = get_image_urls(taxon_id, pages=pages)
        
        if not image_urls:
            print(f"❌ 未找到 {chinese_name} 的图片")
            continue
        
        print(f"✅ 找到 {len(image_urls)} 个图片URL")
        
        # 下载图片到unlabeled目录
        print(f"3. 下载 {chinese_name} 的图片到 {unlabeled_dir}...")
        downloaded = download_images(image_urls, unlabeled_dir, max_images=max_images, max_retries=3)
        
        if downloaded > 0:
            print(f"✅ 成功下载 {downloaded} 张 {chinese_name} 的图片")
            total_downloaded += downloaded
        else:
            print(f"❌ 下载 {chinese_name} 的图片失败")
    
    print(f"\n" + "=" * 60)
    print(f"所有植物下载完成！")
    print(f"总共下载了 {total_downloaded} 张图片到 {unlabeled_dir}")
    print("=" * 60)

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='从iNaturalist使用学名下载植物图片到unlabeled目录')
    parser.add_argument('--max-images', type=int, default=25, help='每种植物最大下载图片数')
    parser.add_argument('--pages', type=int, default=5, help='从iNaturalist获取的页数')
    
    args = parser.parse_args()
    
    # 用户提供的植物列表
    plants = [
        {"chinese": "地黄", "scientific": "Rehmannia glutinosa", "rank": "species"},
        {"chinese": "鹅观草", "scientific": "Roegneria kamoji", "rank": "species"},
        {"chinese": "繁缕", "scientific": "Stellaria media", "rank": "species"},
        {"chinese": "狗尾草", "scientific": "Setaria", "rank": "genus"},
        {"chinese": "狗牙根", "scientific": "Cynodon dactylon", "rank": "species"},
        {"chinese": "红蓼", "scientific": "Persicaria", "rank": "genus"},
        {"chinese": "苔草类", "scientific": "Carex", "rank": "genus"},
        {"chinese": "南荻", "scientific": "Triarrhena sacchariflora", "rank": "species"},
        {"chinese": "蛇床", "scientific": "Cnidium monnieri", "rank": "species"},
        {"chinese": "天胡荽", "scientific": "Hydrocotyle", "rank": "genus"},
        {"chinese": "通泉草", "scientific": "Mazus pumilus", "rank": "species"},
        {"chinese": "紫花地丁", "scientific": "Viola philippica", "rank": "species"},
        {"chinese": "紫云英", "scientific": "Astragalus sinicus", "rank": "species"}
    ]
    
    print(f"准备下载以下植物的图片:")
    for plant in plants:
        print(f"- {plant['chinese']} ({plant['scientific']})")
    print(f"\n每种植物最大下载图片数: {args.max_images}")
    print(f"从iNaturalist获取的页数: {args.pages}")
    
    # 开始下载
    download_plant_images(plants, max_images=args.max_images, pages=args.pages)

if __name__ == '__main__':
    main()
