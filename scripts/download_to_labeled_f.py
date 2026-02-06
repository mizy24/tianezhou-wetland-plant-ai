import os
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置参数 - 使用F盘路径
PROJECT_ROOT = "F:/湿地植物识别ai/python_project/scripts"
LABELED_DIR = os.path.join(PROJECT_ROOT, "data", "labeled")
MAX_IMAGES_PER_PLANT = 25
MAX_RETRIES = 3
TIMEOUT = 30

# 植物列表
PLANTS = [
    {"name": "地黄", "scientific_name": "Rehmannia glutinosa", "rank": "species"},
    {"name": "鹅观草", "scientific_name": "Roegneria kamoji", "rank": "species"},
    {"name": "繁缕", "scientific_name": "Stellaria media", "rank": "species"},
    {"name": "狗尾草", "scientific_name": "Setaria", "rank": "genus"},
    {"name": "狗牙根", "scientific_name": "Cynodon dactylon", "rank": "species"},
    {"name": "红蓼", "scientific_name": "Persicaria", "rank": "genus"},
    {"name": "苔草类", "scientific_name": "Carex", "rank": "genus"},
    {"name": "南荻", "scientific_name": "Triarrhena sacchariflora", "rank": "species"},
    {"name": "蛇床", "scientific_name": "Cnidium monnieri", "rank": "species"},
    {"name": "天胡荽", "scientific_name": "Hydrocotyle", "rank": "genus"},
    {"name": "通泉草", "scientific_name": "Mazus pumilus", "rank": "species"},
    {"name": "紫花地丁", "scientific_name": "Viola philippica", "rank": "species"},
    {"name": "紫云英", "scientific_name": "Astragalus sinicus", "rank": "species"}
]

# 确保目录存在
os.makedirs(LABELED_DIR, exist_ok=True)

def get_existing_files_count(directory):
    """获取目录中已存在的文件数量"""
    if not os.path.exists(directory):
        return 0
    
    # 查找所有jpg文件并提取编号
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    if not files:
        return 0
    
    # 提取文件编号
    numbers = []
    for file in files:
        match = re.match(r'(\d+)\.jpg$', file)
        if match:
            numbers.append(int(match.group(1)))
    
    return max(numbers) if numbers else 0

def download_image(url, save_path, retries=MAX_RETRIES):
    """下载单个图片"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        except Exception as e:
            print(f"  尝试 {attempt+1}/{retries} 失败: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                return False

def search_inaturalist_images(scientific_name, rank, max_images=MAX_IMAGES_PER_PLANT):
    """从iNaturalist搜索图片"""
    base_url = "https://api.inaturalist.org/v1/search"
    image_urls = []
    page = 1
    per_page = 30
    
    while len(image_urls) < max_images:
        params = {
            "q": scientific_name,
            "rank": rank,
            "type": "Taxon",
            "per_page": per_page,
            "page": page
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("results"):
                break
            
            # 获取第一个匹配的分类单元
            taxon = data["results"][0]
            taxon_id = taxon["record"]["id"]
            
            # 搜索该分类单元的观察记录
            observations_url = "https://api.inaturalist.org/v1/observations"
            obs_params = {
                "taxon_id": taxon_id,
                "per_page": per_page,
                "page": page,
                "quality_grade": "research,casual",
                "photo_license": "cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,publicdomain",
                "has_photo": True
            }
            
            obs_response = requests.get(observations_url, params=obs_params, timeout=TIMEOUT)
            obs_response.raise_for_status()
            obs_data = obs_response.json()
            
            if not obs_data.get("results"):
                break
            
            # 提取图片URL
            for obs in obs_data["results"]:
                if len(image_urls) >= max_images:
                    break
                
                for photo in obs.get("photos", []):
                    if len(image_urls) >= max_images:
                        break
                    # 获取中等大小的图片
                    image_url = photo["url"].replace("square", "medium")
                    image_urls.append(image_url)
            
            page += 1
            time.sleep(1)  # 避免API限制
            
        except Exception as e:
            print(f"  搜索图片失败: {str(e)}")
            break
    
    return image_urls

def download_plant_images(plant):
    """下载单个植物的图片"""
    name = plant["name"]
    scientific_name = plant["scientific_name"]
    rank = plant["rank"]
    
    # 创建或使用现有文件夹
    save_dir = os.path.join(LABELED_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取已存在的文件数量
    start_idx = get_existing_files_count(save_dir) + 1
    print(f"开始下载 {name} (学名: {scientific_name}) 的图片...")
    print(f"  保存目录: {save_dir}")
    print(f"  从编号 {start_idx} 开始")
    
    # 搜索图片
    image_urls = search_inaturalist_images(scientific_name, rank)
    print(f"  找到 {len(image_urls)} 张图片")
    
    # 计算需要下载的数量
    remaining = MAX_IMAGES_PER_PLANT - (start_idx - 1)
    if remaining <= 0:
        print(f"  已达到最大图片数量 ({MAX_IMAGES_PER_PLANT} 张)")
        return
    
    # 限制下载数量
    image_urls = image_urls[:remaining]
    print(f"  将下载 {len(image_urls)} 张图片")
    
    # 下载图片
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {}
        
        for i, url in enumerate(image_urls):
            save_path = os.path.join(save_dir, f"{start_idx + i}.jpg")
            future = executor.submit(download_image, url, save_path)
            future_to_url[future] = (url, save_path)
        
        for future in as_completed(future_to_url):
            url, save_path = future_to_url[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                    print(f"  成功: {os.path.basename(save_path)}")
                else:
                    failed_count += 1
                    print(f"  失败: {url}")
            except Exception as e:
                failed_count += 1
                print(f"  错误: {str(e)}")
    
    print(f"{name} 下载完成:")
    print(f"  成功: {success_count} 张")
    print(f"  失败: {failed_count} 张")
    print(f"  总计: {success_count + failed_count} 张")
    print()

def main():
    """主函数"""
    print("=====================================")
    print("植物图片下载脚本")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"保存目录: {LABELED_DIR}")
    print(f"每种植物最大下载数量: {MAX_IMAGES_PER_PLANT}")
    print(f"总计植物种类: {len(PLANTS)}")
    print("=====================================")
    print()
    
    total_success = 0
    total_failed = 0
    
    for plant in PLANTS:
        download_plant_images(plant)
    
    print("=====================================")
    print("所有植物下载完成！")
    print("=====================================")

if __name__ == "__main__":
    main()