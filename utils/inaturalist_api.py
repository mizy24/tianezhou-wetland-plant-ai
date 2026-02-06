import requests
import os
from PIL import Image
from io import BytesIO

def get_taxon_id(name):
    """
    通过植物中文名获取iNaturalist的taxon_id
    :param name: 植物中文名
    :return: taxon_id或None
    """
    try:
        url = "https://api.inaturalist.org/v1/taxa"
        params = {"q": name, "rank": "species", "locale": "zh-CN"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("results"):
            return data["results"][0]["id"]
        return None
    except Exception as e:
        print(f"获取taxon_id失败: {e}")
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

def download_images(urls, save_dir, max_images=50):
    """
    下载图片到指定目录
    :param urls: 图片URL列表
    :param save_dir: 保存目录
    :param max_images: 最大下载图片数
    :return: 下载成功的图片数
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算现有文件数量，从现有数量+1开始编号
    existing_files = len([f for f in os.listdir(save_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')])
    count = 0
    
    print(f"开始下载图片，现有文件数: {existing_files}")
    
    for i, url in enumerate(urls[:max_images]):
        try:
            print(f"下载第 {i+1} 张图片: {url}")
            response = requests.get(url, timeout=15)
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
            if count >= max_images:
                break
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            continue
    
    print(f"下载完成，成功下载 {count} 张图片")
    return count

def get_plant_info(name):
    """
    获取植物的详细信息
    :param name: 植物中文名
    :return: 植物信息字典
    """
    try:
        taxon_id = get_taxon_id(name)
        if not taxon_id:
            return {"error": "未找到该植物"}
        
        # 获取详细信息
        url = f"https://api.inaturalist.org/v1/taxa/{taxon_id}"
        r = requests.get(url, params={"locale": "zh-CN"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if data.get("results"):
            taxon = data["results"][0]
            return {
                "name": name,
                "scientific_name": taxon.get("name", ""),
                "taxon_id": taxon_id,
                "rank": taxon.get("rank", ""),
                "family": taxon.get("family", ""),
                "genus": taxon.get("genus", ""),
                "description": taxon.get("description", {}).get("value", ""),
                "image_urls": get_image_urls(taxon_id, pages=2)
            }
        return {"error": "未找到详细信息"}
    except Exception as e:
        print(f"获取植物信息失败: {e}")
        return {"error": str(e)}
