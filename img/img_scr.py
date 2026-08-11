import requests
import os
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_all_pages(max_pages=200):
    
    folder = "wine_images_all"
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 Создана папка: {folder}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    base_url = "https://vino-svoe.ru/wines"
    current_url = base_url
    page = 1
    all_images = []
    processed_urls = set()
    
    print(f"🚀 Начинаем скачивание с {base_url}")
    print("-" * 50)
    
    while page <= max_pages:
        print(f"\n📄 Страница {page}")
        print(f"🌐 Загрузка: {current_url}")
        
        try:
            response = requests.get(current_url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Находим все изображения
            images = soup.find_all('img', src=re.compile(r'api\.vino-svoe\.ru'))
            
            if not images:
                print(f"⚠️ Изображения не найдены на странице {page}")
                # Проверяем пагинацию
                pagination = soup.find('ul', class_=re.compile(r'pagination|pages'))
                if pagination:
                    next_link = pagination.find('a', class_=re.compile(r'next|next-page'))
                    if next_link:
                        current_url = urljoin(current_url, next_link.get('href'))
                        page += 1
                        time.sleep(1)
                        continue
                break
            
            new_images = 0
            for img in images:
                img_url = img.get('src')
                if not img_url or img_url.startswith('data:') or img_url in processed_urls:
                    continue
                
                processed_urls.add(img_url)
                
                # Получаем название
                name = img.get('alt') or img.get('title') or ''
                if name:
                    name = re.sub(r'[<>:"/\\|?*]', '', name)
                    name = re.sub(r'\s+', '_', name).strip('_')
                else:
                    # Пробуем найти название рядом
                    parent = img.parent
                    if parent:
                        text = parent.get_text(strip=True)
                        lines = text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and len(line) > 3 and not line.startswith('http'):
                                name = re.sub(r'[<>:"/\\|?*]', '', line)
                                name = re.sub(r'\s+', '_', name).strip('_')
                                break
                
                if not name:
                    name = f"wine_{len(all_images) + 1}"
                
                all_images.append({
                    'url': img_url,
                    'name': name,
                    'page': page
                })
                new_images += 1
            
            print(f"📸 Найдено {new_images} новых изображений на странице {page}")
            print(f"📊 Всего найдено: {len(all_images)} изображений")
            
            # Ищем следующую страницу
            pagination = soup.find('ul', class_=re.compile(r'pagination|pages'))
            next_url = None
            
            if pagination:
                next_link = pagination.find('a', class_=re.compile(r'next|next-page'))
                if not next_link:
                    # Ищем ссылку "Далее"
                    for link in pagination.find_all('a'):
                        text = link.text.strip().lower()
                        if 'далее' in text or 'next' in text or '→' in text:
                            next_link = link
                            break
                
                if next_link:
                    next_url = urljoin(current_url, next_link.get('href'))
            
            if not next_url:
                # Пробуем добавить page= в URL
                if '?' in current_url:
                    if 'page=' in current_url:
                        next_url = re.sub(r'page=\d+', f'page={page + 1}', current_url)
                    else:
                        next_url = f"{current_url}&page={page + 1}"
                else:
                    next_url = f"{current_url}?page={page + 1}"
            
            current_url = next_url
            page += 1
            
            # Задержка между страницами
            time.sleep(1.5)
            
        except Exception as e:
            print(f"❌ Ошибка на странице {page}: {str(e)}")
            break
    
    print("\n" + "=" * 50)
    print(f"📊 Всего обработано страниц: {page - 1}")
    print(f"📸 Всего найдено изображений: {len(all_images)}")
    
    if all_images:
        # Скачиваем изображения
        print("\n📦 Начинаем скачивание...")
        success = 0
        errors = 0
        skipped = 0
        
        for i, img_info in enumerate(all_images, 1):
            filename = f"{img_info['name']}.webp"
            if not filename.endswith(('.webp', '.jpg', '.png')):
                filename += '.webp'
            
            filepath = os.path.join(folder, filename)
            
            if os.path.exists(filepath):
                print(f"⏭️ {i}/{len(all_images)}: {filename} - уже существует")
                skipped += 1
                continue
            
            try:
                print(f"⏳ {i}/{len(all_images)}: {filename}")
                response = requests.get(img_info['url'], headers=headers, timeout=30)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                success += 1
                print(f"✅ {i}/{len(all_images)}: {filename} - скачано")
                time.sleep(0.3)
            except Exception as e:
                errors += 1
                print(f"❌ {i}/{len(all_images)}: {filename} - ошибка: {str(e)}")
        
        print("-" * 50)
        print(f"📊 ИТОГО: {success} скачано, {skipped} пропущено, {errors} ошибок")
    else:
        print("❌ Изображения не найдены")
    
    print(f"📁 Папка: {os.path.abspath(folder)}")

if __name__ == "__main__":
    try:
        import bs4
    except ImportError:
        print("📦 Установка BeautifulSoup...")
        os.system("pip install beautifulsoup4")
    
    # Запускаем со всеми страницами
    download_all_pages(max_pages=200)
