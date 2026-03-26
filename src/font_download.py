import os
import requests
import re

SAVE_DIR = "assets/fonts"

# Fonts to download (Google Fonts only)
FONT_FAMILIES = [
    "Roboto",
    "Inter",
    "Open Sans",
    "Lato",
    "Poppins",
    "Noto Sans"
]

# User-Agent required (Google blocks default Python requests sometimes)
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def get_font_css(font_name):
    url_name = font_name.replace(" ", "+")
    url = f"https://fonts.googleapis.com/css2?family={url_name}:wght@100;200;300;400;500;600;700;800;900&display=swap"
    
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"[✗] Failed to fetch CSS for {font_name}")
        return None
    
    return response.text

def extract_font_urls(css_text):
    # Extract all font URLs (woff2 usually)
    urls = re.findall(r'url\((https://[^)]+)\)', css_text)
    return list(set(urls))  # remove duplicates

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"[✓] Exists: {save_path}")
        return

    try:
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(r.content)
            print(f"[✔] Saved: {save_path}")
        else:
            print(f"[✗] Failed: {url}")
    except Exception as e:
        print(f"[!] Error downloading {url}: {e}")

def process_font(font_name):
    print(f"\n=== {font_name} ===")
    
    css = get_font_css(font_name)
    if not css:
        return
    
    urls = extract_font_urls(css)
    
    font_dir = os.path.join(SAVE_DIR, font_name.replace(" ", "_"))
    os.makedirs(font_dir, exist_ok=True)
    
    for i, url in enumerate(urls):
        ext = url.split(".")[-1].split("?")[0]
        filename = f"{font_name.replace(' ', '_')}_{i}.{ext}"
        save_path = os.path.join(font_dir, filename)
        
        download_file(url, save_path)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for font in FONT_FAMILIES:
        process_font(font)

    print("\n✅ Done! Fonts saved in:", SAVE_DIR)

if __name__ == "__main__":
    main()