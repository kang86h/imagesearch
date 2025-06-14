import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import glob
import pickle # 캐싱을 위한 라이브러리

# --- 설정 ---
MODEL_NAME = "openai/clip-vit-large-patch14"
IMAGE_DIR = "assets"
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
TOP_N_RESULTS = 6
# 모델에 맞춰 새로운 캐시 파일 이름 정의
CACHE_FILE = "embedding_cache_openai_large.pkl" 

# --- GPU 설정 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 사용하는 장치: {device.upper()}")

# --- 모델 및 프로세서 로드 ---
print(f"🤖 '{MODEL_NAME}' 모델과 프로세서를 로드하는 중입니다...")
try:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print("✅ 모델 준비 완료!")
except Exception as e:
    print(f"🔥 모델 로드 중 오류 발생: {e}")
    print("인터넷 연결을 확인하거나, 모델 이름이 올바른지 확인하세요.")
    exit()

# --- [새로운 함수] assets 폴더의 상태를 가져오는 함수 ---
def get_assets_state(paths):
    """폴더의 현재 상태를 (파일 경로: 최종 수정 시간) 딕셔너리로 반환"""
    # 순서가 보장된 경로 리스트를 기반으로 상태 생성
    return {path: os.path.getmtime(path) for path in paths if os.path.exists(path)}

# --- 이미지 사전 처리 (캐싱 로직 추가) ---
# [!!! 추가 해결책 !!!] 경로 구분자를 '/'로 통일하여 시스템 간 불일치 문제 해결
raw_paths = [p for ext in IMAGE_EXTENSIONS for p in glob.glob(os.path.join(IMAGE_DIR, ext))]
# Windows의 백슬래시(\)를 포워드슬래시(/)로 변환하고, 알파벳 순으로 정렬합니다.
image_paths = sorted([p.replace('\\', '/') for p in raw_paths])

image_embeddings = None
cache_valid = False

if os.path.exists(CACHE_FILE):
    try:
        print(f"ℹ️ 캐시 파일({CACHE_FILE})을 발견했습니다. 유효성을 검사합니다...")
        with open(CACHE_FILE, "rb") as f:
            cached_data = pickle.load(f)
        
        current_state = get_assets_state(image_paths)
        
        if cached_data.get("paths") == image_paths and cached_data.get("state") == current_state:
            print("✅ 유효한 캐시 발견! 이미지 전처리를 건너뜁니다.")
            image_paths = cached_data["paths"]
            image_embeddings = cached_data["embeddings"].to(device)
            cache_valid = True
        else:
            print("⚠️ 캐시가 현재 파일 상태와 다릅니다. 이미지를 새로 처리합니다.")
    except Exception as e:
        print(f"🔥 캐시 파일을 읽는 중 오류 발생: {e}. 이미지를 새로 처리합니다.")

if not cache_valid:
    if image_paths:
        print(f"🖼️ 총 {len(image_paths)}개의 이미지를 사전 처리합니다...")
        embeddings_list, valid_paths = [], []
        with torch.no_grad():
            for path in image_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    embedding = model.get_image_features(**inputs)
                    embeddings_list.append(embedding)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"경고: '{path}' 처리 중 오류: {e}")
        
        if embeddings_list:
            image_paths = valid_paths
            image_embeddings = torch.cat(embeddings_list)
            image_embeddings /= image_embeddings.norm(p=2, dim=-1, keepdim=True)
            print("✅ 이미지 사전 처리 완료!")

            print(f"💾 새로운 전처리 결과를 캐시 파일({CACHE_FILE})에 저장합니다.")
            new_cache_data = {
                "state": get_assets_state(image_paths),
                "paths": image_paths, # 정규화되고 정렬된 경로 저장
                "embeddings": image_embeddings.cpu()
            }
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(new_cache_data, f)
        else:
            image_paths, image_embeddings = [], torch.tensor([])
    else:
        image_paths, image_embeddings = [], torch.tensor([])

if image_embeddings is None or len(image_paths) == 0:
    print(f"경고: '{IMAGE_DIR}' 폴더에 처리할 이미지가 없습니다.")
    image_paths, image_embeddings = [], torch.tensor([])

# --- GUI 애플리케이션 ---
# (이하 코드는 변경할 필요가 없습니다.)
class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CLIP 이미지 검색기")
        self.root.geometry("800x600")

        # 검색 프레임
        search_frame = ttk.Frame(root, padding="10")
        search_frame.pack(fill=tk.X)
        ttk.Label(search_frame, text="검색어 입력:").pack(side=tk.LEFT, padx=5)
        self.search_entry = ttk.Entry(search_frame, width=50)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.search_entry.bind("<Return>", self.search_images)
        self.search_button = ttk.Button(search_frame, text="🔍 검색", command=self.search_images)
        self.search_button.pack(side=tk.LEFT, padx=5)

        # 결과 프레임
        self.result_frame = ttk.Frame(root, padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True)

    def search_images(self, event=None):
        if len(image_paths) == 0:
            print("검색할 이미지가 없습니다.")
            from tkinter import messagebox
            messagebox.showinfo("알림", "assets 폴더에 검색할 이미지가 없습니다.")
            return

        query = self.search_entry.get()
        if not query:
            return

        print(f"\n🔎 검색어 '{query}'로 검색을 시작합니다...")
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        with torch.no_grad():
            text_inputs = processor(text=query, return_tensors="pt").to(device)
            text_embedding = model.get_text_features(**text_inputs)
            text_embedding /= text_embedding.norm(p=2, dim=-1, keepdim=True)

        similarities = (text_embedding @ image_embeddings.T).squeeze(0)
        top_indices = similarities.argsort(descending=True)[:TOP_N_RESULTS]

        print(f"✨ 상위 {len(top_indices)}개 결과:")
        for i, idx in enumerate(top_indices):
            path, score = image_paths[idx], similarities[idx].item()
            print(f"  - {i+1}위: {path} (유사도: {score:.4f})")

            img = Image.open(path)
            img.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(img)

            item_frame = ttk.Frame(self.result_frame)
            img_label = ttk.Label(item_frame, image=photo)
            img_label.image = photo
            img_label.pack()
            score_label = ttk.Label(item_frame, text=f"유사도: {score:.3f}")
            score_label.pack()
            item_frame.grid(row=i // 3, column=i % 3, padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()

