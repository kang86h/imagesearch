import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import glob

# --- 설정 ---
MODEL_NAME = "openai/clip-vit-large-patch14"
IMAGE_DIR = "assets"  # 이미지가 저장된 폴더 이름
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"] # 검색할 이미지 확장자
TOP_N_RESULTS = 6  # 상위 몇 개의 결과를 보여줄지

# --- GPU 설정 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 사용하는 장치: {device.upper()}")

# --- 모델 및 프로세서 로드 ---
print("🤖 CLIP 모델과 프로세서를 로드하는 중입니다...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("✅ 모델 준비 완료!")

# --- 이미지 사전 처리 ---
image_paths = []
for ext in IMAGE_EXTENSIONS:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

image_embeddings = []
if image_paths:
    print(f"🖼️ 총 {len(image_paths)}개의 이미지를 사전 처리합니다. 잠시 기다려주세요...")
    with torch.no_grad():
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                embedding = model.get_image_features(**inputs)
                image_embeddings.append(embedding)
            except Exception as e:
                print(f"경고: '{path}' 파일을 처리하는 중 오류 발생 - {e}")
    
    # 임베딩들을 하나의 텐서로 결합
    image_embeddings = torch.cat(image_embeddings)
    # 정규화 (유사도 계산 성능 향상)
    image_embeddings /= image_embeddings.norm(p=2, dim=-1, keepdim=True)
    print("✅ 이미지 사전 처리 완료!")
else:
    print(f"경고: '{IMAGE_DIR}' 폴더에서 이미지를 찾을 수 없습니다. 프로그램을 종료합니다.")
    exit()

# --- GUI 애플리케이션 ---
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
        self.search_entry.bind("<Return>", self.search_images) # 엔터 키로 검색

        self.search_button = ttk.Button(search_frame, text="🔍 검색", command=self.search_images)
        self.search_button.pack(side=tk.LEFT, padx=5)

        # 결과 프레임
        self.result_frame = ttk.Frame(root, padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True)

    def search_images(self, event=None):
        query = self.search_entry.get()
        if not query:
            return

        print(f"\n🔎 검색어 '{query}'로 검색을 시작합니다...")

        # 이전 결과 삭제
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        # 텍스트 임베딩 생성
        with torch.no_grad():
            text_inputs = processor(text=query, return_tensors="pt").to(device)
            text_embedding = model.get_text_features(**text_inputs)
            text_embedding /= text_embedding.norm(p=2, dim=-1, keepdim=True)

        # 유사도 계산 (코사인 유사도)
        similarities = (text_embedding @ image_embeddings.T).squeeze(0)
        
        # 유사도 순으로 정렬
        top_indices = similarities.argsort(descending=True)[:TOP_N_RESULTS]

        print(f"✨ 상위 {len(top_indices)}개 결과:")
        # 결과 이미지 표시
        for i, idx in enumerate(top_indices):
            path = image_paths[idx]
            score = similarities[idx].item()
            print(f"  - {i+1}위: {path} (유사도: {score:.4f})")

            img = Image.open(path)
            img.thumbnail((200, 200)) # 썸네일 크기 조절
            photo = ImageTk.PhotoImage(img)

            item_frame = ttk.Frame(self.result_frame)
            img_label = ttk.Label(item_frame, image=photo)
            img_label.image = photo
            img_label.pack()
            
            score_label = ttk.Label(item_frame, text=f"유사도: {score:.3f}")
            score_label.pack()

            item_frame.grid(row=i//3, column=i%3, padx=10, pady=10)
    
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()