import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import glob
import pickle

# --- 설정 ---
MODEL_NAME = "openai/clip-vit-large-patch14"
IMAGE_DIR = "assets"
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
INITIAL_LOAD = 20  # 처음 로드할 이미지 수
SCROLL_LOAD = 20   # 스크롤 시 추가 로드할 이미지 수
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
    return {path: os.path.getmtime(path) for path in paths if os.path.exists(path)}

# --- 이미지 사전 처리 (캐싱 로직 추가) ---
raw_paths = [p for ext in IMAGE_EXTENSIONS for p in glob.glob(os.path.join(IMAGE_DIR, ext))]
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
                "paths": image_paths,
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
class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CLIP 이미지 검색기 (v3.0 - Lazy Load & Filter Chain)")
        self.root.geometry("1000x800")

        # --- 검색 상태 관리 ---
        self.active_filters = []
        self.current_image_indices = []
        self.current_scores = torch.tensor([])
        self.filter_rows = []
        self.num_displayed = 0
        self.is_loading = False

        # --- UI 구성 ---
        # 상단 제어 프레임
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(fill=tk.X)

        # 필터들을 담을 영역
        self.filter_area = ttk.Frame(control_frame)
        self.filter_area.pack(fill=tk.X, expand=True, side=tk.LEFT)
        
        # 버튼 영역
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(side=tk.LEFT, padx=(10, 0))
        self.add_filter_button = ttk.Button(buttons_frame, text="✚ 필터 추가", command=self._apply_and_add_filter)
        self.add_filter_button.pack(fill=tk.X, pady=2)
        self.reset_button = ttk.Button(buttons_frame, text="🔄 초기화", command=self.reset_search)
        self.reset_button.pack(fill=tk.X, pady=2)
        
        # 결과 프레임 (스크롤 가능)
        canvas_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.result_frame = ttk.Frame(self.canvas)

        self.result_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.result_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 스크롤 이벤트 바인딩
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Configure>", self._on_scroll)

        # 초기 상태 설정
        self.reset_search()

    def _add_filter_row(self):
        # 이전 필터 입력창 비활성화
        if self.filter_rows:
            last_entry = self.filter_rows[-1]
            last_entry.config(state='disabled')

        # 새 필터 입력창 추가
        row_frame = ttk.Frame(self.filter_area)
        row_frame.pack(fill=tk.X, pady=2)
        
        filter_num = len(self.filter_rows) + 1
        label = ttk.Label(row_frame, text=f"{filter_num}차 필터:")
        label.pack(side=tk.LEFT, padx=(0, 5))
        
        entry = ttk.Entry(row_frame)
        entry.pack(fill=tk.X, expand=True)
        entry.focus()
        
        # 첫 필터는 엔터키로 바로 검색 가능하도록
        if not self.filter_rows:
            entry.bind("<Return>", lambda e: self._apply_and_add_filter(is_first_filter=True))

        self.filter_rows.append(entry)

    def _apply_and_add_filter(self, is_first_filter=False):
        if not self.filter_rows: return
        
        query = self.filter_rows[-1].get()
        if not query.strip():
            messagebox.showinfo("알림", "검색어를 입력하세요.")
            return

        # 검색 실행
        print(f"\n🔎 {len(self.active_filters)+1}차 필터 적용: '{query}'")
        self.active_filters.append(query)
        
        source_indices = list(range(len(image_paths))) if is_first_filter else self.current_image_indices
        
        sorted_indices, sorted_scores = self._perform_search(query, source_indices)
        
        if sorted_indices is None:
            self.active_filters.pop() # 실패 시 필터 원상 복구
            return

        self.current_image_indices = sorted_indices
        self.current_scores = sorted_scores
        
        # 결과 표시 (지연 로딩 시작)
        self.display_results(is_new_search=True)
        
        # 다음 필터 입력을 위한 행 추가
        self._add_filter_row()

    def reset_search(self):
        print("\n🔄 검색 초기화")
        # 모든 필터 위젯 제거
        for widget in self.filter_area.winfo_children():
            widget.destroy()
        
        # 상태 변수 초기화
        self.active_filters.clear()
        self.filter_rows.clear()
        self.current_image_indices = list(range(len(image_paths))) # 초기엔 모든 이미지
        self.current_scores = torch.zeros(len(image_paths))
        
        # 첫 필터 행 추가
        self._add_filter_row()
        
        # 결과창 초기화
        self.display_results(is_new_search=True, show_all=True)

    def _perform_search(self, query, source_indices):
        if not source_indices:
            messagebox.showinfo("알림", "필터링할 이미지가 없습니다.")
            return None, None

        subset_embeddings = image_embeddings[source_indices]
        with torch.no_grad():
            text_inputs = processor(text=query, return_tensors="pt").to(device)
            text_embedding = model.get_text_features(**text_inputs)
            text_embedding /= text_embedding.norm(p=2, dim=-1, keepdim=True)
        
        similarities = (text_embedding @ subset_embeddings.T).squeeze(0)
        sorted_subset_indices = similarities.argsort(descending=True)
        
        final_sorted_original_indices = [source_indices[i] for i in sorted_subset_indices]
        sorted_scores = similarities[sorted_subset_indices]

        return final_sorted_original_indices, sorted_scores
        
    def display_results(self, is_new_search=False, show_all=False):
        if self.is_loading: return
        self.is_loading = True

        if is_new_search:
            for widget in self.result_frame.winfo_children():
                widget.destroy()
            self.num_displayed = 0
            self.canvas.yview_moveto(0)

        offset = self.num_displayed
        limit = len(self.current_image_indices) if show_all else offset + (INITIAL_LOAD if is_new_search else SCROLL_LOAD)
        
        indices_to_show = self.current_image_indices[offset:limit]
        if not indices_to_show:
            if is_new_search: # 새 검색인데 결과가 없는 경우
                ttk.Label(self.result_frame, text="검색 결과가 없습니다.").pack(pady=20)
            self.is_loading = False
            return

        print(f"✨ 결과 표시 중... ( {offset+1} - {offset+len(indices_to_show)} / {len(self.current_image_indices)} )")
        
        is_search_result = len(self.active_filters) > 0 and not show_all

        for i, original_idx in enumerate(indices_to_show):
            path = image_paths[original_idx]
            try:
                img = Image.open(path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)
            except Exception as e:
                print(f"이미지 로드 오류 '{path}': {e}")
                continue

            item_frame = ttk.Frame(self.result_frame)
            img_label = ttk.Label(item_frame, image=photo)
            img_label.image = photo
            img_label.pack()

            if is_search_result:
                score_idx = self.num_displayed + i
                score = self.current_scores[score_idx].item()
                score_label = ttk.Label(item_frame, text=f"유사도: {score:.3f}")
                score_label.pack()

            grid_idx = self.num_displayed + i
            item_frame.grid(row=grid_idx // 4, column=grid_idx % 4, padx=5, pady=5)

        self.num_displayed += len(indices_to_show)
        self.root.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.is_loading = False
    
    def _on_scroll(self, event=None):
        # 스크롤바가 거의 끝에 도달했는지 확인
        top, bottom = self.canvas.yview()
        if bottom > 0.9 and not self.is_loading:
             if self.num_displayed < len(self.current_image_indices):
                self.display_results()

    def _on_mousewheel(self, event):
        # Windows/macOS는 event.delta, Linux는 event.num으로 스크롤 방향 감지
        if event.num == 4: delta = -1 # Linux scroll up
        elif event.num == 5: delta = 1 # Linux scroll down
        else: delta = -1 * (event.delta // 120)
        
        self.canvas.yview_scroll(delta, "units")
        self._on_scroll() # 마우스휠 스크롤 후에도 위치 체크

if __name__ == "__main__":
    if not image_paths:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("오류", f"'{IMAGE_DIR}' 폴더에 이미지가 없습니다.\n프로그램을 종료합니다.")
    else:
        root = tk.Tk()
        app = ImageSearchApp(root)
        root.mainloop()

