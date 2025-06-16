# download_model.py
from transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-large-patch14"
SAVE_DIRECTORY = "./local_clip_model"

if __name__ == "__main__":
    print(f"'{MODEL_NAME}' 모델을 다운로드합니다...")
    
    # 프로세서와 모델 다운로드
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    
    # 지정된 폴더에 저장
    processor.save_pretrained(SAVE_DIRECTORY)
    model.save_pretrained(SAVE_DIRECTORY)
    
    print(f"✅ 모델 파일들이 '{SAVE_DIRECTORY}' 폴더에 성공적으로 저장되었습니다.") 