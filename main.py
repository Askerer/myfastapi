import uvicorn
import io
import os
import time
import tempfile
import warnings
from contextlib import asynccontextmanager

# 壓掉 requests 的版本相容性警告（不影響功能）
warnings.filterwarnings("ignore", message="urllib3.*chardet.*charset_normalizer")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

# 跳過模型源檢查，避免網路不通時卡住
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ── 全域 OCR 實例（模型只載入一次）──────────────────────────
ocr_engine: PaddleOCR | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時載入模型，關閉時釋放。"""
    global ocr_engine
    # PaddleOCR 3.x API
    # use_doc_orientation_classify: 文件方向分類（關閉加速）
    # use_doc_unwarping: 文件彎曲矯正（關閉加速）
    # use_textline_orientation: 文字行方向（關閉加速）
    # lang: "ch" 支援中英文混合
    ocr_engine = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        ocr_version="PP-OCRv4",
        enable_mkldnn=False,
    )
    yield
    ocr_engine = None


app = FastAPI(
    title="PaddleOCR API",
    description="上傳圖片，回傳 OCR 辨識結果（PaddleOCR 3.x）",
    version="2.0.0",
    lifespan=lifespan,
)

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/bmp",
    "image/webp",
    "image/tiff",
}


@app.post("/ocr")
async def ocr_recognize(file: UploadFile = File(...)):
    """
    接收一張圖片，回傳辨識出的文字與座標。

    回傳格式:
    ```json
    {
      "filename": "example.png",
      "elapsed_ms": 123.45,
      "results": [
        {
          "text": "辨識文字",
          "confidence": 0.98,
          "polygon": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        }
      ]
    }
    ```
    """
    # ── 驗證檔案類型 ──
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的圖片格式: {file.content_type}，請上傳 png/jpeg/bmp/webp/tiff",
        )

    # ── 讀取圖片並存為暫存檔（PaddleOCR 3.x predict 接受路徑） ──
    try:
        contents = await file.read()
        # 寫入暫存檔，因為 PaddleOCR 3.x 的 predict 對檔案路徑支援最穩定
        suffix = os.path.splitext(file.filename or "img.png")[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法讀取圖片: {e}")

    # ── 執行 OCR（PaddleOCR 3.x API）──
    try:
        start = time.perf_counter()
        results = ocr_engine.predict(input=tmp_path)
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        # 清理暫存檔
        os.unlink(tmp_path)

    # ── 整理結果 ──
    # PaddleOCR 3.x 回傳 list[OCRResult]
    # OCRResult 是 dict-like，包含 rec_texts, rec_scores, rec_polys, dt_polys
    items = []
    for res in results:
        rec_texts = res.get("rec_texts", [])
        rec_scores = res.get("rec_scores", [])
        rec_polys = res.get("rec_polys", [])

        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            # poly 可能是 np.ndarray，轉換成 list
            if hasattr(poly, "tolist"):
                poly = poly.tolist()
            items.append(
                {
                    "text": text,
                    "confidence": round(float(score), 4),
                    "polygon": [[int(p[0]), int(p[1])] for p in poly],
                }
            )

    return JSONResponse(
        content={
            "filename": file.filename,
            "elapsed_ms": round(elapsed_ms, 2),
            "results": items,
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": ocr_engine is not None}


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run("main:app", host="127.0.0.1", port=8088, reload=True)
