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


async def _perform_ocr(file: UploadFile):
    """內部輔助函數：執行 OCR 並回傳原始結果與耗時。"""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的圖片格式: {file.content_type}，請上傳 png/jpeg/bmp/webp/tiff",
        )

    try:
        contents = await file.read()
        suffix = os.path.splitext(file.filename or "img.png")[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法讀取圖片: {e}")

    try:
        start = time.perf_counter()
        results = ocr_engine.predict(input=tmp_path)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return results, elapsed_ms
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/ocr")
async def ocr_recognize(file: UploadFile = File(...)):
    """回傳詳細的 OCR 結果（含座標）。"""
    results, elapsed_ms = await _perform_ocr(file)

    items = []
    for res in results:
        rec_texts = res.get("rec_texts", [])
        rec_scores = res.get("rec_scores", [])
        rec_polys = res.get("rec_polys", [])

        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
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


@app.post("/ocr/text")
async def ocr_text_only(file: UploadFile = File(...)):
    """僅回傳辨識出的完整字串。"""
    results, elapsed_ms = await _perform_ocr(file)

    all_texts = []
    for res in results:
        all_texts.extend(res.get("rec_texts", []))

    # 使用空格或換行合併文字，這裡選擇換行
    full_text = "\n".join(all_texts)

    return JSONResponse(
        content={
            "filename": file.filename,
            "elapsed_ms": round(elapsed_ms, 2),
            "text": full_text,
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": ocr_engine is not None}


if __name__ == "__main__":
    import uvicorn

    #uvicorn.run("main:app", host="127.0.0.1", port=8088, reload=True)
