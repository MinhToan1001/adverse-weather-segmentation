"""Local Streamlit image segmentation demo."""
import hashlib
from io import BytesIO

from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch

from src.data.labels import CLASS_NAMES, PALETTE
from pathlib import Path
from src.inference import SegmentationPredictor
from src.models.registry import MODEL_NAMES


@st.cache_resource(show_spinner=False)
def get_predictor(model_key,checkpoint,weights_mtime):
    torch.set_num_threads(2)
    return SegmentationPredictor(checkpoint,model_name=model_key)


def png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    st.set_page_config(page_title="Phân đoạn cảnh đường phố", page_icon="🚗", layout="wide")
    st.title("Phân đoạn cảnh đường phố")
    st.write("Tải ảnh đường phố và xem mô hình nhận diện từng vùng như đường, xe, người và bầu trời.")
    model_key=st.selectbox("Mô hình",options=list(MODEL_NAMES))
    checkpoint=st.text_input("Thư mục checkpoint đã train (best/)",value=f"outputs/runs/{model_key}/jobs/{model_key}_adapt/run/best")
    weights=Path(checkpoint)/'model.safetensors'
    weights_mtime=weights.stat().st_mtime_ns if weights.is_file() else 0
    st.caption("Chọn checkpoint của lượt chạy mới đã hoàn tất. Ảnh đường phố nhìn từ camera xe phù hợp với dữ liệu train.")
    uploaded = st.file_uploader("Chọn ảnh", type=["png", "jpg", "jpeg", "webp", "bmp"])
    content = uploaded.getvalue() if uploaded is not None else None
    request_key = (model_key,checkpoint,weights_mtime,hashlib.sha256(content).hexdigest()) if content else None
    previous = st.session_state.get("prediction")
    if previous is not None and previous["key"] != request_key:
        st.session_state.pop("prediction")

    if st.button("Phân đoạn ảnh", type="primary", disabled=not content):
        st.session_state.pop("prediction", None)
        try:
            with st.spinner("Đang nạp mô hình và xử lý ảnh…"):
                with Image.open(BytesIO(content)) as image:
                    prediction = get_predictor(model_key,checkpoint,weights_mtime).predict(image)
            st.session_state["prediction"] = {"key": request_key, "result": prediction}
        except (UnidentifiedImageError, Image.DecompressionBombError):
            st.error("Ảnh chưa đọc được. Hãy chọn tệp PNG, JPEG, WebP hoặc BMP hợp lệ.")
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Chưa xử lý được ảnh. {error}")

    saved = st.session_state.get("prediction")
    if saved is not None and saved["key"] == request_key:
        result = saved["result"]
        columns = st.columns(3)
        for column, image, caption in zip(
            columns, (result.image, result.color_mask, result.overlay),
            ("Ảnh đầu vào", "Mask màu", "Overlay"),
        ):
            column.image(image, caption=caption, width="stretch")
        st.metric("Thời gian xử lý ảnh", f"{result.latency_ms:.0f} ms")
        st.caption(f"Thiết bị {result.device.upper()} · Kích thước {result.image.width} × {result.image.height}. "
                   "Thời gian tính từ bước chuẩn bị ảnh đến khi tạo mask và overlay.")
        downloads = st.columns(3)
        for column, image, label, suffix in zip(
            downloads, (result.train_ids, result.color_mask, result.overlay),
            ("Tải mask nhãn", "Tải mask màu", "Tải overlay"),
            ("train_ids", "color_mask", "overlay"),
        ):
            column.download_button(label, png_bytes(image), file_name=f"{model_key}_{suffix}.png",
                                   mime="image/png", on_click="ignore")
        st.caption("Mask nhãn là PNG một kênh với giá trị lớp từ 0 đến 18.")
    else:
        st.info("Chọn ảnh và bấm Phân đoạn ảnh để xem kết quả.")

    with st.expander("Bảng 19 lớp"):
        st.table([{"ID": index, "Lớp": name.replace("_", " "), "RGB": ", ".join(map(str, color))}
                  for index, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE))])


if __name__ == "__main__":
    main()
