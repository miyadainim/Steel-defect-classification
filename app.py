"""CPU Streamlit inference using the bundled fine-tuned ResNet18 checkpoint."""
import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import CheckpointError, get_checkpoint_path, load_model as _load_model, predict
from steel_defect.deployment.registry import MODELS, DEFAULT_MODEL

DISPLAY_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted surface', 'Rolled-in scale', 'Scratches']


@st.cache_resource
def load_model(checkpoint: str, modified_ns: int, model_name: str):
    """Cache real weights; include file modification time to avoid stale resources."""
    return _load_model(device='cpu', checkpoint=checkpoint, model_name=model_name)


def main():
    st.set_page_config(page_title='Steel Surface Defect Classifier', page_icon='🔬', layout='centered')
    st.title('Steel Surface Defect Classification')
    st.write('Compare five saved six-class classifiers. Fine-tuned ResNet18 is selected by default.')
    st.caption(
        'This model predicts one of six NEU-CLS defect categories. It has not been '
        'validated for production quality control, defect-free classification or other industrial image domains.'
    )
    model_name = st.selectbox('Model', options=list(MODELS), index=list(MODELS).index(DEFAULT_MODEL),
                              format_func=lambda key: MODELS[key]['label'])
    st.caption(f"Selected model: {MODELS[model_name]['label']}")
    if MODELS[model_name]['architecture'] in {'cnn', 'gap'}:
        st.caption('CNN/GAP require 200 x 200 pixel input; grayscale tensors in [0,1], without resizing or normalization.')
    checkpoint = get_checkpoint_path(model_name)
    if not checkpoint.is_file():
        st.info(
            f"Model checkpoint not found for {MODELS[model_name]['label']}. "
            f"Place your compatible local checkpoint at models/{MODELS[model_name]['filename']} after training your own model or obtaining authorized weights. "
            'STEEL_DEFECT_CHECKPOINT can override Fine-tuned ResNet18 only. '
            'Predictions are unavailable until compatible weights are present; no simulated scores are shown.'
        )
        st.file_uploader('Upload a steel surface image', type=['png', 'jpg', 'jpeg', 'bmp'], disabled=True)
        return
    try:
        model = load_model(str(checkpoint.resolve()), checkpoint.stat().st_mtime_ns, model_name)
    except (OSError, CheckpointError, RuntimeError, ValueError):
        st.error(
            'The model checkpoint could not be loaded. Use a trusted, compatible six-class '
            'state dict for the selected architecture and the documented dependencies. Predictions are unavailable; '
            'no replacement model or simulated scores are used.'
        )
        return
    uploaded = st.file_uploader('Upload a steel surface image', type=['png', 'jpg', 'jpeg', 'bmp'])
    if uploaded is None:
        return
    try:
        with Image.open(uploaded) as source:
            image = source.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError):
        st.error('Unable to read this image. Upload a valid PNG, JPEG or BMP file.')
        return
    st.image(image, caption='Uploaded image')
    try:
        index, scores = predict(image, model, device='cpu')
    except ValueError as exc:
        st.error(str(exc))
        return
    except (OSError, RuntimeError):
        st.error('Model inference failed. Check the input image and checkpoint compatibility.')
        return
    st.subheader('Prediction')
    st.success(f'Predicted defect: {DISPLAY_NAMES[index]}')
    st.subheader('Class scores')
    for name, score in sorted(zip(DISPLAY_NAMES, scores), key=lambda item: item[1], reverse=True):
        st.write(f'{name}: {score:.2%}')
        st.progress(float(score))
    st.caption('Scores are softmax outputs, not calibrated probabilities of correctness.')


if __name__ == '__main__':
    main()
