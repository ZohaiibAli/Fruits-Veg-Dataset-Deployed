"""Fruit & Veggie Vision - Streamlit app for the trained Fruits_Vegetables CNN classifier."""

import io
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "Fruit_and_Veg_Model.keras"
SAMPLE_DIR = APP_DIR / "Fruits_Vegetables" / "Fruits_Vegetables" / "test"
IMG_HEIGHT, IMG_WIDTH = 180, 180

# Alphabetical order - must match tf.keras.utils.image_dataset_from_directory's class_names
# produced during training (sorted subdirectory names of the train folder).
CLASS_NAMES = [
    "apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum",
    "carrot", "cauliflower", "chilli pepper", "corn", "cucumber", "eggplant",
    "garlic", "ginger", "grapes", "jalepeno", "kiwi", "lemon", "lettuce",
    "mango", "onion", "orange", "paprika", "pear", "peas", "pineapple",
    "pomegranate", "potato", "raddish", "soy beans", "spinach", "sweetcorn",
    "sweetpotato", "tomato", "turnip", "watermelon",
]

# category, emoji, accent color, one-line fun fact per class
CLASS_META = {
    "apple":         ("Fruit",     "🍎", "#e53935", "Apples float because about 25% of their volume is air."),
    "banana":        ("Fruit",     "🍌", "#fdd835", "Bananas are botanically classified as berries."),
    "beetroot":      ("Vegetable", "🫜", "#8e24aa", "Beetroot juice can stain your hands pink for days."),
    "bell pepper":   ("Vegetable", "🫑", "#43a047", "Bell peppers are the only peppers with no capsaicin, so they aren't spicy."),
    "cabbage":       ("Vegetable", "🥬", "#7cb342", "Cabbage is over 90% water by weight."),
    "capsicum":      ("Vegetable", "🫑", "#ef6c00", "Capsicum ripens from green to red, getting sweeter as it ages."),
    "carrot":        ("Vegetable", "🥕", "#fb8c00", "Carrots were originally purple before orange varieties were bred."),
    "cauliflower":   ("Vegetable", "🥦", "#f5f5f5", "Cauliflower is a cabbage whose flower buds never fully develop."),
    "chilli pepper": ("Vegetable", "🌶️", "#d32f2f", "The heat in chillies is measured in Scoville Heat Units."),
    "corn":          ("Vegetable", "🌽", "#fbc02d", "A single ear of corn has an even number of rows, almost always."),
    "cucumber":      ("Vegetable", "🥒", "#558b2f", "Cucumbers are 95% water, making them extremely hydrating."),
    "eggplant":      ("Vegetable", "🍆", "#5e35b1", "Eggplant is botanically a berry, like tomatoes and bananas."),
    "garlic":        ("Vegetable", "🧄", "#efebe9", "Garlic's strong smell comes from sulfur compounds released when it's cut."),
    "ginger":        ("Vegetable", "🫚", "#ffb300", "Ginger has been used as a spice and medicine for over 5,000 years."),
    "grapes":        ("Fruit",     "🍇", "#8e24aa", "It takes about 2.5 pounds of grapes to make one bottle of wine."),
    "jalepeno":      ("Vegetable", "🌶️", "#2e7d32", "Jalapenos rank 2,500-8,000 on the Scoville scale."),
    "kiwi":          ("Fruit",     "🥝", "#9e9d24", "Kiwi fruit has more vitamin C than an orange, gram for gram."),
    "lemon":         ("Fruit",     "🍋", "#fdd835", "Lemons contain roughly 5-6% citric acid, giving them their sour taste."),
    "lettuce":       ("Vegetable", "🥬", "#7cb342", "Lettuce is a member of the daisy family."),
    "mango":         ("Fruit",     "🥭", "#fb8c00", "Mango is the most consumed fruit in the world."),
    "onion":         ("Vegetable", "🧅", "#bf360c", "Chopping onions releases a gas that irritates your tear glands."),
    "orange":        ("Fruit",     "🍊", "#fb8c00", "Oranges were originally green in regions without cold winters."),
    "paprika":       ("Vegetable", "🌶️", "#c62828", "Paprika is made from dried, ground sweet or hot peppers."),
    "pear":          ("Fruit",     "🍐", "#cddc39", "Pears ripen from the inside out, so check the neck for ripeness."),
    "peas":          ("Vegetable", "🫛", "#558b2f", "Peas are one of the oldest cultivated crops, dating back 10,000 years."),
    "pineapple":     ("Fruit",     "🍍", "#f9a825", "A pineapple takes about 18-24 months to grow and ripen."),
    "pomegranate":   ("Fruit",     "🫐", "#ad1457", "A single pomegranate can contain over 600 seeds."),
    "potato":        ("Vegetable", "🥔", "#8d6e63", "Potatoes were the first vegetable grown in space, in 1995."),
    "raddish":       ("Vegetable", "🌱", "#d81b60", "Radishes are ready to harvest in as little as 3-4 weeks."),
    "soy beans":     ("Vegetable", "🫘", "#6d4c41", "Soybeans are a complete protein, containing all 9 essential amino acids."),
    "spinach":       ("Vegetable", "🥬", "#2e7d32", "Spinach loses over half its folate within a week of harvest."),
    "sweetcorn":     ("Vegetable", "🌽", "#fdd835", "Sweetcorn kernels are actually seeds."),
    "sweetpotato":   ("Vegetable", "🍠", "#e64a19", "Sweet potatoes aren't true potatoes - they're a morning glory relative."),
    "tomato":        ("Vegetable", "🍅", "#e53935", "Tomatoes are botanically a fruit but culinarily treated as a vegetable."),
    "turnip":        ("Vegetable", "🤍", "#ad8762", "Turnip greens are edible and packed with vitamin K."),
    "watermelon":    ("Fruit",     "🍉", "#43a047", "Watermelon is about 92% water."),
}

FRUIT_COLOR = "#e53935"
VEG_COLOR = "#43a047"

# --------------------------------------------------------------------------------------
# Page config + styling
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Fruit & Veggie Vision",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f6fff4 0%, #ffffff 35%); }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #ff8a3d 0%, #ff5f6d 35%, #43a047 100%);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }
    .hero h1 { margin: 0; font-size: 2.2rem; }
    .hero p { margin: 0.3rem 0 0 0; font-size: 1.05rem; opacity: 0.95; }

    .result-card {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #eee;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }
    .class-name {
        font-size: 2rem;
        font-weight: 800;
        text-transform: capitalize;
        margin: 0.3rem 0 0.1rem 0;
    }
    .confidence-big {
        font-size: 1.1rem;
        color: #555;
    }
    .fact-box {
        margin-top: 0.8rem;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        background: #fff8e1;
        border-left: 4px solid #fdd835;
        font-size: 0.92rem;
    }
    .mini-card {
        border: 1px solid #eee;
        border-radius: 14px;
        padding: 0.7rem;
        text-align: center;
        background: white;
        transition: transform 0.15s ease;
    }
    .mini-card:hover { transform: translateY(-3px); box-shadow: 0 8px 18px rgba(0,0,0,0.08); }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Backend: model loading + inference
# --------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_resource(show_spinner=False)
def model_summary_text() -> str:
    model = load_model()
    lines = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize + convert to the exact tensor shape the model expects.

    The trained model has an internal `Rescaling(1./255)` layer, so the raw
    0-255 pixel values must be passed in - do not normalize here.
    """
    image = image.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    array = np.array(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict(image: Image.Image) -> dict:
    model = load_model()
    batch = preprocess_image(image)
    logits = model.predict(batch, verbose=0)
    probs = tf.nn.softmax(logits[0]).numpy()
    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    class_name = CLASS_NAMES[top_idx]
    category, emoji, color, fact = CLASS_META[class_name]
    return {
        "class_name": class_name,
        "category": category,
        "emoji": emoji,
        "color": color,
        "fact": fact,
        "confidence": float(probs[top_idx]),
        "probs": probs,
        "order": order,
    }


def random_sample_image() -> Path | None:
    if not SAMPLE_DIR.exists():
        return None
    class_dirs = [d for d in SAMPLE_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        return None
    chosen_dir = random.choice(class_dirs)
    images = list(chosen_dir.glob("*.jpg")) + list(chosen_dir.glob("*.jpeg")) + list(chosen_dir.glob("*.png"))
    if not images:
        return None
    return random.choice(images)


def add_to_history(source_name: str, result: dict) -> None:
    st.session_state.history.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": source_name,
            "prediction": result["class_name"],
            "category": result["category"],
            "confidence": round(result["confidence"] * 100, 2),
        }
    )


if "history" not in st.session_state:
    st.session_state.history = []
if "sample_path" not in st.session_state:
    st.session_state.sample_path = None

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🥦 Fruit & Veggie Vision")
    st.caption("A CNN that tells 36 fruits & vegetables apart from a single photo.")

    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider("Confidence threshold (%)", 0, 100, 50, 5,
                                      help="Predictions below this confidence are flagged as uncertain.")
    top_k = st.slider("Classes to show in breakdown", 3, 10, 5)

    st.markdown("### 🕒 Session history")
    st.metric("Predictions this session", len(st.session_state.history))
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.caption("Model: custom CNN · 180×180 input · 36 classes")
    st.caption(f"Model file found: {'✅' if MODEL_PATH.exists() else '❌ missing'}")

# --------------------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🍎🥕 Fruit & Veggie Vision</h1>
        <p>Upload a photo, snap one with your camera, or try a random sample -
        the model will identify the produce and tell you how confident it is.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at `{MODEL_PATH.name}`. Run the training notebook "
        "(`Fruit_Veg_Model_Training.ipynb`) first to generate it, then restart this app."
    )
    st.stop()

tab_predict, tab_classes, tab_history, tab_about = st.tabs(
    ["🔍 Predict", "📚 Explore Classes", "🕒 History", "ℹ️ About"]
)

# --------------------------------------------------------------------------------------
# Tab: Predict
# --------------------------------------------------------------------------------------

with tab_predict:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown("#### 📤 Give me a photo")
        input_mode = st.radio(
            "Input method", ["Upload image(s)", "Use camera"], horizontal=True, label_visibility="collapsed"
        )

        uploaded_files = []
        camera_file = None

        if input_mode == "Upload image(s)":
            uploaded_files = st.file_uploader(
                "Drop fruit/veg photos here",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True,
            )
        else:
            camera_file = st.camera_input("Take a photo")

        if st.button("🎲 Try a random sample image", use_container_width=True,
                      disabled=not SAMPLE_DIR.exists()):
            st.session_state.sample_path = random_sample_image()

        if not SAMPLE_DIR.exists():
            st.caption("Sample images unavailable - the dataset folder isn't on disk.")

    with right:
        st.markdown("#### 🧠 Result")
        results_placeholder = st.container()

    sources = []  # list of (display_name, PIL.Image)

    if camera_file is not None:
        try:
            sources.append(("camera capture", Image.open(camera_file)))
        except UnidentifiedImageError:
            st.error("Could not read the photo captured from the camera.")

    for f in uploaded_files or []:
        try:
            sources.append((f.name, Image.open(f)))
        except UnidentifiedImageError:
            st.error(f"⚠️ Skipped `{f.name}` - not a valid image file.")

    if st.session_state.sample_path is not None:
        try:
            sources.append((st.session_state.sample_path.name, Image.open(st.session_state.sample_path)))
        except Exception:
            st.session_state.sample_path = None

    with results_placeholder:
        if not sources:
            st.info("Upload a photo, take one with your camera, or try a random sample to get started.")
        else:
            for display_name, image in sources:
                with st.spinner(f"Analyzing {display_name}..."):
                    result = predict(image)
                add_to_history(display_name, result)

                img_col, info_col = st.columns([1, 1.5])
                with img_col:
                    st.image(image, caption=display_name, use_container_width=True)

                with info_col:
                    low_conf = result["confidence"] * 100 < confidence_threshold
                    badge_color = FRUIT_COLOR if result["category"] == "Fruit" else VEG_COLOR

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <span class="badge" style="background:{badge_color}">{result['category']}</span>
                            <div class="class-name">{result['emoji']} {result['class_name']}</div>
                            <div class="confidence-big">Confidence: <b>{result['confidence']*100:.1f}%</b></div>
                            <div class="fact-box">💡 {result['fact']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if low_conf:
                        st.warning(
                            "Low confidence - try a clearer, well-lit, close-up photo of a single item."
                        )
                    elif result["confidence"] > 0.9:
                        st.balloons()

                    top_n = result["order"][:top_k]
                    chart_df = pd.DataFrame(
                        {
                            "class": [CLASS_NAMES[i] for i in top_n],
                            "probability": [result["probs"][i] * 100 for i in top_n],
                        }
                    )
                    colors = [
                        FRUIT_COLOR if CLASS_META[c][0] == "Fruit" else VEG_COLOR
                        for c in chart_df["class"]
                    ]
                    fig = go.Figure(
                        go.Bar(
                            x=chart_df["probability"],
                            y=chart_df["class"],
                            orientation="h",
                            marker_color=colors,
                            text=[f"{v:.1f}%" for v in chart_df["probability"]],
                            textposition="outside",
                        )
                    )
                    fig.update_layout(
                        height=260,
                        margin=dict(l=10, r=30, t=10, b=10),
                        xaxis_title=None,
                        yaxis=dict(autorange="reversed"),
                        xaxis=dict(range=[0, 100]),
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()

# --------------------------------------------------------------------------------------
# Tab: Explore Classes
# --------------------------------------------------------------------------------------

with tab_classes:
    st.markdown("#### 📚 Everything this model can recognize")

    fruit_count = sum(1 for v in CLASS_META.values() if v[0] == "Fruit")
    veg_count = len(CLASS_META) - fruit_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Total classes", len(CLASS_META))
    m2.metric("🍎 Fruits", fruit_count)
    m3.metric("🥦 Vegetables", veg_count)

    filter_col, search_col = st.columns([1, 2])
    with filter_col:
        category_filter = st.radio("Filter", ["All", "Fruit", "Vegetable"], horizontal=True)
    with search_col:
        search_term = st.text_input("Search classes", placeholder="e.g. pepper")

    filtered = [
        name for name in CLASS_NAMES
        if (category_filter == "All" or CLASS_META[name][0] == category_filter)
        and search_term.lower() in name.lower()
    ]

    if not filtered:
        st.info("No classes match that filter/search.")
    else:
        cols = st.columns(6)
        for i, name in enumerate(filtered):
            category, emoji, color, fact = CLASS_META[name]
            with cols[i % 6]:
                st.markdown(
                    f"""
                    <div class="mini-card">
                        <div style="font-size:2rem">{emoji}</div>
                        <div style="font-weight:700; text-transform:capitalize; margin-top:0.2rem">{name}</div>
                        <span class="badge" style="background:{color}; font-size:0.7rem">{category}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(fact)
                st.write("")

# --------------------------------------------------------------------------------------
# Tab: History
# --------------------------------------------------------------------------------------

with tab_history:
    st.markdown("#### 🕒 Your predictions this session")

    if not st.session_state.history:
        st.info("No predictions yet - head to the Predict tab to analyze a photo.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            csv = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download history as CSV",
                data=csv,
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            cat_counts = hist_df["category"].value_counts()
            fig = go.Figure(
                go.Pie(
                    labels=cat_counts.index,
                    values=cat_counts.values,
                    marker_colors=[FRUIT_COLOR if c == "Fruit" else VEG_COLOR for c in cat_counts.index],
                    hole=0.45,
                )
            )
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# Tab: About
# --------------------------------------------------------------------------------------

with tab_about:
    st.markdown("#### ℹ️ About this app")
    st.markdown(
        f"""
        This app serves a convolutional neural network trained from scratch on the
        **Fruits_Vegetables** dataset to classify **{len(CLASS_NAMES)}** different fruits and
        vegetables from a photo.

        **Pipeline**
        - Images resized to **{IMG_WIDTH}×{IMG_HEIGHT}** and fed into the network as raw RGB pixels.
        - The model's first layer rescales pixel values internally (`Rescaling(1/255)`).
        - Three convolution + max-pooling blocks (16 → 32 → 64 filters) extract visual features.
        - A dropout layer, followed by dense layers, produces class logits.
        - Logits are converted to probabilities with softmax; the highest one is the prediction.

        **Tips for best results**
        - Use a single item per photo, centered and well-lit.
        - Avoid heavy backgrounds/clutter - the model was trained mostly on clean, close-up shots.
        """
    )

    with st.expander("🔬 Model architecture summary"):
        st.code(model_summary_text(), language="text")

    with st.expander("🧾 Full class list"):
        st.write(", ".join(sorted(CLASS_NAMES)))
