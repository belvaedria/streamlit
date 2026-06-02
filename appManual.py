import streamlit as st
import pandas as pd
import joblib
import os
import glob

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🧬",
    layout="wide"
)

FEATURE_NAMES = [
    "concave points_worst",
    "perimeter_worst",
    "concave points_mean",
    "radius_worst",
    "perimeter_mean",
]

def get_model_files():
    model_folder = "model"

    joblib_files = glob.glob(os.path.join(model_folder, "*.joblib"))
    pkl_files = glob.glob(os.path.join(model_folder, "*.pkl"))

    model_files = joblib_files + pkl_files
    return model_files


@st.cache_resource
def load_model(model_path):
    model = joblib.load(model_path)
    return model


def predict_label(prediction):
    """
    Pada dataset breast cancer dari scikit-learn:
    0 = malignant
    1 = benign
    """
    if prediction == 0:
        return "Malignant / Kanker"
    elif prediction == 1:
        return "Benign / Tidak Kanker"
    else:
        return str(prediction)


st.title("🧬 Breast Cancer Prediction App")
st.write(
    "Aplikasi ini digunakan untuk memprediksi breast cancer "
    "berdasarkan input fitur secara manual."
)

st.header("Pilih Model")

model_files = get_model_files()

if len(model_files) == 0:
    st.error("Belum ada file model di folder `model/`.")
    st.stop()

model_names = [os.path.basename(file) for file in model_files]

selected_model_name = st.selectbox(
    "Pilih model machine learning",
    model_names
)

selected_model_path = model_files[model_names.index(selected_model_name)]

try:
    model = load_model(selected_model_path)
    st.success(f"Model yang digunakan: {selected_model_name}")
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()


st.header("Input Data Manual")

input_data = {}

col1, col2, col3 = st.columns(3)

for i, feature in enumerate(FEATURE_NAMES):
    if i % 3 == 0:
        with col1:
            input_data[feature] = st.number_input(
                label=feature,
                value=0.0,
                format="%.6f"
            )
    elif i % 3 == 1:
        with col2:
            input_data[feature] = st.number_input(
                label=feature,
                value=0.0,
                format="%.6f"
            )
    else:
        with col3:
            input_data[feature] = st.number_input(
                label=feature,
                value=0.0,
                format="%.6f"
            )

input_df = pd.DataFrame([input_data])
input_df = input_df[FEATURE_NAMES]

st.subheader("Data yang Dimasukkan")
st.dataframe(input_df)

if st.button("Prediksi"):
    try:
        prediction = model.predict(input_df)[0]
        prediction_label = predict_label(prediction)

        st.subheader("Hasil Prediksi")
        st.success(f"Hasil prediksi menggunakan {selected_model_name}: {prediction_label}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]

            if hasattr(model, "classes_"):
                class_names = [predict_label(cls) for cls in model.classes_]
            else:
                class_names = [f"Class {i}" for i in range(len(proba))]

            proba_df = pd.DataFrame({
                "Class": class_names,
                "Probability": proba
            })

            st.subheader("Probabilitas Prediksi")
            st.dataframe(proba_df)
        else:
            st.info("Model ini tidak memiliki fungsi `predict_proba`, jadi probabilitas tidak ditampilkan.")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")