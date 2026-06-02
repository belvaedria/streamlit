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


def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau Excel.")


def validate_input_columns(df):
    missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
    return missing_cols


def prepare_input_data(df):
    input_df = df[FEATURE_NAMES].copy()

    for col in FEATURE_NAMES:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    return input_df


st.title("🧬 Breast Cancer Prediction App")
st.write(
    "Aplikasi ini digunakan untuk memprediksi breast cancer menggunakan beberapa model machine learning."
)

# =========================
# PILIH MODEL
# =========================

st.sidebar.header("Pengaturan Model")

model_files = get_model_files()

if len(model_files) == 0:
    st.error("Belum ada file model di folder `model/`.")
    st.stop()

model_names = [os.path.basename(file) for file in model_files]

selected_model_name = st.sidebar.selectbox(
    "Pilih model machine learning",
    model_names
)

selected_model_path = model_files[model_names.index(selected_model_name)]

try:
    model = load_model(selected_model_path)
    st.sidebar.success(f"Model aktif: {selected_model_name}")
except Exception as e:
    st.sidebar.error(f"Model gagal dimuat: {e}")
    st.stop()


# =========================
# PILIH MENU
# =========================

menu = st.sidebar.radio(
    "Pilih Menu",
    ["Prediksi Manual", "Prediksi CSV/Excel"]
)


# =========================
# MENU 1: PREDIKSI MANUAL
# =========================

if menu == "Prediksi Manual":
    st.header("1. Prediksi Data Manual")
    st.write("Masukkan nilai fitur breast cancer secara manual.")

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
    st.dataframe(input_df, use_container_width=True)

    if st.button("Prediksi Data Manual"):
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
                st.dataframe(proba_df, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")


# =========================
# MENU 2: PREDIKSI CSV/EXCEL
# =========================

elif menu == "Prediksi CSV/Excel":
    st.header("2. Prediksi Banyak Data dari CSV/Excel")
    st.write(
        "Upload file data uji dalam format CSV atau Excel. "
        "File harus memiliki kolom fitur yang sesuai dengan data breast cancer."
    )

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            df = read_uploaded_file(uploaded_file)

            st.subheader("Preview Data Uji")
            st.dataframe(df.head(), use_container_width=True)

            st.write(f"Jumlah data: {df.shape[0]} baris")
            st.write(f"Jumlah kolom: {df.shape[1]} kolom")

            missing_cols = validate_input_columns(df)

            if len(missing_cols) > 0:
                st.error("Kolom pada file belum sesuai dengan fitur yang dibutuhkan model.")

                st.write("Kolom yang belum ada:")
                st.write(missing_cols)

                st.write("Kolom yang harus ada:")
                st.write(FEATURE_NAMES)

            else:
                input_df = prepare_input_data(df)

                if input_df.isnull().sum().sum() > 0:
                    st.error(
                        "Ada nilai kosong atau data non-numerik pada kolom fitur. "
                        "Silakan cek kembali file CSV/Excel."
                    )

                    missing_value_info = input_df.isnull().sum()
                    missing_value_info = missing_value_info[missing_value_info > 0]

                    st.write("Jumlah nilai bermasalah per kolom:")
                    st.write(missing_value_info)

                else:
                    st.success("Format data sudah sesuai. Data siap diprediksi.")

                    st.subheader("Data Fitur yang Digunakan untuk Prediksi")
                    st.dataframe(input_df.head(), use_container_width=True)

                    if st.button("Prediksi Data CSV/Excel"):
                        try:
                            predictions = model.predict(input_df)

                            result_df = df.copy()
                            result_df["prediction"] = predictions
                            result_df["prediction_label"] = [
                                predict_label(pred) for pred in predictions
                            ]

                            if hasattr(model, "predict_proba"):
                                probabilities = model.predict_proba(input_df)

                                if probabilities.shape[1] == 2:
                                    result_df["probability_malignant"] = probabilities[:, 0]
                                    result_df["probability_benign"] = probabilities[:, 1]

                            st.subheader("Hasil Prediksi")
                            st.dataframe(result_df, use_container_width=True)

                            csv_result = result_df.to_csv(index=False).encode("utf-8")

                            st.download_button(
                                label="Download Hasil Prediksi",
                                data=csv_result,
                                file_name="hasil_prediksi_breast_cancer.csv",
                                mime="text/csv"
                            )

                        except Exception as e:
                            st.error(f"Terjadi error saat prediksi: {e}")

        except Exception as e:
            st.error(f"File gagal dibaca: {e}")