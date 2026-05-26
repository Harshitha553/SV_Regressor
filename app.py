import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# --------------------------
# LOGGER
# --------------------------

def log(message):
    timestamp=datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"[{timestamp}] {message}"
    )


# --------------------------
# SESSION STATE
# --------------------------

if "df_clean" not in st.session_state:
    st.session_state.df_clean=None


# --------------------------
# FOLDER SETUP
# --------------------------

BASE_DIR=os.path.dirname(
    os.path.abspath(__file__)
)

RAW_DIR=os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

CLEAN_DIR=os.path.join(
    BASE_DIR,
    "data",
    "cleaned"
)

os.makedirs(
    RAW_DIR,
    exist_ok=True
)

os.makedirs(
    CLEAN_DIR,
    exist_ok=True
)


# --------------------------
# PAGE CONFIG
# --------------------------

st.set_page_config(
    page_title="End-to-End SVM Regressor",
    layout="wide"
)

st.title(
    "🏠 End-to-End SVM Regressor"
)

log(
    "Application Started"
)


# --------------------------
# SIDEBAR SETTINGS
# --------------------------

st.sidebar.header(
    "SVR Settings"
)

kernel=st.sidebar.selectbox(
    "Kernel",
    ["linear","rbf","poly"]
)

C=st.sidebar.slider(
    "Regularization (C)",
    0.1,
    10.0,
    1.0
)

gamma=st.sidebar.selectbox(
    "Gamma",
    ["scale","auto"]
)


# --------------------------
# STEP 1 DATA INGESTION
# --------------------------

st.header(
    "Step 1 : Data Ingestion"
)

option=st.radio(
    "Choose Data Source",
    ["California Dataset",
     "Upload CSV"]
)

df=None

if option=="California Dataset":

    housing=fetch_california_housing()

    df=pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )

    df["Price"]=housing.target

    st.success(
        "California Dataset Loaded"
    )

    log(
        "California dataset loaded"
    )


elif option=="Upload CSV":

    uploaded=st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded:

        df=pd.read_csv(
            uploaded
        )

        st.success(
            "CSV Uploaded Successfully"
        )

        log(
            "CSV Uploaded"
        )


# --------------------------
# STEP 2 EDA
# --------------------------

if df is not None:

    st.header(
        "Step 2 : Exploratory Data Analysis"
    )

    st.dataframe(
        df.head()
    )

    st.write(
        "Shape:",
        df.shape
    )

    st.write(
        "Missing Values"
    )

    st.write(
        df.isnull().sum()
    )

    st.write(
        "Statistical Summary"
    )

    st.write(
        df.describe()
    )


    st.subheader(
        "Correlation Heatmap"
    )

    fig,ax=plt.subplots(
        figsize=(12,8)
    )

    sns.heatmap(
        df.corr(
            numeric_only=True
        ),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(
        fig
    )



# --------------------------
# STEP 3 CLEANING
# --------------------------

if df is not None:

    st.header(
        "Step 3 : Data Cleaning"
    )

    strategy=st.selectbox(
        "Missing value strategy",
        ["Mean",
         "Median",
         "Drop"]
    )

    df_clean=df.copy()

    if strategy=="Drop":

        df_clean=df_clean.dropna()

    else:

        for col in df_clean.select_dtypes(
            include=np.number
        ):

            if strategy=="Mean":

                df_clean[col]=(
                    df_clean[col].fillna(
                        df_clean[col].mean()
                    )
                )

            else:

                df_clean[col]=(
                    df_clean[col].fillna(
                        df_clean[col].median()
                    )
                )

    st.session_state.df_clean=df_clean

    st.success(
        "Data Cleaning Completed"
    )



# --------------------------
# STEP 4 SAVE DATA
# --------------------------

if st.button(
    "Save Cleaned Dataset"
):

    if st.session_state.df_clean is None:

        st.error(
            "No cleaned data available"
        )

    else:

        filename=f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        path=os.path.join(
            CLEAN_DIR,
            filename
        )

        st.session_state.df_clean.to_csv(
            path,
            index=False
        )

        st.success(
            "Dataset Saved Successfully"
        )

        st.info(
            path
        )

        log(
            f"Saved:{path}"
        )


# --------------------------
# STEP 5 LOAD DATA
# --------------------------

st.header(
    "Step 5 : Load Cleaned Dataset"
)

df_model=None

files=os.listdir(
    CLEAN_DIR
)

if files:

    selected=st.selectbox(
        "Select Dataset",
        files
    )

    df_model=pd.read_csv(

        os.path.join(
            CLEAN_DIR,
            selected
        )
    )

    st.success(
        f"Loaded:{selected}"
    )

    st.dataframe(
        df_model.head()
    )

else:

    st.warning(
        "No cleaned datasets found"
    )



# --------------------------
# STEP 6 TRAIN SVR
# --------------------------

st.header(
    "Step 6 : Train SVR"
)

if df_model is not None:

    target=st.selectbox(
        "Select Target Column",
        df_model.columns
    )

    X=df_model.drop(
        columns=[target]
    )

    X=X.select_dtypes(
        include=np.number
    )

    y=df_model[target]


    scaler=StandardScaler()

    X=scaler.fit_transform(
        X
    )


    X_train,X_test,y_train,y_test=(
        train_test_split(

            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )


    model=SVR(

        kernel=kernel,
        C=C,
        gamma=gamma
    )


    model.fit(
        X_train,
        y_train
    )


    y_pred=model.predict(
        X_test
    )


# --------------------------
# METRICS
# --------------------------

    st.header(
        "Model Performance"
    )

    mse=mean_squared_error(
        y_test,
        y_pred
    )

    rmse=np.sqrt(
        mse
    )

    mae=mean_absolute_error(
        y_test,
        y_pred
    )

    r2=r2_score(
        y_test,
        y_pred
    )


    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "MSE",
        round(mse,3)
    )

    c2.metric(
        "RMSE",
        round(rmse,3)
    )

    c3.metric(
        "MAE",
        round(mae,3)
    )

    c4.metric(
        "R² Score",
        round(r2,3)
    )


# --------------------------
# ACTUAL VS PREDICTED
# --------------------------

    st.subheader(
        "Actual vs Predicted"
    )

    fig,ax=plt.subplots()

    ax.scatter(
        y_test,
        y_pred
    )

    ax.set_xlabel(
        "Actual"
    )

    ax.set_ylabel(
        "Predicted"
    )

    st.pyplot(
        fig
    )

else:

    st.info(
        "Please complete previous steps first"
    )
