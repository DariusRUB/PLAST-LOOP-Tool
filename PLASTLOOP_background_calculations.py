# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:00:34 2025

@author: steda22
"""



#import libraries
import os
import sys
import pandas as pd
import numpy as np

# ---- Model families used by the dispatcher -------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression

# ---- Not currently used in the shown code, but commonly used in adjacent GUI/ML workflows --------
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# ---- Metrics + preprocessing + validation --------------------------------------------------------
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_validate



def resource_path(relative_path: str) -> str:
    """
    Return absolute path to resource, works for dev and PyInstaller EXE.

    PURPOSE / LOGIC:
    - When running as a PyInstaller-built executable, resources are extracted into a temporary folder
      and referenced via `sys._MEIPASS`.
    - When running in development (plain .py), resources are relative to the current file.

    RESULT:
    - You can call `resource_path("backgrounddata/file.csv")` and get a usable absolute path in both
      runtime modes.
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)



def readin_data(numberInputparameters, filepath):
    """
    Read experiment data from CSV and return:
      - labdata: full DataFrame
      - inputnames: list of column names interpreted as model inputs

    FUNCTIONAL CONTRACT:
    - CSV is expected to be ';'-separated and German-style decimals ','.
    - The first column (index 0) is treated as "name/id" (not an input).
    - The next `numberInputparameters` columns are the inputs.

    NOTE:
    - This function does not validate missing values, types, or ranges.
    """
    # import the data with the library pandas to
    print(filepath)
    labdata = pd.read_csv(filepath, delimiter=';', encoding='utf-8-sig', decimal=',')
    inputnames = labdata.columns[1:numberInputparameters+1].tolist()
    
    return labdata, inputnames

def _build_design_matrix(model_type: str, inputnames, X_scaled: pd.DataFrame) -> pd.DataFrame:
    """
    Baut die gleiche Feature-Matrix wie die jeweilige _regress_* Funktion.
    X_scaled enthält bereits die skalierten inputnames-Spalten.
    """
    if model_type == "Linear Regression (Scheffé)":
        mixture_names = inputnames[:3]
        process_names = inputnames[3:]

        X = pd.DataFrame(index=X_scaled.index)

        # lineare Mischungsanteile
        for m in mixture_names:
            X[m] = X_scaled[m]

        # Mischung × Prozess
        for m in mixture_names:
            for p in process_names:
                X[f"{m}_{p}"] = X_scaled[m] * X_scaled[p]

        # Prozess × Prozess
        for i in range(len(process_names)):
            for j in range(i + 1, len(process_names)):
                p1 = process_names[i]
                p2 = process_names[j]
                X[f"{p1}_{p2}"] = X_scaled[p1] * X_scaled[p2]

        return X

    # Standardfall: Hauptterme + alle paarweisen Interaktionen
    X = X_scaled.copy()
    for i in range(len(inputnames)):
        for j in range(i + 1, len(inputnames)):
            f1 = inputnames[i]
            f2 = inputnames[j]
            X[f"{f1}_{f2}"] = X_scaled[f1] * X_scaled[f2]
    return X


def _make_estimator(model_type: str, X_feat: pd.DataFrame):
    """
    Erstellt den Estimator mit denselben Defaults wie in deinen _regress_* Funktionen.
    """
    if model_type == "Linear Regression":
        return LinearRegression()

    if model_type == "Linear Regression (Scheffé)":
        return LinearRegression(fit_intercept=False)

    if model_type == "Ridge Regression":
        return Ridge(alpha=1.0, fit_intercept=True)

    if model_type == "Partial Least Squares Regression":
        n_samples, n_features = X_feat.shape
        max_comp = min(n_samples - 1, n_features)
        n_components = min(3, max_comp) if max_comp >= 1 else 1
        return PLSRegression(n_components=n_components)

    if model_type == "Support Vector Regression":
        return SVR(kernel="rbf", C=1.0, epsilon=0.1)

    if model_type == "Decision Tree Regression":
        return DecisionTreeRegressor(random_state=0, max_depth=None)

    raise ValueError(f"Unknown model_type: {model_type}")


def regressModell(labdata, Output, inputnames, filepath, model_type, return_predict_fn=False):

    # --- Validate + coerce numeric ---
    if Output not in labdata.columns:
        raise ValueError(f"Output column '{Output}' not found in dataset.")

    missing_inputs = [c for c in inputnames if c not in labdata.columns]
    if missing_inputs:
        raise ValueError(f"Input columns missing in dataset: {missing_inputs}")

    X_raw = labdata[inputnames].copy()
    X_raw = X_raw.apply(pd.to_numeric, errors="coerce")

    y_series = pd.to_numeric(labdata[Output], errors="coerce")

    if X_raw.isna().any().any():
        bad_cols = X_raw.columns[X_raw.isna().any()].tolist()
        raise ValueError(f"Some input columns contain non-numeric or missing values: {bad_cols}")

    if y_series.isna().any():
        raise ValueError(f"Output column '{Output}' contains non-numeric or missing values.")

    # --- Scale inputs to [-1, 1] ONCE ---
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw),
        columns=inputnames,
        index=X_raw.index
    )

    # --- Dispatch: your _regress_* expect scaled inputs ---
    if model_type == "Linear Regression":
        coefficientsDataFrame, validationVariables = _regress_linear(inputnames, X_scaled, y_series)
    elif model_type == "Linear Regression (Scheffé)":
        coefficientsDataFrame, validationVariables = _regress_scheffe(inputnames, X_scaled, y_series)
    elif model_type == "Ridge Regression":
        coefficientsDataFrame, validationVariables = _regress_ridge(inputnames, X_scaled, y_series)
    elif model_type == "Partial Least Squares Regression":
        coefficientsDataFrame, validationVariables = _regress_pls(inputnames, X_scaled, y_series)
    elif model_type == "Support Vector Regression":
        coefficientsDataFrame, validationVariables = _regress_svr(inputnames, X_scaled, y_series)
    elif model_type == "Decision Tree Regression":
        coefficientsDataFrame, validationVariables = _regress_tree(inputnames, X_scaled, y_series)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if not return_predict_fn:
        return coefficientsDataFrame, validationVariables

    # --- Build a final fitted estimator for prediction (same feature engineering) ---
    X_feat_train = _build_design_matrix(model_type, inputnames, X_scaled)
    est = _make_estimator(model_type, X_feat_train)

    y = y_series.values.astype(float)
    if isinstance(est, PLSRegression):
        y_fit = y.reshape(-1, 1)
    else:
        y_fit = y.ravel()

    est.fit(X_feat_train, y_fit)
    feat_cols = list(X_feat_train.columns)

    def predict_fn(X_df: pd.DataFrame):
        # expects RAW inputs with columns=inputnames
        X_pred_raw = X_df[inputnames].copy()
        X_pred_raw = X_pred_raw.apply(pd.to_numeric, errors="coerce")

        if X_pred_raw.isna().any().any():
            raise ValueError("predict_fn received non-numeric/missing input values.")

        X_scaled_pred = pd.DataFrame(
            scaler.transform(X_pred_raw),
            columns=inputnames,
            index=X_df.index
        )

        X_feat_pred = _build_design_matrix(model_type, inputnames, X_scaled_pred)
        X_feat_pred = X_feat_pred.reindex(columns=feat_cols, fill_value=0.0)

        y_pred = est.predict(X_feat_pred)
        return np.ravel(y_pred)

    return coefficientsDataFrame, validationVariables, predict_fn

def _regress_tree(inputnames, labdatainputs, labdataoutputs):
   
    # target vector
    y = labdataoutputs.values.ravel().astype(float)

    # start with all (scaled) input factors
    X = labdatainputs.copy()

    # add all pairwise interdependencies: A_B = A * B
    for i in range(len(inputnames)):
        for j in range(i + 1, len(inputnames)):
            f1 = inputnames[i]
            f2 = inputnames[j]
            X[f"{f1}_{f2}"] = labdatainputs[f1] * labdatainputs[f2]

    # decision tree model
    tree = DecisionTreeRegressor(random_state=0, max_depth=None)

    # leave-one-out cross validation
    loo = LeaveOneOut()

    rmse_train_list = []   # für Eq. 3 (Fold-RMSE_train)
    r2_train_list = []     # für Eq. 4 (Fold-R2_train)

    y_pred_loo = np.zeros_like(y, dtype=float)  # sammelt ŷ_i für Eq. 6/7

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        tree.fit(X_train, y_train)

        # ---------------------------
        # TEST: speichere ŷ_i (LOO)
        # ---------------------------
        y_pred_test = tree.predict(X_test)
        y_pred_loo[test_idx[0]] = float(y_pred_test[0])

        # ---------------------------
        # TRAIN (im Fold): Eq. 3 & 4
        # ---------------------------
        y_pred_train = tree.predict(X_train)

        # Eq. 3 innerer Term: RMSE_train(i) = sqrt( 1/(n-1) * sum (y_j - ŷ_j,train)^2 )
        rmse_train_i = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        rmse_train_list.append(float(rmse_train_i))

        # Eq. 5: ȳ_train(i)
        ybar_train_i = float(np.mean(y_train))

        # Eq. 4 innerer Term: R2_train(i) = 1 - SSE_train / SST_train
        sse_train_i = float(np.sum((y_train - y_pred_train) ** 2))
        sst_train_i = float(np.sum((y_train - ybar_train_i) ** 2))
        r2_train_i = np.nan if sst_train_i == 0.0 else (1.0 - sse_train_i / sst_train_i)
        r2_train_list.append(float(r2_train_i))

    # ---------------------------
    # Aggregation: Eq. 3 & 4
    # ---------------------------
    averaged_RMSE_train = float(np.mean(rmse_train_list))  # Eq. 3
    averaged_R2_train   = float(np.nanmean(r2_train_list)) # Eq. 4

    # ---------------------------
    # TEST-Metriken aus ŷ_loo: Eq. 6, 7, 8
    # ---------------------------
    # Eq. 8: ȳ
    ybar = float(np.mean(y))

    # SSE_test und SST_test
    sse_test = float(np.sum((y - y_pred_loo) ** 2))
    sst_test = float(np.sum((y - ybar) ** 2))

    # Eq. 6: RMSE_test = sqrt( 1/n * sum (y_i - ŷ_i)^2 )
    RMSE_test = float(np.sqrt(sse_test / len(y)))

    # Eq. 7: R2_test = 1 - SSE_test / SST_test
    R2_test = np.nan if sst_test == 0.0 else float(1.0 - sse_test / sst_test)

    # fit on all data (for importances output)
    tree.fit(X, y)

    # feature importances as "coefficients"
    importances = tree.feature_importances_
    coefficientnames = X.columns.to_numpy()

    coefficientsDataFrame = pd.DataFrame(
        [importances],
        columns=coefficientnames,
        index=["Feature Importances"]
    )

    # include the requested variables for GUI later
    validationVariables = pd.DataFrame(
        {
            "averaged_RMSE_train": [averaged_RMSE_train],  # Eq. 3
            "averaged_R2_train":   [averaged_R2_train],    # Eq. 4
            "RMSE_test":           [RMSE_test],            # Eq. 6
            "R2_test":             [R2_test],              # Eq. 7
        }
    )

    return coefficientsDataFrame, validationVariables

def _regress_svr(inputnames, labdatainputs, labdataoutputs):

    X = labdatainputs.copy()
    y = labdataoutputs.values.ravel().astype(float)

    # add pairwise interaction terms
    for i in range(len(inputnames)):
        for j in range(i + 1, len(inputnames)):
            f1, f2 = inputnames[i], inputnames[j]
            X[f"{f1}_{f2}"] = labdatainputs[f1] * labdatainputs[f2]

    # SVR model
    svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)

    # leave-one-out CV
    loo = LeaveOneOut()

    rmse_train_list = []   # Eq. 3: RMSE_train(i) pro Fold
    r2_train_list = []     # Eq. 4: R2_train(i) pro Fold

    y_pred_loo = np.zeros_like(y, dtype=float)  # sammelt ŷ_i für Eq. 6/7

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svr.fit(X_train, y_train)

        # ---------------------------
        # TEST: out-of-fold ŷ_i (LOO)
        # ---------------------------
        y_pred_test = svr.predict(X_test)
        y_pred_loo[test_idx[0]] = float(y_pred_test[0])

        # ---------------------------
        # TRAIN (im Fold): Eq. 3 & 4
        # ---------------------------
        y_pred_train = svr.predict(X_train)

        # Eq. 3 innerer Term: RMSE_train(i)
        rmse_train_i = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        rmse_train_list.append(float(rmse_train_i))

        # Eq. 5: ybar_train(i)
        ybar_train_i = float(np.mean(y_train))

        # Eq. 4 innerer Term: R2_train(i) = 1 - SSE_train / SST_train
        sse_train_i = float(np.sum((y_train - y_pred_train) ** 2))
        sst_train_i = float(np.sum((y_train - ybar_train_i) ** 2))
        r2_train_i = np.nan if sst_train_i == 0.0 else (1.0 - sse_train_i / sst_train_i)
        r2_train_list.append(float(r2_train_i))

    # ---------------------------
    # Aggregation: Eq. 3 & 4
    # ---------------------------
    averaged_RMSE_train = float(np.mean(rmse_train_list))    # Eq. 3
    averaged_R2_train   = float(np.nanmean(r2_train_list))   # Eq. 4/5

    # ---------------------------
    # TEST-Metriken aus ŷ_loo: Eq. 6, 7, 8
    # ---------------------------
    ybar = float(np.mean(y))                                  # Eq. 8
    sse_test = float(np.sum((y - y_pred_loo) ** 2))
    sst_test = float(np.sum((y - ybar) ** 2))

    RMSE_test = float(np.sqrt(sse_test / len(y)))             # Eq. 6
    R2_test = np.nan if sst_test == 0.0 else float(1.0 - sse_test / sst_test)  # Eq. 7

    # Fit final model on all data (for metadata output)
    svr.fit(X, y)

    # Extract meaningful SVR info (1 row, multiple columns)
    n_support = len(svr.support_)
    frac_support = n_support / len(X)

    # gamma extraction (sklearn stores it differently depending on version)
    if hasattr(svr, "_gamma"):
        gamma_val = svr._gamma
    elif hasattr(svr, "gamma"):
        gamma_val = svr.gamma
    else:
        gamma_val = "unknown"

    coefficientsDataFrame = pd.DataFrame([{
        "Model type": "SVR (nonlinear kernel model)",
        "Kernel": svr.kernel,
        "C": svr.C,
        "Epsilon": svr.epsilon,
        "Gamma": gamma_val,
        "Support vectors (abs)": n_support,
        "Support vectors (%)": f"{frac_support*100:.1f}%"
    }])

    # Validation summary (aligned with Eq. 3–8 and with your tree function)
    validationVariables = pd.DataFrame({
        "averaged_RMSE_train": [averaged_RMSE_train],  # Eq. 3
        "averaged_R2_train":   [averaged_R2_train],    # Eq. 4
        "RMSE_test":           [RMSE_test],            # Eq. 6
        "R2_test":             [R2_test],              # Eq. 7
    })

    return coefficientsDataFrame, validationVariables

def _regress_pls(inputnames, labdatainputs, labdataoutputs):
    

    # target vector (1D)
    y = labdataoutputs.values.ravel().astype(float)

    # start with all (scaled) input factors
    X = labdatainputs.copy()

    # add all pairwise interdependencies: A_B = A * B
    for i in range(len(inputnames)):
        for j in range(i + 1, len(inputnames)):
            f1 = inputnames[i]
            f2 = inputnames[j]
            X[f"{f1}_{f2}"] = labdatainputs[f1] * labdatainputs[f2]

    # choose number of PLS components (not more than samples-1 or features)
    n_samples, n_features = X.shape
    max_comp = min(n_samples - 1, n_features)
    n_components = min(3, max_comp)  # simple default: up to 3 components

    pls = PLSRegression(n_components=n_components)

    # leave-one-out cross validation (manual, aligned to Eq. 3–8)
    loo = LeaveOneOut()

    rmse_train_list = []   # Eq. 3: RMSE_train(i) pro Fold
    r2_train_list = []     # Eq. 4: R2_train(i) pro Fold
    y_pred_loo = np.zeros_like(y, dtype=float)  # ŷ_i out-of-fold für Eq. 6/7

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pls.fit(X_train, y_train)

        # ---------------------------
        # TEST: out-of-fold ŷ_i (LOO)
        # ---------------------------
        y_pred_test = pls.predict(X_test)
        y_pred_loo[test_idx[0]] = float(np.ravel(y_pred_test)[0])

        # ---------------------------
        # TRAIN (im Fold): Eq. 3 & 4
        # ---------------------------
        y_pred_train = pls.predict(X_train)
        y_pred_train = np.ravel(y_pred_train)

        # Eq. 3 innerer Term: RMSE_train(i)
        rmse_train_i = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        rmse_train_list.append(float(rmse_train_i))

        # Eq. 5: ybar_train(i)
        ybar_train_i = float(np.mean(y_train))

        # Eq. 4 innerer Term: R2_train(i) = 1 - SSE_train / SST_train
        sse_train_i = float(np.sum((y_train - y_pred_train) ** 2))
        sst_train_i = float(np.sum((y_train - ybar_train_i) ** 2))
        r2_train_i = np.nan if sst_train_i == 0.0 else (1.0 - sse_train_i / sst_train_i)
        r2_train_list.append(float(r2_train_i))

    # ---------------------------
    # Aggregation: Eq. 3 & 4
    # ---------------------------
    averaged_RMSE_train = float(np.mean(rmse_train_list))     # Eq. 3
    averaged_R2_train = float(np.nanmean(r2_train_list))      # Eq. 4/5

    # ---------------------------
    # TEST: Eq. 6, 7, 8 aus ŷ_loo
    # ---------------------------
    ybar = float(np.mean(y))                                  # Eq. 8
    sse_test = float(np.sum((y - y_pred_loo) ** 2))
    sst_test = float(np.sum((y - ybar) ** 2))

    RMSE_test = float(np.sqrt(sse_test / len(y)))             # Eq. 6
    R2_test = np.nan if sst_test == 0.0 else float(1.0 - sse_test / sst_test)  # Eq. 7

    # fit final model on all data (for coefficients output)
    pls.fit(X, y)

    # coefficients from PLS
    coef = pls.coef_.ravel()

    # compute means directly from current X and y
    x_mean = X.mean(axis=0).values
    y_mean = float(y.mean())

    # intercept so that mean prediction matches mean(y)
    intercept = float(y_mean - x_mean @ coef)

    coefficientnames = X.columns.to_numpy()

    coefficientsDataFrame = pd.DataFrame(
        [[intercept] + list(coef)],
        columns=["Intercept"] + list(coefficientnames),
    )
    coefficientsDataFrame.index = ["Coefficients"]

    # validation summary (aligned across models)
    validationVariables = pd.DataFrame(
        {
            "averaged_RMSE_train": [averaged_RMSE_train],  # Eq. 3
            "averaged_R2_train":   [averaged_R2_train],    # Eq. 4
            "RMSE_test":           [RMSE_test],            # Eq. 6
            "R2_test":             [R2_test],              # Eq. 7
        }
    )

    return coefficientsDataFrame, validationVariables

def _regress_ridge(inputnames, labdatainputs, labdataoutputs):
   

    # target vector (1D)
    y = labdataoutputs.values.ravel().astype(float)

    # start with all (scaled) input factors
    X = labdatainputs.copy()

    # add all pairwise interdependencies: A_B = A * B
    for i in range(len(inputnames)):
        for j in range(i + 1, len(inputnames)):
            f1 = inputnames[i]
            f2 = inputnames[j]
            X[f"{f1}_{f2}"] = labdatainputs[f1] * labdatainputs[f2]

    # ridge regression (with intercept)
    ridge = Ridge(alpha=1.0, fit_intercept=True)

    # leave-one-out cross validation (manual, aligned to Eq. 3–8)
    loo = LeaveOneOut()

    rmse_train_list = []   # Eq. 3: RMSE_train(i) pro Fold
    r2_train_list = []     # Eq. 4: R2_train(i) pro Fold
    y_pred_loo = np.zeros_like(y, dtype=float)  # ŷ_i out-of-fold für Eq. 6/7

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        ridge.fit(X_train, y_train)

        # ---------------------------
        # TEST: out-of-fold ŷ_i (LOO)
        # ---------------------------
        y_pred_test = ridge.predict(X_test)
        y_pred_loo[test_idx[0]] = float(y_pred_test[0])

        # ---------------------------
        # TRAIN (im Fold): Eq. 3 & 4
        # ---------------------------
        y_pred_train = ridge.predict(X_train)

        # Eq. 3 innerer Term: RMSE_train(i)
        rmse_train_i = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        rmse_train_list.append(float(rmse_train_i))

        # Eq. 5: ybar_train(i)
        ybar_train_i = float(np.mean(y_train))

        # Eq. 4 innerer Term: R2_train(i) = 1 - SSE_train / SST_train
        sse_train_i = float(np.sum((y_train - y_pred_train) ** 2))
        sst_train_i = float(np.sum((y_train - ybar_train_i) ** 2))
        r2_train_i = np.nan if sst_train_i == 0.0 else (1.0 - sse_train_i / sst_train_i)
        r2_train_list.append(float(r2_train_i))

    # ---------------------------
    # Aggregation: Eq. 3 & 4
    # ---------------------------
    averaged_RMSE_train = float(np.mean(rmse_train_list))     # Eq. 3
    averaged_R2_train = float(np.nanmean(r2_train_list))      # Eq. 4/5

    # ---------------------------
    # TEST: Eq. 6, 7, 8 aus ŷ_loo
    # ---------------------------
    ybar = float(np.mean(y))                                  # Eq. 8
    sse_test = float(np.sum((y - y_pred_loo) ** 2))
    sst_test = float(np.sum((y - ybar) ** 2))

    RMSE_test = float(np.sqrt(sse_test / len(y)))             # Eq. 6
    R2_test = np.nan if sst_test == 0.0 else float(1.0 - sse_test / sst_test)  # Eq. 7

    # fit final model on all data (for coefficients output)
    ridge.fit(X, y)
    coefficients = np.ravel(ridge.coef_)
    intercept = float(ridge.intercept_)
    coefficientnames = X.columns.to_numpy()

    # put intercept + coefficients into DataFrame (one row)
    coefficientsDataFrame = pd.DataFrame(
        [[intercept] + list(coefficients)],
        columns=["Intercept"] + list(coefficientnames),
    )
    coefficientsDataFrame.index = ["Coefficients"]

    # validation summary (aligned across models)
    validationVariables = pd.DataFrame(
        {
            "averaged_RMSE_train": [averaged_RMSE_train],  # Eq. 3
            "averaged_R2_train":   [averaged_R2_train],    # Eq. 4
            "RMSE_test":           [RMSE_test],            # Eq. 6
            "R2_test":             [R2_test],              # Eq. 7
        }
    )

    return coefficientsDataFrame, validationVariables

def _regress_scheffe(inputnames, labdatainputs, labdataoutputs):
   

    # split the factors into mixture and process factors
    mixture_names = inputnames[:3]      # e.g. ["PP", "PE", "PS"]
    process_names = inputnames[3:]      # e.g. ["T", "Cat", ...]

    # target vector (1D)
    y = labdataoutputs.values.ravel().astype(float)

    # design matrix to fill
    X = pd.DataFrame(index=labdatainputs.index)

    # add linear mixture terms: PP, PE, PS
    for m in mixture_names:
        X[m] = labdatainputs[m]

    # add mixture × process terms: PP_T, PP_Cat, ...
    for m in mixture_names:
        for p in process_names:
            X[f"{m}_{p}"] = labdatainputs[m] * labdatainputs[p]

    # add process × process terms: T_Cat, T_RPM, ...
    for i in range(len(process_names)):
        for j in range(i + 1, len(process_names)):
            p1 = process_names[i]
            p2 = process_names[j]
            X[f"{p1}_{p2}"] = labdatainputs[p1] * labdatainputs[p2]

    # linear regression without intercept (Scheffé form)
    model = LinearRegression(fit_intercept=False)

    # leave-one-out cross validation (manual, aligned to Eq. 3–8)
    loo = LeaveOneOut()

    rmse_train_list = []   # Eq. 3: RMSE_train(i) pro Fold
    r2_train_list = []     # Eq. 4: R2_train(i) pro Fold
    y_pred_loo = np.zeros_like(y, dtype=float)  # ŷ_i out-of-fold für Eq. 6/7

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        # ---------------------------
        # TEST: out-of-fold ŷ_i (LOO)
        # ---------------------------
        y_pred_test = model.predict(X_test)
        y_pred_loo[test_idx[0]] = float(np.ravel(y_pred_test)[0])

        # ---------------------------
        # TRAIN (im Fold): Eq. 3 & 4
        # ---------------------------
        y_pred_train = model.predict(X_train)
        y_pred_train = np.ravel(y_pred_train)

        # Eq. 3: RMSE_train(i)
        rmse_train_i = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        rmse_train_list.append(float(rmse_train_i))

        # Eq. 5: ybar_train(i)
        ybar_train_i = float(np.mean(y_train))

        # Eq. 4: R2_train(i) = 1 - SSE_train / SST_train
        sse_train_i = float(np.sum((y_train - y_pred_train) ** 2))
        sst_train_i = float(np.sum((y_train - ybar_train_i) ** 2))
        r2_train_i = np.nan if sst_train_i == 0.0 else (1.0 - sse_train_i / sst_train_i)
        r2_train_list.append(float(r2_train_i))

    # ---------------------------
    # Aggregation: Eq. 3 & 4
    # ---------------------------
    averaged_RMSE_train = float(np.mean(rmse_train_list))     # Eq. 3
    averaged_R2_train = float(np.nanmean(r2_train_list))      # Eq. 4/5

    # ---------------------------
    # TEST: Eq. 6, 7, 8 aus ŷ_loo
    # ---------------------------
    ybar = float(np.mean(y))                                  # Eq. 8
    sse_test = float(np.sum((y - y_pred_loo) ** 2))
    sst_test = float(np.sum((y - ybar) ** 2))

    RMSE_test = float(np.sqrt(sse_test / len(y)))             # Eq. 6
    R2_test = np.nan if sst_test == 0.0 else float(1.0 - sse_test / sst_test)  # Eq. 7

    # final fit on all data
    model.fit(X, y)
    coefficients = np.ravel(model.coef_)
    coefficientnames = X.columns.to_numpy()

    # put coefficients into DataFrame (one row)
    coefficientsDataFrame = pd.DataFrame(
        [coefficients],
        columns=coefficientnames,
    )
    coefficientsDataFrame.index = ["Coefficients"]

    # validation summary (aligned across models)
    validationVariables = pd.DataFrame(
        {
            "averaged_RMSE_train": [averaged_RMSE_train],  # Eq. 3
            "averaged_R2_train":   [averaged_R2_train],    # Eq. 4
            "RMSE_test":           [RMSE_test],            # Eq. 6
            "R2_test":             [R2_test],              # Eq. 7
        }
    )

    return coefficientsDataFrame, validationVariables

def _regress_linear(inputnames, labdatainputs, labdataoutputs):
    

    # target vector (1D)
    y = labdataoutputs.values.ravel().astype(float)

    # keep your structure
    labdatainputsWithInterdependencies = labdatainputs.copy()

    # add the interdependencies to the input
    for i in range(0, len(inputnames)):
        for j in range(i + 1, len(inputnames)):
            nameOfNewColumn = f"{inputnames[i]}_{inputnames[j]}"
            labdatainputsWithInterdependencies.loc[:, nameOfNewColumn] = (
                labdatainputs[inputnames[i]] * labdatainputs[inputnames[j]]
            )

    X = labdatainputsWithInterdependencies

    model = LinearRegression()

    # leave-one-out CV (manual, aligned to Eq. 3–8)
    loo = LeaveOneOut()

    rmse_train_list = []   # Eq. 3: RMSE_train(i) pro Fold
    r2_train_list = []     # Eq. 4: R2_train(i) pro Fold
    y_pred_loo = np.zeros_like(y, dtype=float)  # ŷ_i out-of-fold für Eq. 6/7

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        # ---------------------------
        # TEST: out-of-fold ŷ_i (LOO)
        # ---------------------------
        y_pred_test = model.predict(X_test)
        y_pred_loo[test_idx[0]] = float(np.ravel(y_pred_test)[0])

        # ---------------------------
        # TRAIN (im Fold): Eq. 3 & 4
        # ---------------------------
        y_pred_train = np.ravel(model.predict(X_train))

        # Eq. 3: RMSE_train(i)
        rmse_train_i = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        rmse_train_list.append(float(rmse_train_i))

        # Eq. 5: ybar_train(i)
        ybar_train_i = float(np.mean(y_train))

        # Eq. 4: R2_train(i) = 1 - SSE_train / SST_train
        sse_train_i = float(np.sum((y_train - y_pred_train) ** 2))
        sst_train_i = float(np.sum((y_train - ybar_train_i) ** 2))
        r2_train_i = np.nan if sst_train_i == 0.0 else (1.0 - sse_train_i / sst_train_i)
        r2_train_list.append(float(r2_train_i))

    # ---------------------------
    # Aggregation: Eq. 3 & 4
    # ---------------------------
    averaged_RMSE_train = float(np.mean(rmse_train_list))      # Eq. 3
    averaged_R2_train = float(np.nanmean(r2_train_list))       # Eq. 4/5

    # ---------------------------
    # TEST: Eq. 6, 7, 8 aus ŷ_loo
    # ---------------------------
    ybar = float(np.mean(y))                                   # Eq. 8
    sse_test = float(np.sum((y - y_pred_loo) ** 2))
    sst_test = float(np.sum((y - ybar) ** 2))

    RMSE_test = float(np.sqrt(sse_test / len(y)))              # Eq. 6
    R2_test = np.nan if sst_test == 0.0 else float(1.0 - sse_test / sst_test)  # Eq. 7

    # final fit on all data for coefficients output
    model.fit(X, y)
    coefficients = np.ravel(model.coef_)
    intercept = float(model.intercept_)
    coefficientnames = X.columns.to_numpy()

    coefficientsDataFrame = pd.DataFrame(
        [[intercept] + list(coefficients)],
        columns=["Intercept"] + list(coefficientnames),
        index=["Coefficients"]
    )

    validationVariables = pd.DataFrame({
        "averaged_RMSE_train": [averaged_RMSE_train],  # Eq. 3
        "averaged_R2_train":   [averaged_R2_train],    # Eq. 4
        "RMSE_test":           [RMSE_test],            # Eq. 6
        "R2_test":             [R2_test],              # Eq. 7
    })

    return coefficientsDataFrame, validationVariables

def estimationOfResults(inputnames, estParameters, filepath):
   
    # import the data with the library pandas to
    labdata = pd.read_csv(filepath, delimiter=';', encoding='utf-8-sig', decimal=',')
    
    # Teake the inputs
    labdatainputs = labdata[inputnames]
    labdataoutputsall = labdata[[col for col in labdata.columns if col not in inputnames and col != labdata.columns[0]]] # lege fest, dass die Outputs alle spalten sind, die nicht inputs sind. Lasse dazu die erste Spaöte aus
    
    # add interdependencies, and therefore normalize the input parameters between 0 and 1
    
    # Initialize the StandardScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Normalize all columns
    labdatainputs.loc[:, inputnames] = scaler.fit_transform(labdatainputs[inputnames])

    # the following line is just to avoid a warning message
    labdatainputsWithInterdependencies = labdatainputs

    labdatainputsWithInterdependencies = labdatainputsWithInterdependencies.copy()
    #add the interdependencies to the input
    for i in range(0, len(inputnames)):
        for j in range(i+1, len(inputnames)):
            nameOfNewColumn = f"{inputnames[i]}_{inputnames[j]}"
            labdatainputsWithInterdependencies.loc[:, nameOfNewColumn] = labdatainputs[inputnames[i]] * labdatainputs[inputnames[j]]

    coefficientnames = labdatainputsWithInterdependencies.columns
    coefficientnames = coefficientnames.to_numpy()
    coefficientsDataFrame = pd.DataFrame([],columns=["Intercept"] + list(coefficientnames))
    
    estimatedResults = pd.DataFrame([],columns =["Prediction of experimental results"])
        
    # let the regression model predict the outputs based on the inputs in the textfields "estParameters"
    
    # First normalize parameters exactly like MinMaxScaler(feature_range=(-1, 1))
    estParametersNormed = []
    estParamCheck = []
    
    for i in range(len(estParameters)):
        maxParam = max(labdata[inputnames[i]])
        minParam = min(labdata[inputnames[i]])
    
        # scale input to [-1, 1]
        # Formula matches MinMaxScaler feature_range=(-1,1) mapping:
        #   x_scaled = (x - min)/(max-min) mapped to [0,1], then to [-1,1].
        scaled = 2 * (estParameters[i] - minParam) / (maxParam - minParam) - 1
        estParametersNormed.append(scaled)
    
        # check if out of training range
        if estParameters[i] < minParam or estParameters[i] > maxParam:
            estParamCheck.append(1)
        else:
            estParamCheck.append(0)

    
    
    #calculate the regression model for all outputs
    for i in range(len(labdataoutputsall.columns)):
        labdataoutput = labdataoutputsall[labdataoutputsall.columns[i]].values
        model = LinearRegression()
        model.fit(labdatainputsWithInterdependencies, labdataoutput)
        coefficients = model.coef_
        intercept = model.intercept_
        coefficients = np.ravel(coefficients)
        coefficientsDataFrame.loc[labdataoutputsall.columns[i]] = [intercept] + list(coefficients)
        
        #now sum up the estimation based on the parameters obtained. Start with intercept 
        total = intercept
        m = len(estParametersNormed)
        # calculate the model predictions of the main effect
        # NOTE: The code assumes coefficient ordering:
        #   [main effects...] followed by [interaction effects...]
        # and uses `m` as the running index into the interaction coefficient block.
        for k in range(len(estParametersNormed)):
            total += estParametersNormed[k] * coefficients[k]
            # calculate the model predictions of the interdependencies
            for j in range(k+1,len(estParametersNormed)):
                total += estParametersNormed[k] * estParametersNormed[j] * coefficients[m]
                m +=1
        estimatedResults.loc[labdataoutputsall.columns[i]] = total
        
    
    
    return estimatedResults, estParamCheck


def visualizationGetData(labdata, inputnames,vizualizationParameters,vizualizationOutputs,vis_values):
    
    # Teake the inputs

    labdatainputs = labdata[inputnames]
    labdataoutputsall = labdata[[col for col in labdata.columns if col not in inputnames and col != labdata.columns[0]]] # lege fest, dass die Outputs alle spalten sind, die nicht inputs sind. Lasse dazu die erste Spaöte aus

    # add interdependencies, and therefore normalize the input parameters between 0 and 1

    # Initialize the StandardScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Normalize all columns
    labdatainputs.loc[:, inputnames] = scaler.fit_transform(labdatainputs[inputnames])

    # the following line is just to avoid a warning message
    labdatainputsWithInterdependencies = labdatainputs

    labdatainputsWithInterdependencies = labdatainputsWithInterdependencies.copy()
    #add the interdependencies to the input
    for i in range(0, len(inputnames)):
        for j in range(i+1, len(inputnames)):
            nameOfNewColumn = f"{inputnames[i]}_{inputnames[j]}"
            labdatainputsWithInterdependencies.loc[:, nameOfNewColumn] = labdatainputs[inputnames[i]] * labdatainputs[inputnames[j]]

    coefficientnames = labdatainputsWithInterdependencies.columns
    coefficientnames = coefficientnames.to_numpy()
    coefficientsDataFrame = pd.DataFrame([],columns=["Intercept"] + list(coefficientnames))
        
        
    #calculate the regression model for all outputs
    for i in range(len(labdataoutputsall.columns)):
        labdataoutput = labdataoutputsall[labdataoutputsall.columns[i]].values
        model = LinearRegression()
        model.fit(labdatainputsWithInterdependencies, labdataoutput)
        coefficients = model.coef_
        intercept = model.intercept_
        coefficients = np.ravel(coefficients)
        coefficientsDataFrame.loc[labdataoutputsall.columns[i]] = [intercept] + list(coefficients)
        
    # select the chosen output vor visualization
    # NOTE: `vizualizationOutputs` appears to be a list-like; `.loc[list]` returns multiple rows.
    selectedRegressmodel = coefficientsDataFrame.loc[vizualizationOutputs]
    p1, p2 = vizualizationParameters            # e.g. "PP", "Cat"
    out_name = vizualizationOutputs[0]          # e.g. "Oil [-]"

    # 1) ranges for the two axes (use raw, unnormalized labdata values)
    n_grid = 60
    p1_vals = np.linspace(labdata[p1].min(), labdata[p1].max(), n_grid)
    p2_vals = np.linspace(labdata[p2].min(), labdata[p2].max(), n_grid)
    P1, P2 = np.meshgrid(p1_vals, p2_vals)

    # 2) build a raw-input grid for ALL inputs (others fixed via vis_values)
    #    start with the fixed values for every grid point
    grid_len = P1.size
    df_inputs_raw = pd.DataFrame(
        {name: np.full(grid_len, np.nan, dtype=float) for name in inputnames}
    )

    # set the two varying params
    df_inputs_raw[p1] = P1.ravel()
    df_inputs_raw[p2] = P2.ravel()

    # set the remaining params from vis_values (ignore any extra keys)
    # This ensures the visualization is a 2D slice through the full feature space.
    for name in set(inputnames) - set([p1, p2]):
        if name not in vis_values:
            raise ValueError(f"Missing fixed value for '{name}' in vis_values.")
        df_inputs_raw[name] = vis_values[name]

    # 3) normalize with the SAME scaler used for training
    X_norm = pd.DataFrame(
        scaler.transform(df_inputs_raw[inputnames]),
        columns=inputnames
    )

    # 4) add interaction terms in the SAME order you used for training
    X_full = X_norm.copy()
    for i in range(len(inputnames)):
        for j in range(i + 1, len(inputnames)):
            name_ij = f"{inputnames[i]}_{inputnames[j]}"
            X_full[name_ij] = X_norm[inputnames[i]] * X_norm[inputnames[j]]

    # 5) evaluate the linear model from `selectedRegressmodel`
    coef_row = selectedRegressmodel.iloc[0]     # Series: Index = ["Intercept"] + features
    intercept = float(coef_row["Intercept"])
    coef_series = coef_row.drop("Intercept")

    # align features safely (fill 0 for any missing-by-accident)
    # This is a defensive step to avoid misalignment if columns differ.
    coef_series = coef_series.reindex(X_full.columns, fill_value=0.0)

    y_pred = intercept + X_full.values.dot(coef_series.values)

    # 6) tidy (long) DataFrame for plotting
    df_plot = pd.DataFrame({
        p1: P1.ravel(),
        p2: P2.ravel(),
        out_name: y_pred
    })
    print(df_plot)
    return df_plot


def getLCAlist():
   
    
    # LCA_values = pd.read_csv("LCIA_values_selection.csv", delimiter=';', encoding='utf-8-sig', decimal=',')
    csv_path = resource_path(os.path.join("backgrounddata", "LCIA_values_selection.csv"))
    LCA_values = pd.read_csv(csv_path, delimiter=';', encoding='utf-8-sig', decimal=',')
    
    
    LCA_list = LCA_values.iloc[:, 0].astype(str).tolist()
    return LCA_list

def calculateLCAForTrials(labdata, linkedParameters):
    
    # LCAdata = pd.read_csv("LCIA_values_selection.csv", delimiter=';', encoding='utf-8-sig', decimal=',')
    csv_path = resource_path(os.path.join("backgrounddata", "LCIA_values_selection.csv"))
    LCAdata = pd.read_csv(csv_path, delimiter=';', encoding='utf-8-sig', decimal=',')
    
    linkedParametersLCAdata = [item[1] for item in linkedParameters.items()]
    linkedParametersDATA = list(linkedParameters.keys())

    LCAresult = pd.DataFrame(columns=["Contributor"])
    LCAresult["Contributor"] = linkedParametersLCAdata

    for j in range(len(labdata)):
        colname = labdata.iloc[j, 0]
        LCAresult[colname] = None
        for i in range(len(LCAresult)):
            val = LCAdata.loc[LCAdata.iloc[:, 0] == LCAresult.loc[i, "Contributor"]].iloc[0, 2] * labdata.loc[j, linkedParametersDATA[i]]
            if "(output as product)" in LCAresult.loc[i, "Contributor"]:
                val = -val
            LCAresult.loc[i, colname] = val
    df = LCAresult   
    
    
    # the rest of this function is only to convert the pandasDataframe to numeric, so that it can be displayed with rounded numbers (from ChatGPT)
    # FUNCTIONAL PURPOSE:
    # - GUI tables often want numeric dtype to format/round.
    # - This block unwraps 0-d numpy arrays and coerces columns to numeric where possible.
    for c in df.columns:
        # If cells are 0-d arrays, unwrap them first
        if df[c].map(lambda x: isinstance(x, np.ndarray) and x.shape == ()).any():
            df[c] = df[c].map(lambda x: x.item() if isinstance(x, np.ndarray) and x.shape == () else x)

        # Try numeric coercion (non-numerics become NaN, which is fine for rounding/formatting)
        coerced = pd.to_numeric(df[c], errors="coerce")
        # If we got at least some numbers, keep the coerced version
        if coerced.notna().any():
            df[c] = coerced
        
    df_num = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")  # start at 2nd col
    df_sum = df_num.sum().to_frame(name="sum")  # column sums (one per DATI_*)    
    dataFullwithLCA = labdata.copy()
    dataFullwithLCA["LCA_sum"] = df_sum["sum"].to_numpy()
    return df, dataFullwithLCA

def calculateLCAForPrediction(estParameters, estParametersValues, estimatedResults, linkedParameters):
   
    # LCAdata = pd.read_csv("LCIA_values_selection.csv", delimiter=';', encoding='utf-8-sig', decimal=',')
    csv_path = resource_path(os.path.join("backgrounddata", "LCIA_values_selection.csv"))
    LCAdata = pd.read_csv(csv_path, delimiter=';', encoding='utf-8-sig', decimal=',')
    
    linkedParametersLCAdata = [item[1] for item in linkedParameters.items()]
    linkedParametersDATA = list(linkedParameters.keys())
    
    labdata = pd.DataFrame(columns=["Name"] + estParameters + estimatedResults.index.tolist())
    labdata.loc[len(labdata)] = ["Prediction"] + estParametersValues + estimatedResults.iloc[:, 0].tolist()
    LCAresult = pd.DataFrame(columns=["Contributor"])
    LCAresult["Contributor"] = linkedParametersLCAdata

    for j in range(len(labdata)):
        colname = labdata.iloc[j, 0]
        LCAresult[colname] = None
        for i in range(len(LCAresult)):
            val = LCAdata.loc[LCAdata.iloc[:, 0] == LCAresult.loc[i, "Contributor"]].iloc[0, 2] * labdata.loc[j, linkedParametersDATA[i]]
            if "(output as product)" in LCAresult.loc[i, "Contributor"]:
                val = -val
            LCAresult.loc[i, colname] = val
    df = LCAresult   

    return df