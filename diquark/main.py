import os

import json
import pendulum
import joblib

import numpy as np
import pandas as pd
from tqdm import tqdm

import plotly.graph_objects as go

from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    auc,
    precision_recall_curve,
)

import tensorflow as tf

tfkl = tf.keras.layers
tfk = tf.keras

from diquark.constants import (
    DATA_KEYS,
    PATH_DICT_ATLAS_130_80, 
    CROSS_SECTION_ATLAS_130_80, 
    PATH_DICT_ATLAS_130_65,
    CROSS_SECTION_ATLAS_130_65,
    PATH_DICT_ATLAS_136_80,
    CROSS_SECTION_ATLAS_136_80,
    PATH_DICT_ATLAS_136_65,
    CROSS_SECTION_ATLAS_136_65,
    PATH_DICT_CMS_130_80,
    CROSS_SECTION_CMS_130_80,
    PATH_DICT_CMS_130_65,
    CROSS_SECTION_CMS_130_65,
    PATH_DICT_CMS_136_80,
    CROSS_SECTION_CMS_136_80,
    PATH_DICT_CMS_136_65,
    CROSS_SECTION_CMS_136_65,
    ATLAS_TOTAL_LUMI,
    CMS_TOTAL_LUMI,
)
from diquark.helpers import create_data_dict, mass_score_cut
from diquark.load import read_jet_delphes, lower_cut_suu_mass
from diquark.features import (
    jet_multiplicity,
    leading_jet_arr,
    calculate_delta_r,
    combined_invariant_mass,
    n_jet_invariant_mass,
    n_jet_vector_sum_pt,
)
from diquark.plotting import make_histogram, make_histogram_with_double_gaussian_fit


if os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("..")

configs = {
    "ATLAS_130_80": (PATH_DICT_ATLAS_130_80, CROSS_SECTION_ATLAS_130_80, ATLAS_TOTAL_LUMI),
    "ATLAS_130_65": (PATH_DICT_ATLAS_130_65, CROSS_SECTION_ATLAS_130_65, ATLAS_TOTAL_LUMI),
    "ATLAS_136_80": (PATH_DICT_ATLAS_136_80, CROSS_SECTION_ATLAS_136_80, ATLAS_TOTAL_LUMI),
    "ATLAS_136_65": (PATH_DICT_ATLAS_136_65, CROSS_SECTION_ATLAS_136_65, ATLAS_TOTAL_LUMI),
    "CMS_130_80": (PATH_DICT_CMS_130_80, CROSS_SECTION_CMS_130_80, CMS_TOTAL_LUMI),
    "CMS_130_65": (PATH_DICT_CMS_130_65, CROSS_SECTION_CMS_130_65, CMS_TOTAL_LUMI),
    "CMS_136_80": (PATH_DICT_CMS_136_80, CROSS_SECTION_CMS_136_80, CMS_TOTAL_LUMI),
    "CMS_136_65": (PATH_DICT_CMS_136_65, CROSS_SECTION_CMS_136_65, CMS_TOTAL_LUMI),
}

if __name__ == "__main__":

    for config_key, config in configs.items():
        print(f"Running {config_key}...")
        PATH_DICT = config[0]
        CROSS_SECTION_DICT = config[1]
        TOTAL_LUMI = config[2]

        run_id = pendulum.now().strftime("%Y%m%d%H%M%S")
        workdir = f"models/run_{config_key}_{run_id}"

        if not os.path.exists(workdir):
            os.makedirs(workdir)
            os.makedirs(f"{workdir}/plots")

        datasets = {key: read_jet_delphes(PATH_DICT[key]) for key in tqdm(DATA_KEYS)}
        datasets = {
            key: read_jet_delphes(PATH_DICT[key])
            if not key.startswith("SIG")
            else lower_cut_suu_mass(read_jet_delphes(PATH_DICT[key]), int(config_key.split("_")[-1])*100)
            for key in tqdm(DATA_KEYS)
        }
        jet_multiplicities = {key: jet_multiplicity(ds) for key, ds in tqdm(datasets.items())}
        jet_pts = {key: leading_jet_arr(data, key="Jet/Jet.PT") for key, data in tqdm(datasets.items())}
        jet_etas = {key: leading_jet_arr(data, key="Jet/Jet.Eta") for key, data in tqdm(datasets.items())}
        jet_phis = {key: leading_jet_arr(data, key="Jet/Jet.Phi") for key, data in tqdm(datasets.items())}
        combined_masses = {key: combined_invariant_mass(arr) for key, arr in tqdm(datasets.items())}
        delta_rs = {}
        avg_delta_rs = {}
        m3j_s = {}
        m3j_m6j = {}
        m2j_s = {}
        m2j_m6j = {}
        max_delta_rs = {}
        smallest_delta_r_masses = {}
        n_jet_pairs_near_W_mass = {}
        max_vector_sum_pt = {}
        max_vector_sum_pt_delta_r = {}

        for key, data in tqdm(datasets.items()):
            etas = leading_jet_arr(data, 6, key="Jet/Jet.Eta")
            phis = leading_jet_arr(data, 6, key="Jet/Jet.Phi")
            pts = leading_jet_arr(data, 6, key="Jet/Jet.PT")

            # Calculate ΔR for each pair of jets
            delta_rs[key] = calculate_delta_r(etas, phis, pts)
            avg_delta_rs[key] = np.mean(delta_rs[key], where=delta_rs[key] != 0)
            max_delta_rs[key] = np.max(delta_rs[key], axis=1)

            # Calculate invariant masses for 3-jet combinations
            m3j_s[key] = n_jet_invariant_mass(data, n=6, k=3)
            m3j_m6j[key] = np.divide(
                m3j_s[key].mean(axis=-1, where=m3j_s[key] != 0),
                combined_masses[key],
                out=np.zeros_like(combined_masses[key]),
                where=combined_masses[key] != 0,
            )

            # Calculate invariant masses for 2-jet combinations
            m2j_s[key] = n_jet_invariant_mass(data, n=6, k=2)
            m2j_m6j[key] = np.divide(
                m2j_s[key].mean(axis=-1, where=m2j_s[key] != 0),
                combined_masses[key],
                out=np.zeros_like(combined_masses[key]),
                where=combined_masses[key] != 0,
            )

            # Find the mass of the jet pair with the smallest ΔR
            smallest_delta_r_indices = np.argmin(delta_rs[key], axis=1)
            smallest_delta_r_masses[key] = np.choose(smallest_delta_r_indices, m2j_s[key].T)

            # Count jet pairs within 20 GeV of the W mass
            n_jet_pairs_near_W_mass[key] = np.sum((m2j_s[key] >= 60) & (m2j_s[key] <= 100), axis=1)

            # Calculate vector sum pT for 2-jet combinations
            vector_sum_pts = n_jet_vector_sum_pt(data, n=6, k=2)
            max_vector_sum_pt[key] = np.max(vector_sum_pts, axis=1)

            # Indices of the jet pairs (flat index across the combination matrix)
            jet_pair_indices = np.argmax(vector_sum_pts, axis=1)

            # calculate the ΔR between the two jets with the largest vector sum pT
            max_vector_sum_pt_delta_r[key] = np.choose(jet_pair_indices, delta_rs[key].T)
        ds = create_data_dict(
            **{
                "multiplicity": jet_multiplicities,
                "delta_R": delta_rs,
                "m3j": m3j_s,
                "m2j_s": m2j_s,
                "inv_mass": combined_masses,
                "m3j_m6j": m3j_m6j,
                "m2j_m6j": m2j_m6j,
                "pt": jet_pts,
                "eta": jet_etas,
                "phi": jet_phis,
                "max_delta_R": max_delta_rs,
                "m2j_min_delta_R": smallest_delta_r_masses,
                "nj_mW_pm20": n_jet_pairs_near_W_mass,
                "max_vector_sum_pt": max_vector_sum_pt,
                "max_vector_sum_pt_delta_r": max_vector_sum_pt_delta_r,
            }
        )
        df = pd.DataFrame(ds).fillna(0)
        df["target"] = df["Truth"].apply(lambda x: 1 if "SIG" in x else 0)
        df.to_parquet(f"{workdir}/full_sample.parquet", index=False)

        # Separate signal and background events
        df_bkg = df[df["target"] == 0]
        df_sig = df[df["target"] == 1]

        # Split the background events
        df_bkg_train = df_bkg.sample(frac=0.8, random_state=0)
        df_bkg_test = df_bkg.drop(df_bkg_train.index)

        # Separate the signal events
        # df_sig_train = df_sig.sample(frac=0.8, random_state=0)
        # df_sig_test = df_sig.drop(df_sig_train.index)
        df_sig_test = df_sig.sample(n=len(df_bkg_test) // df_bkg["Truth"].nunique(), random_state=0)
        df_sig_train = df_sig.drop(df_sig_test.index)

        ## CROSS SECTION FIX:
        CROSS_SECTION_DICT["SIG:Suu"] = CROSS_SECTION_DICT["SIG:Suu"] * len(df_sig)/(len(df_bkg) // df_bkg["Truth"].nunique())
 
        # Oversample the signal class in the training set to match the number of background instances
        df_sig_train_oversampled = resample(
            df_sig_train,
            replace=True,  # sample with replacement
            n_samples=len(df_bkg_train),  # match number in majority class
            random_state=0,
        )  # reproducible results

        # Combine the oversampled signal class with the background class to form the training set
        df_train = pd.concat([df_sig_train_oversampled, df_bkg_train])

        # Combine signal and background test sets
        df_test = pd.concat([df_sig_test, df_bkg_test])

        # Shuffle the training and test sets
        df_train = shuffle(df_train, random_state=0)
        df_test = shuffle(df_test, random_state=0)

        # Scale the features to be between 0 and 1
        scaler = MinMaxScaler()

        # Separate features and targets
        x_train = df_train.drop(["target", "Truth", "inv_mass"], axis=1).to_numpy()
        x_train = scaler.fit_transform(x_train)
        x_test = df_test.drop(["target", "Truth", "inv_mass"], axis=1).to_numpy()
        x_test = scaler.transform(x_test)
        y_train = df_train["target"].to_numpy()
        y_test = df_test["target"].to_numpy()

        df_train.to_parquet(f"{workdir}/train.parquet")
        df_test.to_parquet(f"{workdir}/test.parquet")

        test_df = df_test[["Truth", "inv_mass"]].reset_index(drop=True)
        train_df = df_train[["Truth", "inv_mass"]].reset_index(drop=True)

        m6j_test = {}
        for key in DATA_KEYS:
            m6j_test[key] = test_df[test_df["Truth"] == key]["inv_mass"].to_numpy()

        m6j_train = {}
        for key in DATA_KEYS:
            m6j_train[key] = train_df[train_df["Truth"] == key]["inv_mass"].to_numpy()

        joblib.dump(m6j_test, f"{workdir}/m6j_test.data.joblib")
        joblib.dump(m6j_train, f"{workdir}/m6j_train.data.joblib")

        np.savez(f"{workdir}/data.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

        metric_instance = tfk.metrics.FalsePositives(name="fp")

        model = tfk.Sequential(
            [
                tfkl.InputLayer(input_shape=(76)),
                tfkl.Dense(64, activation="relu", name="dense_1"),
                tfkl.Dropout(0.2, name="dropout_1"),
                tfkl.Dense(32, activation="relu", name="dense_2"),
                tfkl.Dropout(0.1, name="dropout_2"),
                tfkl.Dense(32, activation="relu", name="dense_3"),
                tfkl.Dense(1, activation="sigmoid", name="output"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[metric_instance, "accuracy"])
        model.summary()

        history = model.fit(
            x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test), verbose=1
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=history.epoch,
                y=history.history["loss"],
                name="Training Loss",
                mode="lines",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=history.epoch,
                y=history.history["val_loss"],
                name="Validation Loss",
                mode="lines",
                line=dict(color="red"),
            )
        )
        fig.update_layout(
            title="Training and Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            width=800,
            height=600,
        )
        fig.write_image(f"{workdir}/plots/loss.pdf")
        # fig.show()

        model.save(f"{workdir}/model.keras")

        y_pred_nn = model.predict(x_test)
        scores_test_nn = {}
        for key in DATA_KEYS:
            scores_test_nn[key] = y_pred_nn.flatten()[test_df[test_df["Truth"] == key].index]

        fig = make_histogram(scores_test_nn, 50, clip_top_prc=100, clip_bottom_prc=0, cross=None)
        fig.update_layout(
            title_text="Test scores distribution",
            barmode="stack",
            yaxis_type="log",
            xaxis_title="NN Output",
            yaxis_title="Count",
        )
        fig.write_image(f"{workdir}/plots/NN-output.pdf")
        # fig.show()

        fig, m6j_mean_nn99, m6j_std_nn99 = make_histogram_with_double_gaussian_fit(
            mass_score_cut(m6j_test, scores_test_nn, cut=0.99, prc=True), 20, clip_top_prc=100, cross=CROSS_SECTION_DICT
        )
        fig.update_layout(
            title="6-jet Mass",
            xaxis_title="Invariant Mass [GeV]",
            yaxis_title_text="count x sigma",
            barmode="stack",
            bargap=0,
            width=1600 * (5 / 6),
            height=900 * (5 / 6),
        )
        fig.write_image(f"{workdir}/plots/6jet_mass_NN_cut_099_fit.pdf")
        # fig.show()

        rf_clf = RandomForestClassifier(n_jobs=-1).fit(x_train, y_train)
        print("RFC trained")
        joblib.dump(rf_clf, f"{workdir}/rfc.joblib", compress=3)

        gb_clf = xgb.XGBClassifier(tree_method="hist").fit(x_train, y_train)
        print("GBC trained")
        joblib.dump(gb_clf, f"{workdir}/gbc.joblib", compress=3)

        y_pred_rf = rf_clf.predict_proba(x_test)[:, 1]
        y_pred_gb = gb_clf.predict_proba(x_test)[:, 1]

        sample_weights = [CROSS_SECTION_DICT[label] for label in test_df["Truth"]]

        # For the first model
        # precision_nn, recall_nn, thresholds_nn = precision_recall_curve(y_test, y_pred_nn)
        precision_nn, recall_nn, thresholds_nn = precision_recall_curve(y_test, y_pred_nn, sample_weight=sample_weights)
        pr_auc_nn = auc(recall_nn, precision_nn)

        # For the Gradient Boosting model
        # precision_gb, recall_gb, thresholds_gb = precision_recall_curve(y_test, y_pred_gb)
        precision_gb, recall_gb, thresholds_gb = precision_recall_curve(y_test, y_pred_gb, sample_weight=sample_weights)
        pr_auc_gb = auc(recall_gb, precision_gb)

        # For the Random Forest model
        # precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, y_pred_rf)
        precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, y_pred_rf, sample_weight=sample_weights)
        pr_auc_rf = auc(recall_rf, precision_rf)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recall_gb,
                y=precision_gb,
                customdata=thresholds_gb,
                hovertemplate="Threshold=%{customdata}<br>Recall=%{x}<br>Precision=%{y}",
                mode="lines",
                name=f"BDT - AUC={pr_auc_gb:.3f}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=recall_rf,
                y=precision_rf,
                customdata=thresholds_rf,
                hovertemplate="Threshold=%{customdata}<br>Recall=%{x}<br>Precision=%{y}",
                mode="lines",
                name=f"RF - AUC={pr_auc_rf:.3f}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=recall_nn,
                y=precision_nn,
                customdata=thresholds_nn,
                hovertemplate="Threshold=%{customdata}<br>Recall=%{x}<br>Precision=%{y}",
                mode="lines",
                name=f"NN - AUC={pr_auc_nn:.3f}",
            )
        )
        fig.update_layout(
            title="Cross-Section Weighted Precision-Recall Curves",
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=1200 * (2 / 3),
            height=800 * (2 / 3),
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.write_image(f"{workdir}/plots/PR-curve.pdf")
        # fig.show()

        pr_curves = {
            "precision_nn": precision_nn.tolist(),
            "recall_nn": recall_nn.tolist(),
            "threshold_nn": thresholds_nn.tolist(),
            "auc_nn": pr_auc_nn,
            "precision_gb": precision_gb.tolist(),
            "recall_gb": recall_gb.tolist(),
            "threshold_gb": thresholds_gb.tolist(),
            "auc_gb": pr_auc_gb,
            "precision_rf": precision_rf.tolist(),
            "recall_rf": recall_rf.tolist(),
            "threshold_rf": thresholds_rf.tolist(),
            "auc_rf": pr_auc_rf,
        }
        with open(f"{workdir}/pr_curves.json", "w") as f:
            json.dump(pr_curves, f)

        

        # Get feature importance from Random Forest
        rf_importance = rf_clf.feature_importances_
        gb_importance = gb_clf.feature_importances_

        # Get feature names
        feature_names = df_train.drop(["target", "Truth", "inv_mass"], axis=1).columns

        # Create a grouped bar chart
        fig = go.Figure()

        # Add bars for Random Forest
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=rf_importance,
                name="Random Forest",
                offsetgroup=1,
                marker=dict(color="#E4D91B"),
            )
        )

        # Add bars for Random Forest
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=gb_importance,
                name="Gradient Boosting",
                offsetgroup=2,
                marker=dict(color="#D91BE4"),
            )
        )

        fig.update_layout(
            title="Feature Importances for Random Forest",
            xaxis_title="Features",
            yaxis_title="Importance Value",
            # legend_title='Classifier',
            xaxis=dict(tickangle=45),
            barmode="group",
            width=1200,
        )
        fig.write_image(f"{workdir}/plots/feature_importances.pdf")
        # fig.show()

        df_importances = pd.DataFrame(
            {
                "feature": feature_names,
                "rf_importance": rf_importance,
                "gb_importance": gb_importance,
            }
        )
        df_importances.to_csv(f"{workdir}/feature_importances.csv")

        # print top 10 features by importance
        print("Top 10 features by importance")
        print("Random Forest")
        print(feature_names[np.argsort(rf_importance)[::-1][:10]])
        print("Gradient Boosting")
        print(feature_names[np.argsort(gb_importance)[::-1][:10]])

        # vertical bar chart for random forest top 10
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=feature_names[np.argsort(rf_importance)[::-1][:10]][::-1],
                x=rf_importance[np.argsort(rf_importance)[::-1][:10]][::-1],
                name="Random Forest",
                offsetgroup=1,
                marker=dict(color="#E4D91B"),
                orientation="h",
            )
        )
        fig.update_layout(
            title="Top 10 Features by Importance for Random Forest",
            xaxis_title="Features",
            yaxis_title="Importance Value",
            # legend_title='Classifier',
            xaxis=dict(tickangle=45),
            barmode="group",
            width=600,
            height=800,
        )
        fig.write_image(f"{workdir}/plots/top10_RF.pdf")
        # fig.show()

         # vertical bar chart for random forest top 10
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=feature_names[np.argsort(gb_importance)[::-1][:10]][::-1],
                x=gb_importance[np.argsort(gb_importance)[::-1][:10]][::-1],
                name="Random Forest",
                offsetgroup=1,
                marker=dict(color="#E4D91B"),
                orientation="h",
            )
        )
        fig.update_layout(
            title="Top 10 Features by Importance for Gradient Boosting",
            xaxis_title="Features",
            yaxis_title="Importance Value",
            # legend_title='Classifier',
            xaxis=dict(tickangle=45),
            barmode="group",
            width=600,
            height=800,
        )
        fig.write_image(f"{workdir}/plots/top10_GB.pdf")
        # fig.show()

        scores_test_rf = {}
        for key in DATA_KEYS:
            scores_test_rf[key] = y_pred_rf.flatten()[test_df[test_df["Truth"] == key].index]

        scores_test_gb = {}
        for key in DATA_KEYS:
            scores_test_gb[key] = y_pred_gb.flatten()[test_df[test_df["Truth"] == key].index]

        fig = make_histogram(scores_test_rf, 50, clip_top_prc=100, clip_bottom_prc=0, cross=None)
        fig.update_layout(
            title_text="Data Sample Content by Model Output Cut",
            barmode="stack",
            yaxis_type="log",
            xaxis_title="RF Output",
            yaxis_title="Probability Density",
            width=1600 * (5 / 6),
            height=900 * (5 / 6),
        )
        fig.write_image(f"{workdir}/plots/RF-output.pdf")
        # fig.show()

        fig, m6j_mean_rf99, m6j_std_rf99 = make_histogram_with_double_gaussian_fit(
            mass_score_cut(m6j_test, scores_test_rf, 0.99, prc=True), 20, clip_top_prc=100, cross=CROSS_SECTION_DICT
        )
        fig.update_layout(
            title="6-jet Mass",
            xaxis_title="Invariant Mass [GeV]",
            yaxis_title_text="count x sigma",
            # yaxis_type="log",
            barmode="stack",
            bargap=0,
            width=1600 * (2 / 3),
            height=900 * (2 / 3),
        )
        fig.write_image(f"{workdir}/plots/6jet_mass_RF_cut_099_fit.pdf")
        # fig.show()

        fig = make_histogram(scores_test_gb, 50, clip_top_prc=100, clip_bottom_prc=0, cross=None)
        fig.update_layout(
            title_text="Data Sample Content by Model Output Cut",
            barmode="stack",
            yaxis_type="log",
            xaxis_title="GB Output",
            yaxis_title="Probability Density",
            width=1600 * (5 / 6),
            height=900 * (5 / 6),
        )
        fig.write_image(f"{workdir}/plots/GB-output.pdf")
        # fig.show()

        fig, m6j_mean_gb99, m6j_std_gb99 = make_histogram_with_double_gaussian_fit(
            mass_score_cut(m6j_test, scores_test_gb, 0.99, prc=True), 20, clip_top_prc=100, cross=CROSS_SECTION_DICT
        )
        fig.update_layout(
            title="6-jet Mass",
            xaxis_title="Invariant Mass [GeV]",
            yaxis_title_text="count x sigma",
            # yaxis_type="log",
            barmode="stack",
            bargap=0,
            width=1600 * (2 / 3),
            height=900 * (2 / 3),
        )
        fig.write_image(f"{workdir}/plots/6jet_mass_GB_cut_099_fit.pdf")

        fits = {
            "nn": {"mean": m6j_mean_nn99, "std": m6j_std_nn99},
            "rf": {"mean": m6j_mean_rf99, "std": m6j_std_rf99},
            "gb": {"mean": m6j_mean_gb99, "std": m6j_std_gb99},
        }
        with open(f"{workdir}/fits.json", "w") as f:
            json.dump(fits, f)

        res_rf = {}
        for cut in (0.2, 0.5, 0.8, 0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99):
            scores = mass_score_cut(m6j_test, scores_test_rf, cut, prc=True)
            counts = {k: len(v) * TOTAL_LUMI * CROSS_SECTION_DICT[k]/m6j_test[key].shape[0] for k, v in scores.items()}
            res_rf[cut] = counts
        df_counts_rf = pd.DataFrame(res_rf)
        bkg_counts_rf = df_counts_rf.iloc[:-1].T.sum(axis=1)
        sig_counts_rf = df_counts_rf.iloc[-1]
        s_over_b_rf = sig_counts_rf / bkg_counts_rf

        # add s_over_b as a row
        df_counts_rf.loc["BKG:sum"] = bkg_counts_rf
        df_counts_rf.loc["S/B"] = s_over_b_rf
        df_counts_rf.to_csv(f"{workdir}/counts_rf.csv")


        res_nn = {}
        for cut in (0.2, 0.5, 0.8, 0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99):
            scores = mass_score_cut(m6j_test, scores_test_nn, cut, prc=True)
            counts =  {k: len(v) * TOTAL_LUMI * CROSS_SECTION_DICT[k]/m6j_test[key].shape[0] for k, v in scores.items()}
            res_nn[cut] = counts
        df_counts_nn = pd.DataFrame(res_nn)

        bkg_counts_nn = df_counts_nn.iloc[:-1].T.sum(axis=1)
        sig_counts_nn = df_counts_nn.iloc[-1]
        s_over_b_nn = sig_counts_nn / bkg_counts_nn

        # add s_over_b as a row
        df_counts_nn.loc["BKG:sum"] = bkg_counts_nn
        df_counts_nn.loc["S/B"] = s_over_b_nn
        df_counts_nn.to_csv(f"{workdir}/counts_nn.csv")
        

        res_gb = {}
        for cut in (0.2, 0.5, 0.8, 0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99):
            scores = mass_score_cut(m6j_test, scores_test_gb, cut, prc=True)
            counts = {k: len(v) * TOTAL_LUMI * CROSS_SECTION_DICT[k]/m6j_test[key].shape[0] for k, v in scores.items()}
            res_gb[cut] = counts
        df_counts_gb = pd.DataFrame(res_gb)

        bkg_counts_gb = df_counts_gb.iloc[:-1].T.sum(axis=1)
        sig_counts_gb = df_counts_gb.iloc[-1]
        s_over_b_gb = sig_counts_gb / bkg_counts_gb

        # add s_over_b as a row
        df_counts_gb.loc["BKG:sum"] = bkg_counts_gb
        df_counts_gb.loc["S/B"] = s_over_b_gb
        df_counts_gb.to_csv(f"{workdir}/counts_gb.csv")