#!/usr/bin/env python3
"""
offline_ids_model_tester.py

Safely tests saved IDS machine-learning models using synthetic NSL-KDD-style
feature rows. This script does NOT send packets, scan networks, or execute
real attacks. It only creates offline feature vectors that resemble common
traffic/attack categories and passes them into saved .pkl models.

Example:
    python offline_ids_model_tester.py --rf ids_random_forest_model.pkl --svm ids_svm_model.pkl
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]


def base_normal_sample():
    return {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 215,
        "dst_bytes": 45076,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 1,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": 1,
        "srv_count": 1,
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 9,
        "dst_host_srv_count": 9,
        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.11,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0,
    }


def make_synthetic_samples():
    samples = []

    normal_http = base_normal_sample()
    normal_http["scenario"] = "normal_http"
    normal_http["expected_type"] = "normal"
    samples.append(normal_http)

    normal_dns = base_normal_sample()
    normal_dns.update({
        "protocol_type": "udp",
        "service": "domain_u",
        "flag": "SF",
        "src_bytes": 44,
        "dst_bytes": 120,
        "logged_in": 0,
        "count": 3,
        "srv_count": 3,
        "dst_host_count": 20,
        "dst_host_srv_count": 20,
        "scenario": "normal_dns",
        "expected_type": "normal",
    })
    samples.append(normal_dns)

    dos_syn = base_normal_sample()
    dos_syn.update({
        "service": "http",
        "flag": "S0",
        "src_bytes": 0,
        "dst_bytes": 0,
        "logged_in": 0,
        "count": 250,
        "srv_count": 250,
        "serror_rate": 1.0,
        "srv_serror_rate": 1.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "dst_host_count": 255,
        "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0,
        "dst_host_serror_rate": 1.0,
        "dst_host_srv_serror_rate": 1.0,
        "scenario": "dos_syn_flood_like",
        "expected_type": "attack",
    })
    samples.append(dos_syn)

    probe_scan = base_normal_sample()
    probe_scan.update({
        "service": "private",
        "flag": "REJ",
        "src_bytes": 0,
        "dst_bytes": 0,
        "logged_in": 0,
        "count": 120,
        "srv_count": 8,
        "rerror_rate": 1.0,
        "srv_rerror_rate": 1.0,
        "same_srv_rate": 0.07,
        "diff_srv_rate": 0.93,
        "dst_host_count": 255,
        "dst_host_srv_count": 8,
        "dst_host_same_srv_rate": 0.03,
        "dst_host_diff_srv_rate": 0.97,
        "dst_host_rerror_rate": 1.0,
        "dst_host_srv_rerror_rate": 1.0,
        "scenario": "probe_port_scan_like",
        "expected_type": "attack",
    })
    samples.append(probe_scan)

    r2l_login = base_normal_sample()
    r2l_login.update({
        "duration": 5,
        "service": "ftp",
        "flag": "SF",
        "src_bytes": 125,
        "dst_bytes": 331,
        "hot": 3,
        "num_failed_logins": 5,
        "logged_in": 0,
        "is_guest_login": 1,
        "count": 4,
        "srv_count": 4,
        "dst_host_count": 30,
        "dst_host_srv_count": 6,
        "dst_host_same_srv_rate": 0.20,
        "scenario": "r2l_failed_login_like",
        "expected_type": "attack",
    })
    samples.append(r2l_login)

    u2r_priv = base_normal_sample()
    u2r_priv.update({
        "duration": 20,
        "service": "telnet",
        "flag": "SF",
        "src_bytes": 300,
        "dst_bytes": 1500,
        "hot": 8,
        "logged_in": 1,
        "num_compromised": 3,
        "root_shell": 1,
        "su_attempted": 1,
        "num_root": 3,
        "num_file_creations": 2,
        "num_shells": 1,
        "num_access_files": 2,
        "count": 2,
        "srv_count": 2,
        "scenario": "u2r_privilege_escalation_like",
        "expected_type": "attack",
    })
    samples.append(u2r_priv)

    df = pd.DataFrame(samples)

    for col in NSL_KDD_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    return df[["scenario", "expected_type"] + NSL_KDD_COLUMNS]


def load_optional(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"Warning: optional file not found: {path}")
        return None
    return joblib.load(p)


def prepare_features_for_model(model, raw_features, feature_columns=None):
    try:
        model.predict(raw_features.head(1))
        return raw_features
    except Exception:
        pass

    encoded = pd.get_dummies(raw_features)

    if feature_columns is None:
        raise ValueError(
            "Model did not accept raw features, and no feature_columns.pkl was supplied. "
            "Provide --features feature_columns.pkl or save the full preprocessing Pipeline."
        )

    return encoded.reindex(columns=feature_columns, fill_value=0)


def decode_prediction(pred, encoder=None):
    if encoder is None:
        return str(pred)
    try:
        return encoder.inverse_transform([pred])[0]
    except Exception:
        return str(pred)


def get_attack_probability(model, X_model, encoder=None):
    if not hasattr(model, "predict_proba"):
        return [None] * len(X_model)

    proba = model.predict_proba(X_model)

    try:
        if encoder is not None:
            attack_label = encoder.transform(["attack"])[0]
        else:
            attack_label = "attack"
        attack_col = list(model.classes_).index(attack_label)
    except Exception:
        attack_col = 1 if proba.shape[1] > 1 else 0

    return proba[:, attack_col]


def evaluate_model(name, model_path, raw_df, encoder=None, feature_columns=None):
    print(f"\n{'=' * 70}")
    print(f"Testing model: {name}")
    print(f"Model file: {model_path}")
    print(f"{'=' * 70}")

    model = joblib.load(model_path)

    metadata = raw_df[["scenario", "expected_type"]].copy()
    raw_features = raw_df.drop(columns=["scenario", "expected_type"])

    X_model = prepare_features_for_model(model, raw_features, feature_columns)

    preds = model.predict(X_model)
    attack_probs = get_attack_probability(model, X_model, encoder)

    results = metadata.copy()
    results["predicted_label"] = [decode_prediction(p, encoder) for p in preds]

    if attack_probs[0] is not None:
        results["attack_probability"] = np.round(attack_probs, 4)
    else:
        results["attack_probability"] = "N/A"

    results["correct"] = results["expected_type"] == results["predicted_label"]

    print(results.to_string(index=False))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Safely test saved IDS .pkl models using synthetic NSL-KDD-style feature rows."
    )
    parser.add_argument("--rf", default="ids_random_forest_model.pkl", help="Path to saved Random Forest .pkl model")
    parser.add_argument("--svm", default="ids_svm_model.pkl", help="Path to saved SVM .pkl model")
    parser.add_argument("--encoder", default="label_encoder.pkl", help="Path to saved LabelEncoder .pkl")
    parser.add_argument("--features", default="feature_columns.pkl", help="Path to saved feature columns .pkl")
    parser.add_argument("--output", default="synthetic_ids_test_results.csv", help="CSV file for saving results")
    parser.add_argument("--save-samples", default="synthetic_ids_samples.csv", help="CSV file for saving generated samples")
    args = parser.parse_args()

    df = make_synthetic_samples()
    df.to_csv(args.save_samples, index=False)
    print(f"Saved synthetic feature samples to: {args.save_samples}")

    encoder = load_optional(args.encoder)
    feature_columns = load_optional(args.features)

    all_results = []

    if Path(args.rf).exists():
        rf_results = evaluate_model("Random Forest", args.rf, df, encoder, feature_columns)
        rf_results.insert(0, "model", "Random Forest")
        all_results.append(rf_results)
    else:
        print(f"Random Forest model not found: {args.rf}")

    if Path(args.svm).exists():
        svm_results = evaluate_model("SVM", args.svm, df, encoder, feature_columns)
        svm_results.insert(0, "model", "SVM")
        all_results.append(svm_results)
    else:
        print(f"SVM model not found: {args.svm}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nSaved combined results to: {args.output}")
    else:
        print("\nNo models were tested. Check your --rf and --svm paths.")


if __name__ == "__main__":
    main()
