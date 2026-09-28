import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from metric import precision_at_recall

RANDOM_STATE = 42
DATA_DIR = "data"



# Load-
def read_data():
    train = pd.read_csv(
        os.path.join(DATA_DIR, "train.csv"),
        parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"],
    )
    test = pd.read_csv(
        os.path.join(DATA_DIR, "test.csv"),
        parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"],
    )
    events = pd.read_csv(
        os.path.join(DATA_DIR, "events.csv.gz"),
        compression="gzip",
        parse_dates=["event_ts"],
    )
    return train, test, events


# Feature helpers
def entropy(s: pd.Series) -> float:
    p = s.value_counts(normalize=True, dropna=True)
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log(p + 1e-9)).sum())


def iet_stats(ts: pd.Series) -> pd.Series:
    ts = pd.Series(ts).sort_values()
    if len(ts) < 2:
        return pd.Series({
            "iet_mean": np.nan,
            "iet_std": np.nan,
            "iet_min": np.nan,
            "iet_max": np.nan,
            "iet_median": np.nan,
            "iet_p90": np.nan,
            "iet_lt1": 0.0,
            "iet_lt05": 0.0,
            "iet_lt01": 0.0,
        })

    diffs = ts.diff().dt.total_seconds().dropna()
    return pd.Series({
        "iet_mean": diffs.mean(),
        "iet_std": diffs.std(),
        "iet_min": diffs.min(),
        "iet_max": diffs.max(),
        "iet_median": diffs.median(),
        "iet_p90": diffs.quantile(0.9),
        "iet_lt1": (diffs < 1).mean(),
        "iet_lt05": (diffs < 0.5).mean(),
        "iet_lt01": (diffs < 0.1).mean(),
    })


def session_stats(df: pd.DataFrame) -> pd.Series:
    ts = df["event_ts"].sort_values()

    if len(ts) == 0:
        return pd.Series({
            "n_sessions": 0,
            "events_per_session": 0.0,
            "session_duration_mean": 0.0,
            "session_duration_max": 0.0,
        })

    if len(ts) == 1:
        return pd.Series({
            "n_sessions": 1,
            "events_per_session": 1.0,
            "session_duration_mean": 0.0,
            "session_duration_max": 0.0,
        })

    diffs = ts.diff().dt.total_seconds().fillna(0)
    breaks = (diffs > 1800).cumsum()   # 30-minute inactivity gap
    sessions = ts.groupby(breaks)

    durations = sessions.apply(lambda x: (x.max() - x.min()).total_seconds())
    sizes = sessions.size()

    return pd.Series({
        "n_sessions": sessions.ngroups,
        "events_per_session": sizes.mean(),
        "session_duration_mean": durations.mean(),
        "session_duration_max": durations.max(),
    })


def build_features(meta: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()

    meta["age_days"] = (
        meta["window_start_ts"] - meta["cookie_created_at"]
    ).dt.total_seconds() / 86400.0
    meta["window_start_hour"] = meta["window_start_ts"].dt.hour
    meta["window_start_dow"] = meta["window_start_ts"].dt.dayofweek

    # Keep only events inside each cookie observation window.
    ev = events.merge(
        meta[["cookie_id", "window_start_ts", "window_end_ts"]],
        on="cookie_id",
        how="inner",
    )
    ev = ev[
        (ev["event_ts"] >= ev["window_start_ts"])
        & (ev["event_ts"] < ev["window_end_ts"])
    ].copy()

    ev.sort_values(["cookie_id", "event_ts"], inplace=True)
    ev["platform"] = ev["platform"].astype(str).str.lower()
    ev["event_name"] = ev["event_name"].astype(str).str.lower()

    features = pd.DataFrame(index=meta["cookie_id"].unique())
    features.index.name = "cookie_id"

    if ev.empty:
        return features

    g = ev.groupby("cookie_id")

    # Basic counts
    features["n_events"] = g.size().reindex(features.index, fill_value=0)
    features["n_unique_items"] = g.item_id.nunique().reindex(features.index, fill_value=0)
    features["n_unique_categories"] = g.item_category.nunique().reindex(features.index, fill_value=0)
    features["n_unique_locations"] = g.item_location.nunique().reindex(features.index, fill_value=0)
    features["n_unique_sellers"] = g.seller_type.nunique().reindex(features.index, fill_value=0)
    features["n_unique_platforms"] = g.platform.nunique().reindex(features.index, fill_value=0)
    features["n_unique_user_agents"] = g.user_agent.nunique().reindex(features.index, fill_value=0)
    features["n_unique_eids"] = g.eid.nunique().reindex(features.index, fill_value=0)
    features["n_unique_event_names"] = g.event_name.nunique().reindex(features.index, fill_value=0)
    features["n_unique_search_queries"] = g.search_query.nunique().reindex(features.index, fill_value=0)
    features["n_unique_search_pages"] = g.search_page.nunique().reindex(features.index, fill_value=0)

    first_ts = g.event_ts.min().reindex(features.index)
    last_ts = g.event_ts.max().reindex(features.index)

    features["duration_sec"] = (last_ts - first_ts).dt.total_seconds()
    features["events_per_sec"] = features["n_events"] / (features["duration_sec"] + 1.0)
    features["events_per_min"] = features["n_events"] / (features["duration_sec"] / 60.0 + 1.0)
    features["active_hours"] = features["duration_sec"] / 3600.0

    # Missing ratios
    for col in [
        "item_id", "item_category", "item_location", "seller_type",
        "search_query", "search_page", "pointer_x", "pointer_y", "user_agent",
    ]:
        if col in ev.columns:
            features[f"missing_{col}_ratio"] = (
                g[col].apply(lambda s: s.isna().mean())
                .reindex(features.index, fill_value=1.0)
            )

    # Duplicate ratio
    dup = ev.duplicated(
        subset=["cookie_id", "event_ts", "eid", "item_id"],
        keep=False,
    )
    features["duplicate_ratio"] = (
        dup.groupby(ev["cookie_id"]).mean()
        .reindex(features.index, fill_value=0.0)
    )

    # Event type counts
    for col, prefix in [("eid", "eid"), ("event_name", "event")]:
        piv = ev.pivot_table(
            index="cookie_id",
            columns=col,
            values="event_ts",
            aggfunc="count",
            fill_value=0,
        )
        piv.columns = [f"{prefix}_{str(c)}_count" for c in piv.columns]
        features = features.join(piv, how="left")

    # Platform counts
    piv = ev.pivot_table(
        index="cookie_id",
        columns="platform",
        values="event_ts",
        aggfunc="count",
        fill_value=0,
    )
    piv.columns = [f"platform_{str(c)}_count" for c in piv.columns]
    features = features.join(piv, how="left")

    # Ratios for all count columns
    count_cols = [c for c in features.columns if c.endswith("_count")]
    for c in count_cols:
        features[c + "_ratio"] = features[c] / features["n_events"].replace(0, np.nan)

    # Entropy / diversity
    for col, name in [
        ("item_category", "category"),
        ("item_location", "location"),
        ("seller_type", "seller"),
        ("search_query", "search_query"),
    ]:
        features[f"{name}_entropy"] = (
            g[col].apply(entropy)
            .reindex(features.index, fill_value=0.0)
        )

    # Search features
    features["search_page_mean"] = g.search_page.mean().reindex(features.index)
    features["search_page_max"] = g.search_page.max().reindex(features.index)
    features["search_page_std"] = g.search_page.std().reindex(features.index)
    features["search_page_gt1_ratio"] = (
        g.search_page.apply(lambda s: (s > 1).mean())
        .reindex(features.index, fill_value=0.0)
    )

    # Inter-event time
    iet = g.event_ts.apply(iet_stats).unstack()
    features = features.join(iet, how="left")

    # Sessions
    sess = g.apply(session_stats).unstack()
    features = features.join(sess, how="left")

    # User-Agent flags
    ua = g.user_agent.first().fillna("").reindex(features.index)
    features["ua_is_okhttp"] = ua.str.contains("okhttp", case=False, na=False).astype(int)
    features["ua_is_avito_app"] = ua.str.contains("Avito/", case=False, na=False).astype(int)
    features["ua_is_headless"] = ua.str.contains("HeadlessChrome", case=False, na=False).astype(int)
    features["ua_is_python"] = ua.str.contains("python|urllib|requests", case=False, na=False).astype(int)
    features["ua_is_curl"] = ua.str.contains("curl", case=False, na=False).astype(int)
    features["ua_is_scrapy"] = ua.str.contains("scrapy", case=False, na=False).astype(int)
    features["ua_is_go"] = ua.str.contains("Go-http-client", case=False, na=False).astype(int)
    features["ua_is_mobile"] = ua.str.contains("Mobile|Android|iPhone", case=False, na=False).astype(int)
    features["ua_is_desktop"] = ua.str.contains("Windows|Macintosh|Linux", case=False, na=False).astype(int)

    # Pointer features
    for col in ["pointer_x", "pointer_y"]:
        features[f"{col}_mean"] = g[col].mean().reindex(features.index)
        features[f"{col}_std"] = g[col].std().reindex(features.index)

    features["pointer_nonnull_ratio"] = (
        g.pointer_x.apply(lambda s: s.notna().mean())
        .reindex(features.index, fill_value=0.0)
    )
    features["pointer_unique_ratio"] = (
        g.pointer_x.apply(lambda s: s.nunique() / len(s) if len(s) > 0 else 0.0)
        .reindex(features.index, fill_value=0.0)
    )

    # Time features
    features["hour_min"] = first_ts.dt.hour
    features["hour_max"] = last_ts.dt.hour
    features["night_ratio"] = (
        g.event_ts.apply(lambda s: ((s.dt.hour < 6) | (s.dt.hour >= 23)).mean())
        .reindex(features.index, fill_value=0.0)
    )

    # Metadata
    meta_idx = meta.set_index("cookie_id")
    features["age_days"] = meta_idx["age_days"].reindex(features.index)
    features["window_start_hour"] = meta_idx["window_start_hour"].reindex(features.index)
    features["window_start_dow"] = meta_idx["window_start_dow"].reindex(features.index)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features


#main
def main():
    train, test, events = read_data()

    meta_cols = ["cookie_id", "cookie_created_at", "window_start_ts", "window_end_ts"]

    print("Building train features...")
    Xtr = build_features(train[meta_cols], events)

    print("Building test features...")
    Xte = build_features(test[meta_cols], events)

    all_cols = sorted(set(Xtr.columns) | set(Xte.columns))
    Xtr = Xtr.reindex(columns=all_cols, fill_value=0)
    Xte = Xte.reindex(columns=all_cols, fill_value=0)

    # Align order
    Xtr = Xtr.loc[train["cookie_id"]]
    Xte = Xte.loc[test["cookie_id"]]
    y = train["target"].values

    # Time-based validation: hold out last 20% of observation dates.
    dates = np.sort(train["window_start_ts"].unique())
    n_valid = max(1, int(len(dates) * 0.2))
    valid_dates = dates[-n_valid:]

    valid_mask = train["window_start_ts"].isin(valid_dates)
    train_mask = ~valid_mask

    if train.loc[valid_mask, "target"].sum() < 10:
        valid_dates = dates[-1:]
        valid_mask = train["window_start_ts"].isin(valid_dates)
        train_mask = ~valid_mask

    X_train = Xtr.loc[train.loc[train_mask, "cookie_id"]]
    y_train = train.loc[train_mask, "target"].values

    X_val = Xtr.loc[train.loc[valid_mask, "cookie_id"]]
    y_val = train.loc[valid_mask, "target"].values

    print(f"Validation dates: {valid_dates}")
    print(f"Train shape: {X_train.shape}, valid shape: {X_val.shape}")

    # 
    baseline = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
    )
    baseline.fit(X_train, y_train)
    p_base = baseline.predict_proba(X_val)[:, 1]
    print("Baseline RF P@R0.7:", round(precision_at_recall(y_val, p_base), 4))

    # Improved model
    if HAS_LGBM:
        model = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )

    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]
    print("Improved P@R0.7:", round(precision_at_recall(y_val, p_val), 4))

    #
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X_train.columns)
        print("\nTop 30 features:")
        print(imp.sort_values(ascending=False).head(30))

    #
    print("\nFitting final model on full train...")
    model.fit(Xtr, y)

    p_test = model.predict_proba(Xte)[:, 1]

    sub = pd.DataFrame({
        "cookie_id": test["cookie_id"],
        "score": p_test,
    })

    assert len(sub) == len(test)
    assert sub["score"].between(0, 1).all()

    sub.to_csv("submission.csv", index=False)
    print("\nSaved submission.csv")
    print(sub.head())


if __name__ == "__main__":
    main()