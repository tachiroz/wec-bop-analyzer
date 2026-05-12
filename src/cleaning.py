import pandas as pd
import numpy as np
from config import (
    TARGET_CLASSES, HOMOLOGATION_GROUP_MAP,
    EXCLUDED_FLAGS, NULL_FLAGS_TREATMENT,
    PACE_SESSIONS, STINT_SESSIONS,
    EXCLUDE_STINT_START_LAPS, EXCLUDE_STINT_END_LAPS,
    OUTLIER_MAD_THRESHOLD, MIN_LAPS_PER_STINT
)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.drop(columns=['bpillar_quartile'], errors='ignore')
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['raining'] = df['raining'].astype(bool)
    df['stint_start'] = df['stint_start'].astype(bool)
    for col in ['lap_time', 'lap_time_s1', 'lap_time_s2', 'lap_time_s3']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def add_homologation_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['homologation_group'] = df['homologation'].map(HOMOLOGATION_GROUP_MAP)
    # fallback: if homologation not in map - choose class_normalized
    df['homologation_group'] = df['homologation_group'].fillna(df['class_normalized'])
    return df


def add_car_model_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['car_model_key'] = df['manufacturer'] + '_' + df['homologation_group']
    return df


def mark_pit_laps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_pit_lap'] = df['pit_time'].notna()
    return df


def mark_stint_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # stint_start
    df['is_stint_start_lap'] = df['stint_start'].astype(bool)

    # last stint lap - next string stint_start=True
    # or if it last lap of the car in session
    df = df.sort_values(['session_id', 'car', 'stint_number', 'stint_lap'])
    df['is_stint_end_lap'] = (
        df.groupby(['session_id', 'car', 'stint_number'])['stint_lap']
        .transform(lambda x: x == x.max())
    )
    return df


def mark_flagged_laps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if NULL_FLAGS_TREATMENT == 'treat_as_green':
        flags_filled = df['flags'].fillna('GF')
    else:
        flags_filled = df['flags']

    df['is_flagged_lap'] = flags_filled.isin(EXCLUDED_FLAGS)
    return df


def mark_wet_laps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_wet_lap'] = df['raining'].astype(bool)
    df['weather_bucket'] = np.where(df['is_wet_lap'], 'wet', 'dry')
    return df


def mark_outlier_laps(df: pd.DataFrame) -> pd.DataFrame:
    """MAD-based outlier detection — только по предварительно чистым кругам."""
    df = df.copy()
    df['is_outlier_lap'] = False

    # Считаем MAD только по кругам без явных исключений
    pre_clean_mask = (
        df['lap_time'].notna() &
        ~df['is_pit_lap'] &
        ~df['is_flagged_lap'] &
        ~df['is_stint_start_lap'] &
        ~df['is_stint_end_lap']
    )

    groups = df[pre_clean_mask].groupby(['session_id', 'car'])

    outlier_indices = []
    for (sid, car), group in groups:
        lap_times = group['lap_time']
        if len(lap_times) < MIN_LAPS_PER_STINT:
            outlier_indices.extend(group.index.tolist())
            continue

        median = lap_times.median()
        mad = (lap_times - median).abs().median()

        if mad == 0:
            continue

        threshold = OUTLIER_MAD_THRESHOLD * mad
        bad = group[(lap_times - median).abs() > threshold].index
        outlier_indices.extend(bad.tolist())

    df.loc[outlier_indices, 'is_outlier_lap'] = True
    return df


def add_exclusion_reason(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    conditions = [
        (df['lap_time'].isna(),                'missing_lap_time'),
        (df['is_pit_lap'],                     'pit_lap'),
        (df['is_flagged_lap'],                 'flagged_lap'),
        (df['is_outlier_lap'],                 'outlier_lap'),
        (EXCLUDE_STINT_START_LAPS & df['is_stint_start_lap'], 'stint_start_lap'),
        (EXCLUDE_STINT_END_LAPS   & df['is_stint_end_lap'],   'stint_end_lap'),
    ]
    df['exclusion_reason'] = None
    for mask, reason in conditions:
        df.loc[mask & df['exclusion_reason'].isna(), 'exclusion_reason'] = reason
    return df


def add_validity_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['valid_for_pace_analysis'] = (
        df['exclusion_reason'].isna() &
        df['session'].isin(PACE_SESSIONS) &
        df['class_normalized'].isin(TARGET_CLASSES)
    )

    df['valid_for_stint_analysis'] = (
        df['lap_time'].notna() &
        ~df['is_pit_lap'] &
        ~df['is_flagged_lap'] &
        df['session'].isin(STINT_SESSIONS) &
        df['class_normalized'].isin(TARGET_CLASSES)
    )

    return df


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['track_temp_bucket'] = pd.cut(
        df['track_temp_f'],
        bins=[0, 86, 104, 999],
        labels=['cool', 'medium', 'hot']
    )

    df['tire_age_bucket'] = pd.cut(
        df['est_tire_age'],
        bins=[-1, 10, 25, 999],
        labels=['fresh', 'mid', 'worn']
    )

    return df


def build_clean_layer(df: pd.DataFrame) -> pd.DataFrame:
    df = cast_types(df)
    df = add_homologation_group(df)
    df = add_car_model_key(df)
    df = mark_pit_laps(df)
    df = mark_stint_boundaries(df)
    df = mark_flagged_laps(df)
    df = mark_wet_laps(df)
    df = mark_outlier_laps(df)  
    df = add_exclusion_reason(df)
    df = add_validity_flags(df)
    df = add_buckets(df)
    return df