TARGET_CLASSES = ['HYPERCAR', 'GT3']

HOMOLOGATION_GROUP_MAP = {
    'LMH':  'LMH',
    'LMDh': 'LMDh',
    'GT3':  'GT3',
    'LMP2': 'LMP2',
}

# low confidence (< 1500 laps)
LOW_COVERAGE_MANUFACTURERS = ['Isotta Fraschini', 'Vanwall', 'Lamborghini']

PACE_SESSIONS = ['race']
STINT_SESSIONS = ['race']  # est_tire_age only in race

EXCLUDED_FLAGS = ['FCY', 'RF', 'SF']
NULL_FLAGS_TREATMENT = 'treat_as_green'  # for 2021

MIN_LAPS_PER_STINT = 3
MIN_STINTS_PER_MODEL_EVENT = 1
MIN_EVENTS_FOR_RECOMMENDATION = 2
LOW_SAMPLE_LAPS_THRESHOLD = 300 

OUTLIER_MAD_THRESHOLD = 3.5
EXCLUDE_STINT_START_LAPS = True
EXCLUDE_STINT_END_LAPS = True

PIT_LAP_RULE = 'pit_time_notna'  # is_pit_lap = pit_time.notna()

TIRE_AGE_SESSIONS = ['race']

RECOMMENDER_WEIGHTS = {
    'pace':        0.50,
    'consistency': 0.20,
    'long_run':    0.20,
    'stability':   0.10,
}