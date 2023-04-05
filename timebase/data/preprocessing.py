"""
Helper functions to preprocess CSV files
Reference on data export and formatting of Empatica E4 wristband
https://support.empatica.com/hc/en-us/articles/201608896-Data-export-and-formatting-from-E4-connect-
"""

import os
import mne
import warnings
import typing as t
import numpy as np
import pandas as pd
from tqdm import tqdm
import hrvanalysis as hrv
from math import ceil, floor

from timebase.data import utils
from timebase.utils.utils import update_dict


FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

HR_OFFSET = 10  # HR data record 10s after the t0
# channel names of the csv recorded data
CSV_CHANNELS = ["ACC", "BVP", "EDA", "HR", "TEMP", "IBI"]
# HRV: time domain features
HRV_FEATURES = [
    "mean_nni",
    "sdnn",
    "sdsd",
    "nni_50",
    "pnni_50",
    "nni_20",
    "pnni_20",
    "rmssd",
    "median_nni",
    "range_nni",
    "cvsd",
    "cvnni",
    "mean_hr",
    "max_hr",
    "min_hr",
    "std_hr",
]

# maximum values of each individual symptom in YMRS and HDRS
MAX_YMRS = [4, 4, 4, 4, 8, 8, 4, 8, 8, 4, 4]
MAX_HDRS = [4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 2, 2, 2, 4, 3, 2]
# label format [session ID, is patient, timing, YMRS(1 - 11), HDRS(1 - 17)]
LABEL_SCALE = np.array([1, 1, 1] + MAX_YMRS + MAX_HDRS, dtype=np.float32)

# clinical data information from the spreadsheet
STATUS = ["T0_Session_Code", "T1_Session_Code", "T2_Session_Code", "T3_Session_Code"]
YMRS = [
    "YMRS1",
    "YMRS2",
    "YMRS3",
    "YMRS4",
    "YMRS5",
    "YMRS6",
    "YMRS7",
    "YMRS8",
    "YMRS9",
    "YMRS10",
    "YMRS11",
]
HDRS = [
    "HDRS1",
    "HDRS2",
    "HDRS3",
    "HDRS4",
    "HDRS5",
    "HDRS6",
    "HDRS7",
    "HDRS8",
    "HDRS9",
    "HDRS10",
    "HDRS11",
    "HDRS12",
    "HDRS13",
    "HDRS14",
    "HDRS15",
    "HDRS16",
    "HDRS17",
]


def segmentation(args, features: t.List[np.ndarray]):
    """
    Segment preprocessed features along the temporal dimension into
    N non-overlapping segments where each segment has size args.segment_length
    """
    assert args.segment_length > 0
    segments, discarded = [], []
    for i in range(len(features)):
        duration = features[i].shape[0]
        if duration < args.segment_length:
            discarded.append(1)
            continue
        num_segments = floor(duration // args.segment_length)
        indexes = np.linspace(
            start=0, stop=duration - args.segment_length, num=num_segments, dtype=int
        )
        segments.extend([features[i][j : j + args.segment_length] for j in indexes])
    return np.stack(segments, axis=0)


def create_pairs(
    args, features: t.List[np.ndarray], label: t.List[int], session_ids: t.List[int]
):
    """Segment features and return train, validation and test set pairs
    Returns:
      features: np.ndarray, segmented features
      labels: np.ndarray, paired labels
    """
    features = segmentation(args, features=features)
    get_labels = lambda x: np.tile(label, reps=(len(x), 1)).astype(np.float32)

    total_size, test_size = features.shape[0], args.test_size
    val_size = int((total_size - test_size) * 0.2)

    assert total_size > test_size, f"total size: {total_size}, test size: {test_size}"
    if (remaining := total_size - test_size) < 100:
        print(
            f"Sessions ({session_ids}): only {remaining} segments for training and validation."
        )

    # shuffle segments
    features = np.random.permutation(features)

    x_test = features[:test_size]
    x_val = features[test_size : test_size + val_size]
    x_train = features[test_size + val_size :]

    # TODO hot fix to trim the number of training and validation samples s.t.
    #  the number of samples with different segment length remains the same
    x_train, x_val = x_train[:200], x_val[:50]

    return {
        "x_train": x_train,
        "y_train": get_labels(x_train),
        "x_val": x_val,
        "y_val": get_labels(x_val),
        "x_test": x_test,
        "y_test": get_labels(x_test),
    }


def read_clinical_info(filename: str):
    """Read clinical EXCEL file"""
    assert os.path.isfile(filename), f"clinical file {filename} does not exists."
    xls = pd.ExcelFile(filename)
    info = pd.read_excel(xls, sheet_name=None)  # read all sheets
    return pd.concat(info)


def get_session_info(session_id: int, clinical_info: pd.DataFrame):
    """Extract clinical information of patient given session_id"""
    session, timing = None, 0
    while timing < len(STATUS):
        token = 0
        for idx, val in enumerate(clinical_info[STATUS[timing]].to_list()):
            if str(session_id) in str(val):
                session = clinical_info.loc[clinical_info[STATUS[timing]] == val]
                token += 1
        if token:
            break
        timing += 1
    else:
        print(f"cannot find session {session_id} in the spreadsheet.")
    info = {
        "timing": timing,
        "YMRS": [0] * len(YMRS),
        "HDRS": [0] * len(HDRS),
        "is_patient": False,
    }
    if session is not None:
        info["YMRS"] = [int(session[f"T{timing}_{c}"].values[0]) for c in YMRS]
        info["HDRS"] = [int(session[f"T{timing}_{c}"].values[0]) for c in HDRS]
        info["is_patient"] = session["Control/Patient"].values[0] == "Patient"
    return info


def load_channel(recording_dir: str, channel: str):
    """Load channel CSV data from file
    Returns
      unix_t0: int, the start time of the recording in UNIX time
      sampling_rate: int, sampling rate of the recording (if exists)
      data: np.ndarray, the raw recording data
    """
    assert channel in CSV_CHANNELS, f"unknown channel {channel}"
    raw_data = pd.read_csv(
        os.path.join(recording_dir, f"{channel}.csv"), delimiter=",", header=None
    ).values

    unix_t0, sampling_rate, data = None, -1.0, None
    if channel == "IBI":
        unix_t0 = raw_data[0, 0]
        data = raw_data[1:]
    else:
        unix_t0 = raw_data[0] if raw_data.ndim == 1 else raw_data[0, 0]
        sampling_rate = raw_data[1] if raw_data.ndim == 1 else raw_data[1, 0]
        data = raw_data[2:]
    assert sampling_rate.is_integer(), "sampling rate must be an integer"
    data = np.squeeze(data)
    return int(unix_t0), int(sampling_rate), data.astype(np.float32)


def pad(args, data: np.ndarray, sampling_rate: int):
    """
    Upsample channel whose sampling rate is lower than args.time_alignment
    """

    # trim additional recordings that does not make up a second.
    data = data[: data.shape[0] - (data.shape[0] % sampling_rate)]

    s_shape = [data.shape[0] // sampling_rate, sampling_rate]
    p_shape = [s_shape[0], args.time_alignment]  # padded shape
    o_shape = [s_shape[0] * args.time_alignment]  # output shape
    if len(data.shape) > 1:
        s_shape.extend(data.shape[1:])
        p_shape.extend(data.shape[1:])
        o_shape.extend(data.shape[1:])
    # reshape data s.t. the 1st dimension corresponds to one second
    s_data = np.reshape(data, newshape=s_shape)

    # calculate the padding value
    if args.padding_mode == "zero":
        pad_value = 0
    elif args.padding_mode == "last":
        pad_value = s_data[:, -1, ...]
        pad_value = np.expand_dims(pad_value, axis=1)
    elif args.padding_mode == "average":
        pad_value = np.mean(s_data, axis=1, keepdims=True)
    elif args.padding_mode == "median":
        pad_value = np.median(s_data, axis=1, keepdims=True)
    else:
        raise NotImplementedError(
            f"padding_mode {args.padding_mode} has not been implemented."
        )

    padded_data = np.full(shape=p_shape, fill_value=pad_value, dtype=np.float32)
    padded_data[:, :sampling_rate, ...] = s_data
    padded_data = np.reshape(padded_data, newshape=o_shape)
    return padded_data


def pool(args, data: np.ndarray, sampling_rate: int):
    """
    Downsample channel whose sampling rate is greater than args.time_alignment
    """
    size = data.shape[0] - (data.shape[0] % sampling_rate)
    shape = (
        size // int(sampling_rate / args.time_alignment),
        int(sampling_rate / args.time_alignment),
    )
    if data.ndim > 1:
        shape += (data.shape[-1],)
    data = data[:size]
    new_data = np.reshape(data, newshape=shape)
    # apply pooling on the axis=1
    if args.downsampling == "average":
        new_data = np.mean(new_data, axis=1)
    elif args.downsampling == "max":
        new_data = np.max(new_data, axis=1)
    else:
        raise NotImplementedError(f"unknown downsampling method {args.downsampling}.")
    return new_data


def trim(data: np.ndarray, sampling_rate: int):
    """
    Trim, if necessary, tail of channel whose sampling rate is equal to
    args.time_alignment
    """
    size = data.shape[0] - (data.shape[0] % sampling_rate)
    return data[:size]


def resample(args, data: np.ndarray, sampling_rate: int):
    """
    Resample data so that channels are time aligned based on the required no of
    cycles per second (args.time_alignment)
    """
    ratio = args.time_alignment / sampling_rate
    if ratio > 1:
        new_data = pad(args, data, sampling_rate)
    elif ratio < 1:
        new_data = pool(args, data, sampling_rate)
    else:
        new_data = trim(data, sampling_rate)
    return new_data


def preprocess_channel(args, recording_dir: str, channel: str):
    """
    Load and downsample channel using args.downsampling s.t. each time-step
    corresponds to one second in wall-time
    """
    assert channel in CSV_CHANNELS and channel != "IBI"
    _, sampling_rate, data = load_channel(recording_dir=recording_dir, channel=channel)
    # transform to g for acceleration
    if channel == "ACC":
        data = data * 2 / 128
    # despike, apply filter on EDA and TEMP data
    # note: kleckner2018 uses a length of 2057 for a signal sampled at 32Hz, EDA from Empatica E4 is sampled at 4Hz (1/8)
    if (channel == "EDA") or (channel == "TEMP"):
        data = mne.filter.filter_data(
            data=data.astype("float"),
            sfreq=sampling_rate,
            l_freq=0,
            h_freq=0.35,
            filter_length=257,
            verbose=False,
        )
    # resample channel
    data = resample(args, data=data, sampling_rate=sampling_rate)
    if channel != "HR":
        # HR begins at t0 + 10s, remove first 10s from channels other than HR
        data = data[args.time_alignment * HR_OFFSET :]
    return data


def find_eda_slope(args, recording_dir: str, channel: str):
    """
    Compute eda slope for use in Kleckner et al. 2018 EDA and TEMP quality control
    """
    assert channel == "EDA"
    _, sampling_rate, data = load_channel(recording_dir=recording_dir, channel=channel)
    data = mne.filter.filter_data(
        data=data.astype("float"),
        sfreq=sampling_rate,
        l_freq=0,
        h_freq=0.35,
        filter_length=257,
        verbose=False,
    )
    # EDA is sampled at 4Hz with Empatica E4, in Kleckner et al. 2018, EDA slope
    # is given at muS/sec. EDA is thus down-sampled to one second unit
    shape = (floor(data.shape[0] / sampling_rate), int(sampling_rate))
    data = data[: np.prod(shape)]
    new_data = np.reshape(data, newshape=shape)
    new_data = np.mean(new_data, axis=1)
    # calculate point-by-point slope of EDA
    eda_slope = np.gradient(new_data)
    eda_slope = eda_slope[HR_OFFSET:]

    return np.repeat(eda_slope, repeats=args.time_alignment, axis=0)


def find_ibi_gaps(ibi: np.ndarray):
    """
    Find gaps in IBI channel and assign the first column in data to indicate if
    a gap exists
    """
    times, intervals = ibi[:, 0], ibi[:, 1]
    # manually compute the delta between the time of consecutive inter-beat
    deltas = np.zeros_like(intervals)
    deltas[1:] = np.diff(times)
    # compare the manually computed deltas against the recorded intervals
    gaps = np.isclose(deltas, intervals)
    # assign 1 to indicate there is a gap and 0 otherwise
    gaps = np.logical_not(gaps).astype(np.float32)
    ibi = np.concatenate((gaps[:, np.newaxis], ibi), axis=1)
    return ibi


def clean_and_interpolate_ibi(args, timestamps: np.ndarray, durations: np.ndarray):
    """Fill NaN values in data with args.interpolation method"""

    # trim nan values if they appear at the edges (no extrapolation for IBI)
    is_nan = np.where(np.isnan(durations) == 0)
    idx_start, idx_end = is_nan[0][0], is_nan[0][-1] + 1
    timestamps = timestamps[idx_start:idx_end]
    durations = durations[idx_start:idx_end]

    # transform durations from seconds to milliseconds
    d = list(durations * 1000)

    # outliers from signal: remove RR intervals if not 300ms <= RR_duration <= 2000ms
    d_without_outliers = hrv.remove_outliers(
        rr_intervals=d, low_rri=300, high_rri=2000, verbose=False
    )

    # interpolate durations
    d_interpolated = hrv.interpolate_nan_values(
        rr_intervals=d_without_outliers, interpolation_method=args.ibi_interpolation
    )

    # remove ectopic beats from signal
    nn_intervals_list = hrv.remove_ectopic_beats(
        rr_intervals=d_interpolated, method="malik", verbose=False
    )

    # replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = hrv.interpolate_nan_values(
        rr_intervals=nn_intervals_list
    )

    output = np.zeros(shape=(len(interpolated_nn_intervals), 2), dtype=np.float32)
    output[:, 0] = timestamps
    output[:, 1] = interpolated_nn_intervals
    return output


def fill_ibi_gaps(args, ibi: np.ndarray, t_start: int, t_end: int):
    """
    Remove outliers and ectopic beats, interpolate IBI data (if missing
    values at the edges, these are cropped and corresponding seconds removed
    form all channels) and fill gaps using method specified in
    args.ibi_interpolation
    """
    # crop values that exceed t_end
    if ibi[-1, 1] > t_end:
        ibi = ibi[: np.where(ibi[..., 1] > t_end)[0][0]]

    timestamps, durations = [], []
    # append NaN values if gap exists otherwise the recorded value
    for i in range(ibi.shape[0]):
        if ibi[i, 0]:
            start = floor(timestamps[-1]) + 1 if i else t_start
            end = floor(ibi[i, 1])
            timestamps.extend(list(range(start, end)))
            durations.extend([np.nan] * (end - start))
        else:
            timestamps.append(ibi[i, 1])
            durations.append(ibi[i, 2])
    # append NaN to the end of the IBI recording to match the other channels
    if ceil(timestamps[-1]) < t_end:
        start, end = floor(timestamps[-1]) + 1, t_end + 1
        timestamps.extend(list(range(start, end)))
        durations.extend([np.nan] * (end - start))

    assert len(timestamps) == len(durations)
    assert ceil(timestamps[-1]) == t_end
    timestamps = np.array(timestamps, dtype=np.float32)
    durations = np.array(durations, dtype=np.float32)
    ibi = clean_and_interpolate_ibi(args, timestamps=timestamps, durations=durations)

    return ibi


def compute_hrv_features(args, ibi: np.ndarray):
    """
    Compute heart rate variability from IBI
    reference: https://www.biopac.com/application/ecg-cardiology/advanced-feature/rmssd-for-hrv-analysis/
    """
    assert 1 <= args.hrv_length < args.segment_length
    # HR_OFFSET = first_second if no missing values at the beginning of IBI,
    # otherwise first_second > HR_OFFSET
    start = 0
    times, recordings = ibi[:, 0], ibi[:, 1]
    window_end = max(HR_OFFSET + args.hrv_length, floor(times[0]) + args.hrv_length)

    count, hrv_data = 0, {}
    while times[start] + args.hrv_length <= ceil(times[-1]):
        end = np.where(times > window_end)[0][0]
        features = hrv.get_time_domain_features(recordings[start:end].tolist())
        update_dict(hrv_data, features)
        window_end += args.hrv_length
        start = end
        count += 1

    # get the remaining time domain features
    features = hrv.get_time_domain_features(recordings[start:].tolist())
    update_dict(hrv_data, features)
    # compute the remaining size needed
    remaining_length = (ceil(times[-1]) - floor(times[0])) - (count * args.hrv_length)

    repeat = lambda a, reps: np.reshape(
        np.tile(np.expand_dims(a, axis=-1), reps=(1, reps)), newshape=(-1)
    )

    for k, v in hrv_data.items():
        v = np.expand_dims(v, axis=-1)
        hrv_data[k] = np.concatenate(
            [
                repeat(v[:-1], args.hrv_length),
                repeat(v[-1], remaining_length),
            ],
            axis=0,
            dtype=np.float32,
        )
    return hrv_data


def preprocess_ibi(args, recording_dir: str, t_start: int, t_end: int):
    """Preprocess IBI data and convert to HRV recordings"""
    _, _, ibi = load_channel(recording_dir=recording_dir, channel="IBI")
    ibi = find_ibi_gaps(ibi)
    ibi = fill_ibi_gaps(args, ibi=ibi, t_start=t_start, t_end=t_end)
    hrv_data = compute_hrv_features(args, ibi=ibi)
    return hrv_data, floor(ibi[0, 0]), ceil(ibi[-1, 0])


def remove_zeros(recordings: np.ndarray, threshold: int = 5) -> t.List[np.ndarray]:
    """
    Remove recordings where all channels contain 0s for longer than threshold
    time-steps
    Args:
      recordings: np.ndarray
      threshold: int, the threshold (in time-steps) where channels can contain 0s
    Return:
      features: np.ndarray, filtered recordings where 0 index one continuous recording
    """
    assert 0 < threshold < recordings.shape[0]
    sums = np.sum(np.abs(recordings), axis=-1)
    features = []
    start, end = 0, 0
    while end < sums.shape[0]:
        if sums[end] == 0:
            current = end
            while end < sums.shape[0] and sums[end] == 0:
                end += 1
            if end - current >= threshold:
                features.append(recordings[start:current, ...])
                start = end
        end += 1
    if start + 1 < end:
        features.append(recordings[start:end, ...])
    return features


def eda_temp_hr_quality_control(
    args,
    recordings: np.ndarray,
    eda_slope: np.ndarray,
    channel_names: t.List[str],
):
    """
    Apply Kleckner et al. 2018 quality control to EDA and TEMP plus new rule on HR
    see Figure 1 in https://pubmed.ncbi.nlm.nih.gov/28976309/
    Returns:
        features: np.ndarray, the filtered recordings where 0 index is one continuous recording
    """
    assert (
        np.isnan(recordings).any() == False
    ), "NaN values detected prior to applying Kleckner et al. 2018"

    eda_index = channel_names.index("EDA")
    temp_index = channel_names.index("TEMP")
    hr_index = channel_names.index("HR")

    # Rule 1) EDA not within 0.05 - 60 muS
    recordings[:, eda_index][
        (recordings[:, eda_index] < 0.05) + (recordings[:, eda_index] > 60)
    ] = np.nan

    # Rule 2) EDA slope not within -10 - 10 muS/sec
    eda_slope[(eda_slope < -10) + (eda_slope > 10)] = np.nan

    # Rule 3) TEMP not within 30 - 40 °C
    recordings[:, temp_index][
        (recordings[:, temp_index] < 30) + (recordings[:, temp_index] > 40)
    ] = np.nan

    # Rule 4) EDA surrounded (within 5 sec) by invalid data according to the rules above
    rule4 = np.empty_like(prototype=recordings[:, eda_index])
    assert len(recordings) > 5 * args.time_alignment, (
        "recording is shorter than 5 seconds, cannot apply rule 4) from "
        "Kleckner et al. 2018"
    )

    for i in range(recordings.shape[0]):
        if not np.isnan(recordings[i, eda_index]):
            if i - (5 * args.time_alignment) < 0:
                lower_idx = 0
                upper_idx = 5 * args.time_alignment * 2
            if (i - (5 * args.time_alignment) > 0) and (
                i + (5 * args.time_alignment) < len(recordings)
            ):
                lower_idx = i - (5 * args.time_alignment)
                upper_idx = i + (5 * args.time_alignment)
            if i + (5 * args.time_alignment) > len(recordings):
                upper_idx = len(recordings)
                lower_idx = upper_idx - (5 * args.time_alignment * 2)

            if (
                (np.isnan(recordings[:, eda_index][lower_idx:upper_idx]).any())
                or (np.isnan(eda_slope[lower_idx:upper_idx]).any())
                or (np.isnan(recordings[:, eda_index][lower_idx:upper_idx]).any())
            ):
                rule4[i] = np.nan
            else:
                rule4[i] = recordings[i, eda_index]
        else:
            rule4[i] = recordings[i, eda_index]

    # Rule 5) HR not within 25 - 250 bpm (note this is not from Kleckner2018)
    recordings[:, hr_index][
        (recordings[:, hr_index] < 25) + (recordings[:, hr_index] > 250)
    ] = np.nan

    # drop rows where at least one column has NaN and collect sub-recordings for
    # use in segmentation
    augmented_recordings = np.concatenate((recordings, rule4[..., np.newaxis]), axis=1)

    features = []
    start, end = 0, 0
    while end < len(augmented_recordings) - 1:
        if np.isnan(augmented_recordings[end]).any():
            end += 1
            start = end
        else:
            while not np.isnan(augmented_recordings[end]).any():
                if end < len(augmented_recordings) - 1:
                    end += 1
                else:
                    features.append(augmented_recordings[start:, :-1])
                    break
            else:
                features.append(augmented_recordings[start:end, :-1])
                start = end

    # Check that features length and valid sub-recordings length are the same
    assert sum([len(i) for i in features]) == len(
        augmented_recordings[~np.isnan(augmented_recordings).any(axis=1), :]
    )

    return (
        features,
        (
            np.sum(np.isnan(np.sum(augmented_recordings, axis=1)))
            / augmented_recordings.shape[0]
        )
        * 100,
    )


def preprocess_dir(args, recording_dir: str, clinical_info: pd.DataFrame):
    """
    Preprocess channels in recording_dir and return the preprocessed features
    and corresponding label obtained from spreadsheet.
    Returns:
      features: np.ndarray, preprocessed channels in SAVE_CHANNELS format
      label: List[int], label from clinical spreadsheet in format
            [session ID, is patient, timing, YMRS(1 - 11), HDRS(1 - 17)]
    """
    eda_slope, channel_data, min_length = None, {}, np.inf

    # load and preprocess all channels except IBI
    for channel in CSV_CHANNELS:
        if channel != "IBI":
            channel_data[channel] = preprocess_channel(
                args, recording_dir=recording_dir, channel=channel
            )
            if len(channel_data[channel]) < min_length:
                min_length = len(channel_data[channel])
            if channel == "EDA":
                eda_slope = find_eda_slope(
                    args, recording_dir=recording_dir, channel=channel
                )

    # crop each channel to min_length
    for channel in channel_data.keys():
        channel_data[channel] = channel_data[channel][:min_length]
    eda_slope = eda_slope[:min_length]

    min_length = ceil(min_length / args.time_alignment)

    if args.hrv_features:
        # HR_OFFSET was cropped from start of channels != IBI so in order to
        # get the last second in real time HR_OFFSET should be added to min_length
        hrv_data, first_ibi_second, last_ibi_second = preprocess_ibi(
            args,
            recording_dir=recording_dir,
            t_start=HR_OFFSET,
            t_end=min_length + HR_OFFSET,
        )

        if args.hrv_features == "all":
            args.hrv_features = HRV_FEATURES
        for hrv_features in args.hrv_features:
            channel_data[f"HRV_{hrv_features}"] = np.repeat(
                hrv_data[hrv_features], repeats=args.time_alignment, axis=0
            )

        # IBI missing values at the edges (if any) are not extrapolated
        # the corresponding seconds should therefore be removed from channels != IBI
        idx_start = 0
        if first_ibi_second > HR_OFFSET:
            non_ibi_leading_seconds_to_crop = int(first_ibi_second - HR_OFFSET)
            idx_start = non_ibi_leading_seconds_to_crop * args.time_alignment

        idx_end = int(min_length * args.time_alignment)
        if min_length + HR_OFFSET > last_ibi_second:
            non_ibi_trailing_seconds_to_crop = int(
                (min_length + HR_OFFSET) - last_ibi_second
            )
            trailing_row_to_drop = (
                non_ibi_trailing_seconds_to_crop * args.time_alignment
            )
            idx_end = int((min_length * args.time_alignment) - trailing_row_to_drop)

        for c in channel_data.keys():
            if not c.startswith("HRV"):
                channel_data[c] = channel_data[c][idx_start:idx_end]

    # combine all preprocessed channels to a single array
    features = np.column_stack([channel_data[n] for n in channel_data.keys()])

    session_info = {
        "min": np.min(features, axis=0),
        "max": np.max(features, axis=0),
        "mean": np.mean(features, axis=0),
        "std": np.std(features, axis=0),
        "channel_names": utils.get_channel_names(channel_data),
    }

    # filter features
    if args.filter_mode == 0:
        features = np.expand_dims(features, axis=0)
    elif args.filter_mode == 1:
        features = remove_zeros(recordings=features)
    elif args.filter_mode == 2:
        features, percent_filtered = eda_temp_hr_quality_control(
            args,
            recordings=features,
            eda_slope=eda_slope,
            channel_names=session_info["channel_names"],
        )
    else:
        raise NotImplementedError(
            f"filter mode {args.filter_mode} has not been implemented."
        )

    # get label information
    session_id = int(os.path.basename(recording_dir))
    update_dict(
        session_info,
        get_session_info(session_id=session_id, clinical_info=clinical_info),
        replace=True,
    )
    label = (
        [
            session_id,
            int(session_info["is_patient"]),
            session_info["timing"],
        ]
        + session_info["YMRS"]
        + session_info["HDRS"]
    )

    return (
        features,
        label,
        session_info,
        percent_filtered if args.filter_mode == 2 else 0,
    )


def preprocess(args):
    clinical_info = read_clinical_info(
        os.path.join(FILE_DIRECTORY, "TIMEBASE_database.xlsx")
    )

    args.segment_length *= args.time_alignment

    data, sessions_info, sessions_qc_percentage, channel_names, = (
        {},
        {},
        {},
        None,
    )
    for session_ids in tqdm(
        args.session2class.keys(), desc="Preprocessing", disable=args.verbose == 0
    ):
        features, label = [], None
        for session_id in session_ids:
            recording_dir = utils.unzip_session(args.dataset, session_id=session_id)
            s_features, s_label, session_info, percent_filtered = preprocess_dir(
                args, recording_dir=recording_dir, clinical_info=clinical_info
            )
            sessions_info[session_id] = session_info
            features.extend(s_features)
            if label is None:
                label = s_label
            if channel_names is None:
                channel_names = session_info["channel_names"]
            sessions_qc_percentage[session_id] = percent_filtered
        session_data = create_pairs(
            args, features=features, label=label, session_ids=session_ids
        )
        update_dict(target=data, source=session_data)

    data = {k: np.concatenate(v) for k, v in data.items()}

    data["x_train"], data["y_train"] = utils.shuffle(data["x_train"], data["y_train"])
    data["x_val"], data["y_val"] = utils.shuffle(data["x_val"], data["y_val"])
    data["x_test"], data["y_test"] = utils.shuffle(data["x_test"], data["y_test"])

    ds_info = {
        "channel_names": channel_names,
        "ds_sizes": [len(data["y_train"]), len(data["y_val"]), len(data["y_test"])],
        "input_shape": data["x_train"].shape[1:],
        "sessions_info": sessions_info,
        "sessions_qc_percentage": sessions_qc_percentage,
        "label_scale": LABEL_SCALE,
        "downsampling": args.downsampling,
        "time_alignment": args.time_alignment,
        "padding_mode": args.padding_mode,
        "filter_mode": args.filter_mode,
        "ibi_interpolation": args.ibi_interpolation,
        "hrv_features": args.hrv_features,
        "hrv_length": args.hrv_length,
        "segment_length": args.segment_length,
        "test_size": args.test_size,
    }

    return data, ds_info
