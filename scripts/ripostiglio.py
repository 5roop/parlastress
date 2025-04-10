SAMPLE_RATE = 16_000
FRAME_RATE = 50


def events_to_frames(
    segment_start: float,
    segment_end: float,
    stress_start: float,
    stress_end: float,
    twodimensional=True,
):
    import numpy as np

    duration = segment_end - segment_start
    stress_start_i = int(round((stress_start - segment_start) * FRAME_RATE, 0))
    stress_end_i = int(round((stress_end - segment_start) * FRAME_RATE, 0))
    num_elements = int(FRAME_RATE * duration)
    if not twodimensional:
        labels = np.zeros(num_elements, dtype=np.int8)
        labels[stress_start_i:stress_end_i] = 1
        return labels
    labels = np.zeros((num_elements, 2), dtype=np.int8)
    labels[:, 0] = 1
    labels[stress_start_i:stress_end_i] = [0, 1]

    return labels


def frames_to_events(frames: list) -> list[tuple]:
    import pandas as pd
    from itertools import pairwise

    results = []
    ndf = pd.DataFrame(
        data={
            "time_s": [0.020 * i for i in range(len(frames))],
            "frames": frames,
        }
    )
    ndf = ndf.dropna()
    indices_of_change = ndf.frames.diff()[ndf.frames.diff() != 0].index.values
    for si, ei in pairwise(indices_of_change):
        if ndf.loc[si : ei - 1, "frames"].mode()[0] == 0:
            pass
        else:
            results.append(
                (round(ndf.loc[si, "time_s"], 3), round(ndf.loc[ei, "time_s"], 3))
            )
    return results


2 + 2


import pandas as pd


def events_overlap(this: pd.Interval, other: pd.Interval):
    import pandas as pd

    if isinstance(this, list):
        this = pd.Interval(left=this[0], right=this[1])
    if isinstance(other, list):
        other = pd.Interval(left=other[0], right=other[1])
    return (this.right > other.left) and (this.left < other.right)


def extract(gold_intervals: list[pd.Interval], pred_intervals: list[pd.Interval]):
    import pandas as pd

    TP, FN, FP = [], [], []
    pred_intervals = [i + 0.001 for i in pred_intervals]
    events = gold_intervals + pred_intervals
    inhibited_events = []
    for event in events:
        if event in inhibited_events:
            continue
        others = [e for e in events if ((e not in inhibited_events) and (e != event))]
        overlapping = [other for other in others if events_overlap(event, other)]
        if not overlapping:
            if event in gold_intervals:
                FN.append(event)
                inhibited_events.append(event)
            else:
                FP.append(event)
                inhibited_events.append(event)
        else:
            other = overlapping[0]
            TP.append(
                pd.Interval(
                    min(event.left, other.left),
                    max(event.right, other.right),
                )
            )
            inhibited_events.append(event)
            inhibited_events.append(other)

    assert len(TP) * 2 + len(FN) + len(FP) == len(gold_intervals) + len(pred_intervals)
    return TP, FN, FP
