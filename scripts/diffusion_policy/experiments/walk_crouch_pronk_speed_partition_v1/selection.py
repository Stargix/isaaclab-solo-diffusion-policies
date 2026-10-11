"""Pure anchor eligibility: inspect physical context, not requested speed alone."""
import numpy as np

HISTORY = 8
HORIZON = 100
FUTURE_ACTIONS = 8
DT = .02
WALK_MIN_MPS = .6
PRONK_MAX_MPS = .5
LOCAL_SPEED_TICKS = 10  # 0.20 s; do not threshold every gait-oscillating 20 ms tick
SETTLING_TICKS = 25  # exclude first 0.50 s after each command change


def eligible_anchors(root_pos, command, skill):
    xy = np.asarray(root_pos, dtype=np.float64)[:, :2]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    local = (arc[LOCAL_SPEED_TICKS:] - arc[:-LOCAL_SPEED_TICKS]) / (LOCAL_SPEED_TICKS * DT)
    averages = (arc[HORIZON:] - arc[:-HORIZON]) / (HORIZON * DT)
    step_speed = np.diff(arc) / DT
    changed = np.r_[True, np.any(np.diff(command, axis=0) != 0, axis=1)]
    age = np.arange(len(command)) - np.maximum.accumulate(np.where(changed, np.arange(len(command)), 0))
    stop = min(len(xy) - HORIZON + 1, len(xy) - FUTURE_ACTIONS + 1)
    candidates = np.arange(stop, dtype=np.int64)
    if skill == 'crouch':
        return candidates
    accepted = []
    for anchor in candidates:
        start = max(0, int(anchor) - HISTORY)
        latest = max(0, int(anchor) - 1)
        # Rolling goals for every historical state, including repeated state0
        # of a padded start. Last future goal ends at latest + HORIZON.
        if skill == 'pronk':
            if np.all(averages[start:latest + 1] <= PRONK_MAX_MPS):
                accepted.append(anchor)
            continue
        if skill != 'walk':
            raise ValueError(f'Unknown expert: {skill}')
        if anchor < HISTORY + 1:
            continue  # no padded walk start, including no synthetic past actions
        physical_start = start - 1  # delayed action history begins one tick earlier
        if age[physical_start] < SETTLING_TICKS:
            continue
        end = latest + HORIZON
        if not np.all(averages[start:latest + 1] >= WALK_MIN_MPS):
            continue
        # Check the WHOLE union of histories and 2 s futures, not just the
        # action at the anchor. This includes the 16-action training target.
        if not np.all(local[physical_start:end - LOCAL_SPEED_TICKS + 1] >= WALK_MIN_MPS):
            continue
        if not np.all(np.linalg.norm(command[physical_start:end, :2], axis=1) >= WALK_MIN_MPS):
            continue  # no coasting after a stop/low command
        if np.any(step_speed[physical_start:end] <= .02):
            continue  # no actual near-rest tick, even if smoothed speed is high
        accepted.append(anchor)
    return np.asarray(accepted, dtype=np.int64)
