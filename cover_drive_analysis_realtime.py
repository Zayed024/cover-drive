import os, time, json, cv2, numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

import math
import csv
import statistics
from typing import Tuple, List, Dict

# Load our config file - just basic settings for the analysis
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

# Some helper functions for calculating angles and distances
def calculate_angle(a, b, c):
    """Calculate the angle ABC in degrees. Just basic geometry stuff."""
    a = np.array(a, dtype=float); b = np.array(b, dtype=float); c = np.array(c, dtype=float)
    ab = a - b; cb = c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb))
    if denom == 0: return np.nan
    cosang = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def calculate_foot_angle(ankle, foot_index):
    """Figure out which way the foot is pointing relative to the x-axis."""
    ax, ay = ankle; fx, fy = foot_index
    vx, vy = (fx - ax), (fy - ay)
    if vx == 0 and vy == 0: return np.nan
    ang = np.degrees(np.arctan2(vy, vx))
    ang = abs(ang)  # we just need magnitude vs x-axis
    if ang > 180: ang = 360 - ang
    return float(ang)

def _safe_text(val, fmt="{:.0f}"):
    """Handle NaN values when displaying text - show N/A instead of crashing."""
    return "N/A" if (val is None or np.isnan(val)) else fmt.format(val)


def _nan_to_num(a, fill=0.0):
    """Replace NaN values with a default number. Useful for plotting."""
    a = np.asarray(a, dtype=float)
    a[np.isnan(a)] = fill
    return a

def smooth_mavg(x, win=7):
    """Smooth out the data using a moving average. Helps reduce noise in the angles."""
    x = np.asarray(x, dtype=float)
    if win <= 1 or len(x) == 0: return x
    mask = ~np.isnan(x)
    x0 = np.where(mask, x, 0.0)
    k = np.ones(win, dtype=float)
    num = np.convolve(x0, k, mode="same")
    den = np.convolve(mask.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        y = num / den
    y[den == 0] = np.nan
    return y

def deriv(x, fps):
    """Calculate how fast things are changing - the derivative. Useful for finding sudden movements."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2: return np.zeros_like(x)
    dx = np.gradient(np.where(np.isnan(x), np.nanmean(x), x)) * fps
    return dx

def _normalize01(a):
    """Scale values to be between 0 and 1. Makes it easier to compare different metrics."""
    a = np.asarray(a, dtype=float)
    m = np.nanmin(a); M = np.nanmax(a)
    if not np.isfinite(m) or not np.isfinite(M) or M - m == 0:
        return np.zeros_like(a)
    return (a - m) / (M - m)

def _last_local_min_before(arr, idx, guard=3):
    """Find the last time the values were at a local minimum before a certain point."""
    arr = np.asarray(arr, dtype=float)
    if idx <= 1: return 0
    s = arr[:idx]
    if len(s) < 3: return 0
    # simple local mins
    mins = [i for i in range(1, len(s)-1) if s[i] <= s[i-1] and s[i] <= s[i+1]]
    mins = [i for i in mins if i < idx-guard]
    return mins[-1] if mins else int(max(0, np.nanargmin(s[:idx])))


# Now let's figure out the bat path and how well the player is swinging


def fit_line_least_squares(points: List[Tuple[float,float]]):
    """
    Fit a straight line through the wrist positions to see the bat path.
    Returns the slope and intercept, plus the direction the bat is moving.
    """
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if len(xs) < 2:
        return None
    A = np.vstack([xs, np.ones_like(xs)]).T
    m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
    # direction vector (dx, dy) normalized
    dx = 1.0 / math.sqrt(1 + m*m)
    dy = m * dx
    return (float(m), float(c), float(dx), float(dy))

def point_line_distance(px, py, m, c):
    # How far is this point from our bat line? Basic geometry.
    if not np.isfinite(m) or not np.isfinite(c): return np.nan
    return abs(m*px - py + c) / math.sqrt(m*m + 1)

def estimate_bat_line_metrics(metrics: Dict, downswing_frame: int, impact_frame: int, cfg: Dict):
    """
    Figure out how straight the bat path is during the swing.
    Returns info about the bat line quality and how much the player deviates from it.
    """
    pts = []
    n = len(metrics.get("wrist_px", []))
    after = cfg.get("batline", {}).get("use_frames_after_impact", 3)
    start = max(0, downswing_frame)
    end = min(n-1, impact_frame + after)
    for i in range(start, end+1):
        w = metrics["wrist_px"][i]
        if w is None: continue
        if np.isnan(w[0]) or np.isnan(w[1]): continue
        pts.append((float(w[0]), float(w[1])))

    if len(pts) < cfg.get("batline", {}).get("min_points_for_line", 6):
        return {"line_params": None, "rms_dev": np.nan, "line_angle_deg": np.nan, "num_points": len(pts)}

    m, c, dx, dy = fit_line_least_squares(pts)
    # distances
    dists = [point_line_distance(x,y,m,c) for (x,y) in pts]
    rms = float(np.sqrt(float(np.mean(np.array(dists)**2))))
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    return {"line_params": (m,c,dx,dy), "rms_dev": rms, "line_angle_deg": angle_deg, "num_points": len(pts)}

def draw_bat_line_on_frame(frame, line_params, color=(0,128,255), thickness=2):
    """Draw the bat path line on the video frame so we can see how straight the swing is."""
    if line_params is None: return frame
    m, c, dx, dy = line_params
    h, w = frame.shape[:2]
    x0, x1 = 0, w
    y0 = int(m * x0 + c)
    y1 = int(m * x1 + c)
    cv2.line(frame, (x0, max(0,min(h-1,y0))), (x1, max(0,min(h-1,y1))), color, thickness)
    return frame

# Let's compare this player's performance to some reference data
import pandas as pd
def load_reference_stats(csv_path: str):
    """
    Load data from other players to compare against. CSV should have columns for each metric.
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print("Failed to load reference CSV:", e)
        return None

def benchmark_metrics(metrics_summary: Dict, cfg: Dict, ref_df = None):
    """
    Compare this player's metrics to reference data or ideal values.
    Returns scores and percentiles to see how they stack up.
    """
    out = {}
    if ref_df is not None:
        # compute percentiles
        for k in ["elbow_at_impact","foot_angle","head_alignment","batline_rms","wrist_speed"]:
            if k not in ref_df.columns or metrics_summary.get(k) is None or np.isnan(metrics_summary.get(k)):
                out[k] = {"percentile": None}
                continue
            ref_vals = ref_df[k].dropna().values
            pct = float((ref_vals < metrics_summary[k]).sum()) / max(1, len(ref_vals)) * 100.0
            out[k] = {"percentile": pct}
        # convert to 0-10 score by mapping percentile to decile (higher percentile => higher score)
        scores = []
        for k in out:
            pct = out[k].get("percentile")
            if pct is None: s = 5.0
            else: s = max(0.0, min(10.0, pct/10.0))
            out[k]["score_0_10"] = s
            scores.append(s)
        overall = float(np.mean(scores)) if scores else 5.0
        out["overall_score"] = overall
        return out
    else:
        # use ideal_template
        ideal = cfg.get("ideal_template", {})
        scores = []
        for k, mean_key, std_key in [
            ("elbow_at_impact","elbow_at_impact_mean","elbow_at_impact_std"),
            ("foot_angle","foot_angle_mean","foot_angle_std"),
            ("head_alignment","head_alignment_mean","head_alignment_std"),
            ("batline_rms","batline_rms_mean","batline_rms_std"),
            ("wrist_speed","wrist_speed_mean","wrist_speed_std")
        ]:
            val = metrics_summary.get(k)
            mean = ideal.get(mean_key)
            std = ideal.get(std_key) or 1.0
            if val is None or np.isnan(val) or mean is None:
                score = 5.0
            else:
                z = abs((val - mean) / std)
                # map z=0 -> 10, z=3 -> ~0
                score = max(0.0, min(10.0, 10.0 * math.exp(-0.5 * (z**2) / (1.0**2))))
            out[k] = {"score_0_10": float(score)}
            scores.append(score)
        out["overall_score"] = float(np.mean(scores))
        return out



# Now let's create some charts and reports to visualize the data
def generate_smoothness_chart(elbow_angles, output_path):
    """Plot the elbow angles over time to see how smooth the swing is."""
    y = np.array(elbow_angles, dtype=float)
    x = np.arange(len(y))
    # ignore NaNs for plotting
    plt.figure(figsize=(8, 4))
    plt.plot(x[~np.isnan(y)], y[~np.isnan(y)], label="Elbow Angle")
    plt.xlabel("Frame"); plt.ylabel("Angle (deg)")
    plt.title("Elbow Angle Smoothness")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_path); plt.close()


def plot_bat_line(pts, line_params, outpath):
    """Show the wrist path and how well it fits a straight line."""
    plt.figure(figsize=(6,8))
    if len(pts) == 0:
        # empty plot placeholder
        plt.text(0.5, 0.5, "No wrist points available", ha="center", va="center")
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.savefig(outpath); plt.close()
        return

    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    plt.scatter(xs, ys, s=6, label="Wrist points")
    if line_params:
        m, c, dx, dy = line_params
        x0 = np.linspace(xs.min() - 10, xs.max() + 10, 200)
        plt.plot(x0, m * x0 + c, '-', linewidth=2, color='orange', label="Fitted bat-line")
    plt.gca().invert_yaxis()
    plt.xlabel("X (px)"); plt.ylabel("Y (px)")
    plt.title("Wrist path & bat-line fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath); plt.close()


def generate_html_report(evaluation, chart_path, output_path, batline_path=None):
    """Create a nice HTML report showing all the analysis results."""
    smooth = evaluation.get("smoothness", {})
    sd = smooth.get("std_dev", float("nan"))
    rms = smooth.get("rms_jerk", float("nan"))
    score = smooth.get("smoothness_score", float("nan"))

    with open(output_path, "w") as f:
        f.write("<html><body><h1>Cover Drive Analysis Report</h1>")
        for cat in ["Footwork", "Head Position", "Swing Control", "Balance", "Follow-through"]:
            det = evaluation.get(cat, {"score":"-", "feedback":"-"})
            f.write(f"<h2>{cat}</h2><p>Score: {det['score']}/10</p><p>Feedback: {det['feedback']}</p>")

        f.write(f"<h2>Overall Grade: {evaluation.get('skill_grade','N/A')}</h2>")

        # Smoothness block
        f.write("<h2>Smoothness Evaluation</h2>")
        f.write(f"<p>Elbow Angle Std Dev: {sd:.2f} deg</p>")
        f.write(f"<p>RMS Jerk: {rms:.2f} deg/s³</p>")
        f.write(f"<p>Smoothness Score (0–10): {score:.1f}</p>")
        if chart_path and os.path.exists(chart_path):
            f.write(f"<img src='{os.path.basename(chart_path)}' width='600'>")

        if chart_path and os.path.exists(chart_path):
            f.write(f"<h3>Elbow smoothness</h3><img src='{os.path.basename(chart_path)}' width='600'>")
        if batline_path and os.path.exists(batline_path):
            f.write(f"<h3>Wrist path & bat-line fit</h3><img src='{os.path.basename(batline_path)}' width='400'>")
        f.write("</body></html>")
        

# Here's the cool part - we can detect when the swing starts and when the bat hits the ball
def find_swing_phases_final(metrics):
    """
    Figure out when the downswing starts and when impact happens.
    Uses wrist speed and how fast the angles are changing to detect these key moments.
    Returns: (downswing_frame, impact_frame)
    """
    fps   = metrics.get("fps", 30.0)
    speed = np.asarray(metrics.get("wrist_speeds", []), dtype=float)
    axis  = np.asarray(metrics.get("forearm_axis_angles", []), dtype=float)
    elbow = np.asarray(metrics.get("elbow_angles", []), dtype=float)

    n = len(speed)
    if n == 0:
        return 0, 0

    # Smooth out the signals to reduce noise
    win = max(5, int(round(fps * 0.08)))     # ~80 ms window
    s_s = smooth_mavg(speed, win=win)
    a_s = smooth_mavg(axis,  win=max(5, int(round(fps * 0.12))))
    e_s = smooth_mavg(elbow, win=max(5, int(round(fps * 0.12))))

    # Primary contact cue = wrist speed peak
    if np.all(np.isnan(s_s)):
        impact = n // 2
    else:
        impact0 = int(np.nanargmax(s_s))

        # Refine within a small neighborhood using orientation/angle snap
        w = max(4, int(round(fps * 0.05)))   # ~50 ms neighborhood
        lo, hi = max(0, impact0 - w), min(n, impact0 + w + 1)

        # fast change around impact (|derivative|)
        axis_rate  = np.abs(deriv(a_s, fps))
        elbow_rate = np.abs(deriv(e_s, fps))

        # Normalize and combine
        comp = (
            1.0 * _normalize01(s_s) +
            0.6 * _normalize01(axis_rate) +
            0.4 * _normalize01(elbow_rate)
        )
        impact = int(np.nanargmax(comp[lo:hi])) + lo

    # Downswing start:
    # (a) last local min in speed before impact and below 35% of max,
    # else (b) elbow-angle apex (max) before impact, else (c) fallback a short window before impact.
    smax = np.nanmax(s_s) if np.size(s_s) else 0.0
    ds = _last_local_min_before(s_s, impact, guard=max(3, int(round(fps*0.05))))
    if smax > 0 and s_s[ds] > 0.35 * smax:
        # too high; try elbow apex backup
        cand = np.nanargmax(e_s[:max(impact, 1)]) if impact > 1 else 0
        if np.isfinite(cand):
            ds = int(cand)

    # Final guardrails
    ds = int(np.clip(ds, 0, max(0, impact-1)))
    impact = int(np.clip(impact, ds+1, n-1))

    return ds, impact

def compute_smoothness(elbow_angles, fps):
    """
    Compute smoothness metrics from elbow angle time series.
    Returns dict with std_dev, rms_jerk, smoothness_score (0-10).
    """
    angles = np.array(elbow_angles, dtype=float)
    if len(angles) < 5:
        return {"std_dev": np.nan, "rms_jerk": np.nan, "smoothness_score": 5.0}

    # Remove NaNs
    mask = ~np.isnan(angles)
    angles = angles[mask]
    if len(angles) < 5:
        return {"std_dev": np.nan, "rms_jerk": np.nan, "smoothness_score": 5.0}

    # 1. Variability
    std_dev = float(np.std(angles))

    # 2. Angular velocity and jerk
    vel = np.gradient(angles) * fps         # deg/s
    acc = np.gradient(vel) * fps            # deg/s²
    jerk = np.gradient(acc) * fps           # deg/s³
    rms_jerk = float(np.sqrt(np.mean(jerk**2)))

    # 3. Normalize jerk → smoothness score (heuristic)
    # Assume jerk_rms ~1000 deg/s³ is very jerky (score ~0)
    scale = 1000.0
    score = float(max(0.0, min(10.0, 10.0 - (rms_jerk / scale) * 10.0)))

    return {"std_dev": std_dev, "rms_jerk": rms_jerk, "smoothness_score": score}


# Now let's evaluate the player's technique in different areas
def generate_evaluation(metrics, impact_frame, config):
    """Give the player feedback on their technique in different areas."""
    ev = {}

    # Robust indexing
    n = len(metrics["elbow_angles"])
    i = impact_frame if 0 <= impact_frame < n else n-1

    # Footwork: front-foot angle at impact
    foot = metrics["foot_angles"][i]
    if np.isnan(foot):
        ev["Footwork"] = {"score": 3, "feedback": "Couldn't read foot angle at impact; keep the front toe pointing towards the ball."}
    elif config["ideal_ranges"]["foot_angle_at_impact"]["min"] <= foot <= config["ideal_ranges"]["foot_angle_at_impact"]["max"]:
        ev["Footwork"] = {"score": 8, "feedback": "Good front-foot direction towards the off-side."}
    else:
        ev["Footwork"] = {"score": 5, "feedback": "Front foot a bit closed/open; point the toe more towards the ball."}

    # Head Position: head over front knee (px)
    head_knee = metrics["head_knee_alignments"][i]
    if np.isnan(head_knee):
        ev["Head Position"] = {"score": 3, "feedback": "Head alignment at impact not detected; keep head steady over the front knee."}
    elif head_knee <= config["ideal_ranges"]["head_alignment_at_impact"]["max"]:
        ev["Head Position"] = {"score": 9, "feedback": "Stable head position over the front knee at impact."}
    else:
        ev["Head Position"] = {"score": 5, "feedback": "Head falls away at impact; keep your eyes and head over the front knee."}

    # Swing Control: max elbow elevation up to impact
    upto = metrics["elbow_angles"][:i+1]
    mx = np.nanmax(upto) if len(upto) else np.nan
    if np.isnan(mx):
        ev["Swing Control"] = {"score": 4, "feedback": "Elbow elevation not tracked; aim for a high elbow through the downswing."}
    elif mx >= config["ideal_ranges"]["max_elbow_angle"]["min"]:
        ev["Swing Control"] = {"score": 9, "feedback": f"High elbow ({int(mx)}°) supporting a controlled downswing."}
    else:
        ev["Swing Control"] = {"score": 6, "feedback": f"Elbow a bit low ({int(mx)}°); lift it higher pre-impact."}

    # Balance: average spine lean (lower is better here)
    spine = np.array(metrics["spine_leans"], dtype=float)
    m = np.nanmean(spine) if spine.size else np.nan
    if np.isnan(m):
        ev["Balance"] = {"score": 4, "feedback": "Couldn't estimate spine lean; keep your torso upright and steady."}
    elif abs(m) <= 15:
        ev["Balance"] = {"score": 8, "feedback": "Good, stable spine through the swing."}
    else:
        ev["Balance"] = {"score": 5, "feedback": "Noticeable lateral lean; stay taller over the ball."}

    # Follow-through: end elbow extension heuristic
    end_elbow = metrics["elbow_angles"][-1] if n else np.nan
    if np.isnan(end_elbow):
        ev["Follow-through"] = {"score": 5, "feedback": "Follow-through not detected; allow the arms to extend fully."}
    elif end_elbow >= 140:
        ev["Follow-through"] = {"score": 8, "feedback": "Nice extension through the ball."}
    else:
        ev["Follow-through"] = {"score": 6, "feedback": "Finish is a touch short; extend your arms more after impact."}


    smoothness = compute_smoothness(metrics["elbow_angles"], metrics["fps"])
    ev["smoothness"] = smoothness

    # Skill grade (simple average)
    avg = np.mean([ev[k]["score"] for k in ["Footwork","Head Position","Swing Control","Balance","Follow-through"]])
    ev["skill_grade"] = "Advanced" if avg >= 8 else ("Intermediate" if avg >= 5 else "Beginner")
    return ev

# This is the main function that does all the work
def analyze_cover_drive(config, downswing_start=None, impact_start=None, write_video=True):
    """Analyze a cover drive video and generate all the metrics and feedback."""
    cap = cv2.VideoCapture(config["input_video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {config['input_video_path']}")

    os.makedirs(config["output_dir"], exist_ok=True)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # reliable across systems
        out = cv2.VideoWriter(
            os.path.join(config["output_dir"], config["output_video_name"]),
            fourcc, fps, (width, height)
        )

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=config["model_complexity"],
                        min_detection_confidence=config["pose_confidence"],
                        min_tracking_confidence=0.5)

    # Store all the metrics we calculate for each frame
    metrics = {
        "fps": fps,
        "elbow_angles": [],
        "spine_leans": [],
        "head_knee_alignments": [],
        "foot_angles": [],
        "wrist_px": [],              # wrist position in pixels
        "forearm_axis_angles": [],   # angle of forearm relative to x-axis
        "wrist_speeds": []           # how fast the wrist is moving
    }

    prev_wrist_px = None

    frames = 0
    start_time = time.time()
    log_interval = 30   # print progress every 30 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        elbow_angle = spine_lean = head_knee = foot_angle = np.nan
        wrist_pt = [np.nan, np.nan]

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            L = mp_pose.PoseLandmark
            def P(idx): return (lm[idx].x * width, lm[idx].y * height)

            # Get the key body points we need
            l_sh = (lm[L.LEFT_SHOULDER.value].x * width, lm[L.LEFT_SHOULDER.value].y * height)
            l_el = (lm[L.LEFT_ELBOW.value].x    * width, lm[L.LEFT_ELBOW.value].y    * height)
            l_wr = (lm[L.LEFT_WRIST.value].x    * width, lm[L.LEFT_WRIST.value].y    * height)
            l_hip = (lm[L.LEFT_HIP.value].x     * width, lm[L.LEFT_HIP.value].y     * height)
            l_knee = (lm[L.LEFT_KNEE.value].x   * width, lm[L.LEFT_KNEE.value].y   * height)
            l_ankle = (lm[L.LEFT_ANKLE.value].x * width, lm[L.LEFT_ANKLE.value].y * height)
            l_foot  = (lm[L.LEFT_FOOT_INDEX.value].x * width, lm[L.LEFT_FOOT_INDEX.value].y * height)
            nose    = (lm[L.NOSE.value].x * width, lm[L.NOSE.value].y * height)
            
            # Calculate all our biomechanical metrics
            elbow_angle = calculate_angle(l_sh, l_el, l_wr)
            spine_lean  = calculate_angle((l_hip[0], l_hip[1]-100), l_hip, l_sh)  # vs. vertical
            head_knee   = abs(nose[0] - l_knee[0])                # px
            foot_angle  = calculate_foot_angle(l_ankle, l_foot)   # deg

            # Figure out forearm orientation and wrist speed
            forearm_axis = np.degrees(np.arctan2(l_wr[1]-l_el[1], l_wr[0]-l_el[0]))
            wrist_speed  = 0.0
            if prev_wrist_px is not None:
                wrist_speed = np.linalg.norm(np.array(l_wr) - np.array(prev_wrist_px)) * fps
            prev_wrist_px = l_wr

            if write_video:
                mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Store all the metrics for this frame
        metrics["elbow_angles"].append(elbow_angle)
        metrics["spine_leans"].append(spine_lean)
        metrics["head_knee_alignments"].append(head_knee)
        metrics["foot_angles"].append(foot_angle)
        metrics["wrist_px"].append(wrist_pt)
        metrics["forearm_axis_angles"].append(forearm_axis)
        metrics["wrist_speeds"].append(wrist_speed)
        
        # Add overlays to the video if we're writing one
        if write_video:
            # Figure out what phase of the swing we're in
            phase = "Stance"
            if downswing_start and frames >= downswing_start and impact_start and frames < impact_start:
                phase = "Downswing"
            elif impact_start and frames >= impact_start:
                phase = "Follow-through"

            # Get the metrics for this frame (safely)
            def _get_metric(key, idx):
                try:
                    return metrics[key][idx]
                except Exception:
                    return np.nan

            elbow_angle = _get_metric("elbow_angles", frames)
            spine_lean = _get_metric("spine_leans", frames)
            head_knee = _get_metric("head_knee_alignments", frames)
            foot_angle = _get_metric("foot_angles", frames)

            # Draw the HUD showing real-time metrics
            cv2.putText(frame, f"Phase: {phase}", (30, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Elbow: {_safe_text(elbow_angle)} deg", (30, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Spine: {_safe_text(spine_lean)} deg", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Head-Knee: {_safe_text(head_knee)} px", (30, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Foot: {_safe_text(foot_angle)} deg", (30, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Highlight the impact moment with a red circle
            if impact_start and abs(frames - impact_start) <= 1:
                try:
                    wrist_x = int(metrics["wrist_px"][frames][0])
                    wrist_y = int(metrics["wrist_px"][frames][1])
                    cv2.circle(frame, (wrist_x, wrist_y), 10, (0,0,255), 3)
                except Exception:
                    pass
                cv2.putText(frame, "IMPACT", (30, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Give feedback at impact if head position is off
            if impact_start is not None and frames == impact_start and not np.isnan(head_knee):
                if head_knee > float(config["ideal_ranges"]["head_alignment_at_impact"]["max"]):
                    cv2.putText(frame, "Head not over front knee", (50, height-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            out.write(frame)

        frames += 1
        if frames % log_interval == 0:
            elapsed = time.time() - start_time
            fps_now = frames / elapsed if elapsed > 0 else 0
            print(f"[Progress] Frame {frames} | Current FPS: {fps_now:.2f}")

    cap.release(); pose.close()
    if write_video and out is not None: out.release()
    elapsed_total = time.time() - start_time
    fps_avg = frames / elapsed_total if elapsed_total > 0 else 0
    print(f"Processed {frames} frames in {elapsed_total:.2f}s "
          f"(Average FPS: {fps_avg:.2f})")
    
    return metrics

# This is where the script starts when you run it
if __name__ == "__main__":
    cfg = load_config()
    # First pass: just get the metrics without making a video
    M = analyze_cover_drive(cfg, write_video=False)
    ds, im = find_swing_phases_final(M)

    bat_metrics = estimate_bat_line_metrics(M, ds, im, cfg)

    # Create a plot showing the bat path
    pts = []
    n = len(M.get("wrist_px", []))
    after = cfg.get("batline", {}).get("use_frames_after_impact", 3)
    start = max(0, ds)
    end = min(n - 1, im + after)
    for i in range(start, end + 1):
        w = M["wrist_px"][i]
        try:
            if w is None: continue
            if np.isnan(w[0]) or np.isnan(w[1]): continue
            pts.append((float(w[0]), float(w[1])))
        except Exception:
            continue

    bat_plot_path = os.path.join(cfg["output_dir"], "batline_plot.png")
    plot_bat_line(pts, bat_metrics.get("line_params"), bat_plot_path)

    # Summarize the key metrics for this swing
    metrics_summary = {
        "elbow_at_impact": float(np.nan if len(M["elbow_angles"])<=im else M["elbow_angles"][im]),
        "foot_angle": float(np.nan if len(M["foot_angles"])<=im else M["foot_angles"][im]),
        "head_alignment": float(np.nan if len(M["head_knee_alignments"])<=im else M["head_knee_alignments"][im]),
         "batline_rms": bat_metrics.get("rms_dev"),
        "wrist_speed": float(np.nan if len(M["wrist_speeds"])<=im else M["wrist_speeds"][im])
    }
    # Load reference data if we have it
    ref_df = load_reference_stats(cfg["reference_stats_path"]) if cfg.get("reference_stats_path") else None
    benchmark = benchmark_metrics(metrics_summary, cfg, ref_df)

    # Second pass: now make the annotated video
    analyze_cover_drive(cfg, downswing_start=ds, impact_start=im, write_video=True)

    # Generate all the charts and reports
    chart = os.path.join(cfg["output_dir"], cfg["chart_file_name"])
    generate_smoothness_chart(M["elbow_angles"], chart)
    evaluation = generate_evaluation(M, im, cfg)

    with open(os.path.join(cfg["output_dir"], cfg["evaluation_file_name"]), "w") as f:
        json.dump(evaluation, f, indent=2)

    # Include the bat path plot in the HTML report
    generate_html_report(evaluation, chart, os.path.join(cfg["output_dir"], cfg["report_file_name"]), batline_path=bat_plot_path)
    print("Done. Artifacts in:", cfg["output_dir"])
