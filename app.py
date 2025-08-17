import streamlit as st, os, time

import numpy as np

from cover_drive_analysis_realtime import (
    load_config, analyze_cover_drive, find_swing_phases_final,
    generate_smoothness_chart, generate_evaluation, generate_html_report,
    estimate_bat_line_metrics, benchmark_metrics, plot_bat_line
)

st.set_page_config(page_title="Cricket Shot Analyzer", layout="wide")
st.title(" AI-Powered Cricket Cover Drive Analysis")

# Load our config and set up the output directory
config = load_config()
out_dir = config["output_dir"]
os.makedirs(out_dir, exist_ok=True)

# Let users upload their cover drive video
uploaded = st.file_uploader("Upload a video of your cover drive", type=["mp4","mov","avi"])
if uploaded:
    # Save the uploaded video to our output directory
    in_path = os.path.join(out_dir, "uploaded_video.mp4")
    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success("Video uploaded. Ready to analyze!")

    # Optional: users can also upload reference data to compare against
    ref_file = st.file_uploader("Upload reference CSV (optional)", type=["csv"])
    if ref_file:
        import pandas as pd
        ref_df = pd.read_csv(ref_file)
        # save to temp path and set config["reference_stats_path"] to it
        ref_path = os.path.join(out_dir, "user_ref_stats.csv")
        ref_df.to_csv(ref_path, index=False)
        config["reference_stats_path"] = ref_path



    if st.button("Analyze My Shot!", use_container_width=True):
        config["input_video_path"] = in_path

        with st.spinner("Analyzing…"):
            t0 = time.time()
            # First pass: just get the metrics without making a video
            metrics = analyze_cover_drive(config, write_video=False)
            ds, im = find_swing_phases_final(metrics)
            # Second pass: now make the annotated video
            analyze_cover_drive(config, downswing_start=ds, impact_start=im, write_video=True)

            # Generate all the reports and charts
            chart_path = os.path.join(out_dir, config["chart_file_name"])
            generate_smoothness_chart(metrics["elbow_angles"], chart_path)
            evaluation = generate_evaluation(metrics, im, config)


            # Analyze the bat path quality
            bat_metrics = estimate_bat_line_metrics(metrics, ds, im, config)

              # Generate the batline plot
            pts = []
            n = len(metrics.get("wrist_px", []))
            after = config.get("batline", {}).get("use_frames_after_impact", 3)
            start = max(0, ds)
            end = min(n - 1, im + after)
            for i in range(start, end + 1):
                w = metrics["wrist_px"][i]
                try:
                    if w is None: continue
                    if np.isnan(w[0]) or np.isnan(w[1]): continue
                    pts.append((float(w[0]), float(w[1])))
                except Exception:
                    continue

            bat_plot_path = os.path.join(out_dir, "batline_plot.png")
            plot_bat_line(pts, bat_metrics.get("line_params"), bat_plot_path)

            report_path = os.path.join(out_dir, config["report_file_name"])
            with open(os.path.join(out_dir, config["evaluation_file_name"]), "w") as f:
                import json; json.dump(evaluation, f, indent=2)
            generate_html_report(evaluation, chart_path, report_path, batline_path=bat_plot_path)

        st.success(f" Analysis complete in {time.time()-t0:.2f}s")

        # Show the annotated video with all the overlays
       

        st.subheader("Annotated Video")
        video_path = os.path.join(out_dir, config["output_video_name"])
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.warning("Annotated video not found")

        st.subheader("Scores & Feedback")
        cols = ["Footwork","Head Position","Swing Control","Balance","Follow-through"]
        for c in cols:
            det = evaluation.get(c, {"score":"-","feedback":"No feedback"})
            score_text = f"{det.get('score','-')}/10" if det.get("score") is not None else "-"
            st.metric(c, score_text)
            st.write(det.get("feedback","-"))

        st.metric("Overall Skill Grade", evaluation.get("skill_grade", "N/A"))

        # safe metrics summary values
        metrics_summary = {
            "elbow_at_impact": float(np.nan if len(metrics.get("elbow_angles", [])) <= im else metrics["elbow_angles"][im]),
            "foot_angle": float(np.nan if len(metrics.get("foot_angles", [])) <= im else metrics["foot_angles"][im]),
            "head_alignment": float(np.nan if len(metrics.get("head_knee_alignments", [])) <= im else metrics["head_knee_alignments"][im]),
            "batline_rms": bat_metrics.get("rms_dev") if bat_metrics else None,
            "wrist_speed": float(np.nan if len(metrics.get("wrist_speeds", [])) <= im else metrics["wrist_speeds"][im])
        }

        st.subheader("Bat-line Analysis")
        bat_rms = metrics_summary.get("batline_rms")
        st.metric("Bat-line RMS (px)", f"{bat_rms:.1f}" if bat_rms is not None else "N/A")
        if os.path.exists(bat_plot_path):
            st.image(bat_plot_path)
        else:
            st.warning("Bat-line plot not generated")

        st.subheader("Benchmarking")
        ref_df_obj = None
        if 'ref_path' in locals() and os.path.exists(ref_path):
            import pandas as _pd
            ref_df_obj = _pd.read_csv(ref_path)
        benchmark = benchmark_metrics(metrics_summary, config, ref_df_obj)
        # show benchmark as JSON-like object or table
        try:
            st.json(benchmark)
        except Exception:
            st.write(benchmark)

        # Smoothness handling - fallback compute if evaluation missing it
        def compute_smoothness_from_angles(angles, fps=30.0):
            y = np.array([v for v in angles if not (v is None or np.isnan(v))], dtype=float)
            if len(y) < 3:
                return {"std_dev": float("nan"), "rms_jerk": float("nan"), "smoothness_score": 0.0}
            sd = float(np.std(y))
            vel = np.gradient(y) * fps
            acc = np.gradient(vel) * fps
            jerk = np.gradient(acc) * fps
            rms_jerk = float(np.sqrt(np.mean(jerk**2)))
            smoothness_score = float(min(10.0, 10.0 / (1.0 + rms_jerk)))
            return {"std_dev": sd, "rms_jerk": rms_jerk, "smoothness_score": smoothness_score}

        smooth = evaluation.get("smoothness")
        if not smooth:
            smooth = compute_smoothness_from_angles(metrics.get("elbow_angles", []), metrics.get("fps", 30.0))

        st.subheader("Smoothness Evaluation")
        st.write(f"Elbow Angle Std Dev: {smooth.get('std_dev', float('nan')):.2f} deg")
        st.write(f"RMS Jerk: {smooth.get('rms_jerk', float('nan')):.2f} deg/s³")
        st.metric("Smoothness Score (0–10)", f"{smooth.get('smoothness_score', 0.0):.1f}")
        if os.path.exists(chart_path):
            st.image(chart_path)
        else:
            st.warning("Smoothness chart not generated")

        # Download buttons
        st.subheader("Download Your Results")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(video_path):
                with open(video_path, "rb") as f:
                    st.download_button("Download Annotated Video", f, file_name="annotated_shot.mp4")
            else:
                st.error("Annotated video not found")
        with c2:
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    st.download_button("Download HTML Report", f, file_name="shot_report.html")
            else:
                st.error("HTML report not found")