import streamlit as st, os, time

import numpy as np

from cover_drive_analysis_realtime import (
    load_config, analyze_cover_drive, find_swing_phases_final,
    generate_smoothness_chart, generate_evaluation, generate_html_report,
    estimate_bat_line_metrics, benchmark_metrics
)

st.set_page_config(page_title="Cricket Shot Analyzer", layout="wide")
st.title("🏏 AI-Powered Cricket Cover Drive Analysis")

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

            report_path = os.path.join(out_dir, config["report_file_name"])
            with open(os.path.join(out_dir, config["evaluation_file_name"]), "w") as f:
                import json; json.dump(evaluation, f, indent=2)
            generate_html_report(evaluation, chart_path, report_path)

        st.success(f"✅ Analysis complete in {time.time()-t0:.2f}s")

        # Show the annotated video with all the overlays
        st.subheader("Annotated Video")
        st.video(os.path.join(out_dir, config["output_video_name"]))

        # Display the scores and feedback for each technique area
        st.subheader("Scores & Feedback")
        cols = ["Footwork","Head Position","Swing Control","Balance","Follow-through"]
        for c in cols:
            det = evaluation[c]
            st.metric(c, f"{det['score']}/10")
            st.write(det["feedback"])
        st.metric("Overall Skill Grade", evaluation["skill_grade"])

        # Analyze the bat path quality
        bat_metrics = estimate_bat_line_metrics(metrics, ds, im, config)
        metrics_summary = {
    "elbow_at_impact": float(np.nan if len(metrics.get("elbow_angles", [])) <= im else metrics["elbow_angles"][im]),
    "foot_angle": float(np.nan if len(metrics.get("foot_angles", [])) <= im else metrics["foot_angles"][im]),
    "head_alignment": float(np.nan if len(metrics.get("head_knee_alignments", [])) <= im else metrics["head_knee_alignments"][im]),
    "batline_rms": bat_metrics.get("rms_dev"),
    "wrist_speed": float(np.nan if len(metrics.get("wrist_speeds", [])) <= im else metrics["wrist_speeds"][im])
}
        
        # Load reference data if the user provided it
        ref_df_obj = None
        if 'ref_path' in locals() and os.path.exists(ref_path):
            import pandas as _pd
            ref_df_obj = _pd.read_csv(ref_path)
        benchmark = benchmark_metrics(metrics_summary, config, ref_df_obj)

        # Show how straight the bat path was
        st.subheader("Bat-line Analysis")
        st.metric("Bat-line RMS (px)", f"{metrics_summary['batline_rms']:.1f}")
        st.image(os.path.join(out_dir, "batline_plot.png"))
        st.subheader("Benchmarking")
        st.write(benchmark)  # pretty print or show as dataframe


        # Show how smooth the swing was
        st.subheader("Smoothness Evaluation")
        st.metric("Smoothness Score (0–10)", f"{evaluation['smoothness']['smoothness_score']:.1f}")
        st.image(os.path.join(out_dir, "smoothness_chart.png"))

        # Let users download their results
        st.subheader("Download Your Results")
        c1, c2 = st.columns(2)
        with c1:
            with open(os.path.join(out_dir, config["output_video_name"]), "rb") as f:
                st.download_button("Download Annotated Video", f, file_name="annotated_shot.mp4")
        with c2:
            with open(report_path, "rb") as f:
                st.download_button("Download HTML Report", f, file_name="shot_report.html")
