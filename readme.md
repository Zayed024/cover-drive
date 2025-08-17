

# AI-Powered Cricket Cover Drive Analysis 🏏

This project implements a comprehensive real-time analysis system for cricket cover drive biomechanics, developed as part of the AthleteRise AI-Powered Cricket Analytics assignment. The system processes full videos frame-by-frame to provide live feedback and comprehensive performance evaluation.

## 🎯 Project Overview

This system analyzes cricket cover drive videos using computer vision and pose estimation to extract biomechanical metrics in real-time. It provides instant feedback during video playback and generates comprehensive performance reports with skill grading.

## ✨ Key Features

### Real-Time Video Processing
- **Full Video Analysis**: Processes complete videos frame-by-frame for comprehensive coverage
- **Live Pose Estimation**: Uses MediaPipe to extract 33 key body landmarks per frame
- **Real-Time Overlays**: Displays pose skeleton, metrics, and feedback during video playback

### Biomechanical Analysis
- **Front Elbow Angle**: Tracks elbow elevation throughout the swing
- **Spine Lean**: Measures lateral body positioning
- **Head-Knee Alignment**: Monitors head position relative to front knee
- **Front Foot Direction**: Analyzes foot positioning and angle
- **Bat-line Analysis**: Estimates bat path and deviation metrics
- **Wrist Speed**: Calculates hand movement velocity

### Intelligent Phase Detection
- **Automatic Swing Phases**: Identifies Stance, Downswing, and Follow-through phases
- **Foot Plant Anchoring**: Uses front foot plant as reference point for phase detection
- **Smoothness Evaluation**: Analyzes movement fluidity throughout the swing

### Performance Evaluation
- **Multi-Category Scoring**: Evaluates Footwork, Head Position, Swing Control, Balance, and Follow-through
- **Reference Benchmarking**: Compares performance against ideal ranges and reference statistics
- **Skill Grading**: Assigns Beginner, Intermediate, or Advanced skill levels
- **Comprehensive Feedback**: Provides specific improvement suggestions for each category

## 🛠️ Technical Implementation

### Core Technologies
- **MediaPipe**: Real-time pose estimation with 33 landmark detection
- **OpenCV**: Video processing and frame manipulation
- **NumPy**: Mathematical computations and data processing
- **Matplotlib**: Data visualization and chart generation
- **Streamlit**: Web-based user interface

### Architecture
- **Two-Pass Analysis**: First pass extracts metrics, second pass generates annotated video
- **Modular Design**: Separate functions for pose detection, metric calculation, and evaluation
- **Configurable Parameters**: JSON-based configuration for thresholds and ideal ranges
- **Output Generation**: Multiple formats including video, charts, reports, and data files

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- MediaPipe
- OpenCV
- Streamlit
- Other dependencies listed in `requirements.txt`

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cover-drive
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎬 Usage

### Web Application (Recommended)
1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Upload your cricket video** (MP4, MOV, or AVI format)

3. **Optionally upload reference statistics** (CSV format) for benchmarking

4. **Click "Analyze My Shot!"** to run the analysis

5. **View results** including annotated video, scores, and downloadable reports

### Command Line Interface
1. **Configure your video** in `config.json`
2. **Run the analysis:**
   ```bash
   python cover_drive_analysis_realtime.py
   ```

## 📊 Output Files

The system generates comprehensive outputs in the `/output/` directory:

- **`annotated_video_advanced.mp4`**: Video with real-time overlays and feedback
- **`evaluation_report.json`**: Detailed scores and feedback for each category
- **`smoothness_chart.png`**: Elbow angle visualization throughout the swing
- **`batline_plot.png`**: Bat path analysis and deviation metrics
- **`final_report.html`**: Comprehensive HTML report with all results
- **`uploaded_video.mp4`**: Original uploaded video (if using web app)

## ⚙️ Configuration

The `config.json` file allows customization of:

- **Video Settings**: Input/output paths and video processing parameters
- **Pose Detection**: Confidence thresholds and model complexity
- **Evaluation Thresholds**: Ideal ranges for biomechanical metrics
- **Phase Detection**: Timing parameters for swing phase identification
- **Reference Statistics**: Paths for benchmarking data

## 🔍 Analysis Categories

### Footwork (Score: 0-10)
- Evaluates front foot positioning and angle
- Analyzes foot plant timing and stability

### Head Position (Score: 0-10)
- Measures head alignment over front knee
- Tracks head stability throughout the swing

### Swing Control (Score: 0-10)
- Analyzes elbow elevation and positioning
- Evaluates swing path consistency

### Balance (Score: 0-10)
- Measures lateral body lean
- Assesses overall body stability

### Follow-through (Score: 0-10)
- Evaluates extension and finish position
- Analyzes swing completion quality

### Overall Skill Grade
- **Beginner**: 0-3 average score
- **Intermediate**: 4-7 average score  
- **Advanced**: 8-10 average score

## 📈 Advanced Features

### Bat-line Analysis
- Estimates ideal bat path using least squares fitting
- Calculates RMS deviation from ideal path
- Provides visual representation of bat trajectory

### Smoothness Evaluation
- Analyzes movement fluidity using derivative calculations
- Generates smoothness scores based on movement consistency
- Creates time-series charts for detailed analysis

### Reference Benchmarking
- Compares performance against ideal template values
- Supports custom reference statistics via CSV upload
- Provides percentile rankings and improvement targets

## 🎥 Video Requirements

### Optimal Conditions
- **Camera Angle**: Side-on view of the batter
- **Lighting**: Good illumination for clear pose detection
- **Resolution**: Minimum 720p recommended
- **Duration**: Full swing from stance to follow-through

### Supported Formats
- MP4 (recommended)
- MOV
- AVI

## 🔧 Technical Limitations

### Current Constraints
- **2D Analysis**: Uses 2D coordinates (depth not considered)
- **Right-Handed Focus**: Phase detection optimized for right-handed batters
- **Motion Blur**: High-speed movements may cause detection noise
- **Camera View**: Performance degrades with non-side-on angles

### Performance Considerations
- **Processing Speed**: Real-time analysis on modern hardware
- **Memory Usage**: Efficient frame-by-frame processing
- **GPU Support**: Optional GPU acceleration for pose detection

## 🤝 Contributing

This project is developed as part of the AthleteRise assignment. Contributions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is developed for educational and research purposes as part of the AthleteRise AI-Powered Cricket Analytics assignment.

## 🙏 Acknowledgments

- **MediaPipe**: For real-time pose estimation capabilities
- **AthleteRise**: For the cricket analytics assignment and requirements
- **Open Source Community**: For the various libraries and tools used

---

*Built with ❤️ for cricket analytics and biomechanical research*