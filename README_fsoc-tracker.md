# FSOC Tracker — Coarse Alignment & Beacon Tracking Simulator

A Python-based simulation and visualization framework for **Free-Space Optical Communication (FSOC) coarse alignment and beacon tracking**.

The project provides a desktop GUI for experimenting with a moving optical target/beacon, camera-field-of-view tracking, tracking disturbances, and different detection/tracking conditions. The repository also contains a YOLO-based beacon detector model and separate modules for control, disturbance generation, logging, simulation, UI, and vision.

> **Project repository:** https://github.com/riyyaa28/fsoc-tracker

---

## 🚀 Project Overview

Free-Space Optical Communication uses a narrow optical beam to establish a high-bandwidth communication link between two terminals. Because the beam is highly directional, the receiving terminal must first be brought into the transmitter's field of view.

This project focuses on the **coarse alignment / acquisition stage**:

**Beacon / Target → Vision Detection → Position Estimation → Tracking → Coarse Alignment**

The simulator makes it possible to test the tracking behavior under different motion patterns and environmental disturbances before moving toward a physical or higher-fidelity FSOC test setup.

---

## ✨ Main Features

- **Coarse alignment visualization**
- **Beacon/target tracking**
- **YOLO-based detection support**
- Classical/coast-style tracking visualization
- Multiple target motion patterns:
  - Straight
  - Circular
  - Random
- Adjustable disturbance conditions:
  - Fog
  - Image noise
  - Jitter
  - Rain
- Distance variation during simulation
- Camera / field-of-view representation
- Real-time score and tracking status display
- PyQt-based desktop interface
- Separate modules for:
  - Control
  - Disturbance generation
  - Logging
  - Simulation
  - UI
  - Vision

---

## 🧠 System Concept

```text
                 ┌─────────────────────┐
                 │   Remote Beacon /    │
                 │       Target        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Camera / Sensor    │
                 │       Input         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Vision / Detection  │
                 │  YOLO or Classical  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Position / Error    │
                 │     Estimation      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Tracking / Control  │
                 │   Coarse Alignment  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ FSOC Terminal       │
                 │ stays aligned with  │
                 │ the remote beacon   │
                 └─────────────────────┘
```

The green rectangle in the simulator represents the tracking / camera region, while the target and tracking indicators show the estimated beacon position and alignment behavior.

---

## 🖥️ Simulator Interface

The GUI displays:

- Current distance between the simulated terminals
- Target motion pattern
- Detection/tracking source
- Tracking score
- Main simulated camera view
- Smaller tracking/zoom view
- Field-of-view / tracking region
- Disturbance controls

Example status information shown by the simulator:

```text
Source: yolo
Score: 1.00
Dist: 20.0m
Pattern: straight
```

---

# 📸 Simulation Outputs

The following screenshots are included in this repository under `docs/outputs/`.

## 1. Classical Tracking — Circular Motion

A circular target trajectory with the classical/coast tracking source.

![Classical circular tracking](docs/outputs/01_classical_circular.png)

**Observed configuration**
- Distance: approximately 20 m
- Pattern: circular
- Source: classical/coast
- Tracking score: approximately 0.96

---

## 2. YOLO Detection — Straight Motion

YOLO-based beacon detection during straight target motion.

![YOLO straight tracking](docs/outputs/02_yolo_straight.png)

**Observed configuration**
- Distance: approximately 20 m
- Pattern: straight
- Source: YOLO
- Tracking score: 1.00

---

## 3. Random Motion — Fog and Noise

Random target motion with simulated visual degradation.

![Random motion with fog and noise](docs/outputs/03_random_fog_noise.png)

**Observed configuration**
- Distance: approximately 20 m
- Pattern: random
- Fog: level 3
- Noise: level 2
- Jitter: off
- Rain: off

---

## 4. Random Motion — Fog, Noise and Jitter

A more challenging tracking condition with target motion and additional camera/pointing jitter.

![Random motion with fog noise and jitter](docs/outputs/04_random_fog_noise_jitter.png)

**Observed configuration**
- Distance: approximately 20 m
- Pattern: random
- Fog: level 3
- Noise: level 2
- Jitter: level 3
- Rain: off

---

## 5. Straight Motion — Combined Disturbances

A straight-motion scenario with fog, noise, jitter and rain enabled.

![Straight motion with combined disturbances](docs/outputs/05_straight_fog_noise_jitter_rain.png)

**Observed configuration**
- Distance: approximately 13.3 m
- Pattern: straight
- Fog: level 3
- Noise: level 2
- Jitter: level 3
- Rain: level 3

---

## 🗂️ Repository Structure

```text
fsoc-tracker/
│
├── control/             # Tracking / control logic
├── disturbance/        # Environmental and sensor disturbance models
├── logging/             # Logging utilities
├── logging_/            # Additional logging components
├── sim/                 # Simulation components
├── ui/                  # PyQt GUI / dashboard
├── vision/              # Vision and detection components
│
├── beacon_yolo.pt       # Trained beacon detection model
├── yolo12n.pt           # YOLO model
├── main.py              # Application entry point
├── train_yolo.py        # YOLO training script
├── requirements.txt     # Python dependencies
│
└── docs/
    └── outputs/         # Simulation screenshots
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/riyyaa28/fsoc-tracker.git
cd fsoc-tracker
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Simulator

From the repository root:

```bash
python main.py
```

The application launches the **FSOC Coarse Alignment Simulator** desktop interface.

---

## 🤖 YOLO Detection

The repository includes YOLO-related model files and a training script.

The YOLO pipeline can be used to detect the optical beacon/target in the simulated camera view.

Conceptually:

```text
Camera Frame
     │
     ▼
 YOLO Detector
     │
     ▼
Beacon Bounding Box
     │
     ▼
Beacon Center
     │
     ▼
Tracking Error
     │
     ▼
Coarse Alignment
```

This allows the vision component to provide a target location to the tracking/control layer.

---

## 🌫️ Disturbance Simulation

The simulator includes configurable disturbance conditions to test tracking robustness.

### Fog
Reduces scene visibility and contrast.

### Noise
Introduces image-level noise that can make beacon detection more difficult.

### Jitter
Represents small pointing/camera motion or instability.

### Rain
Introduces additional visual degradation and interference.

These controls allow the same tracking algorithm to be tested under progressively harder conditions.

---

## 🎯 Why Coarse Alignment Matters in FSOC

FSOC terminals generally require accurate pointing because the optical beam is narrow.

The acquisition process can be viewed as:

```text
                 COARSE ALIGNMENT
                       │
                       ▼
        ┌──────────────────────────┐
        │ Find / detect remote     │
        │ beacon in camera FOV     │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │ Reduce pointing error    │
        └────────────┬─────────────┘
                     │
                     ▼
                 FINE POINTING
                     │
                     ▼
               FSOC LINK
```

The current simulator primarily demonstrates the **detection and coarse tracking stage**.

---

## 📊 Example Test Matrix

| Scenario | Motion | Fog | Noise | Jitter | Rain |
|---|---|---:|---:|---:|---:|
| Baseline | Circular | Off | Off | Off | Off |
| YOLO test | Straight | Off | Off | Off | Off |
| Degraded 1 | Random | 3 | 2 | Off | Off |
| Degraded 2 | Random | 3 | 2 | 3 | Off |
| Degraded 3 | Straight | 3 | 2 | 3 | 3 |

---

## 🔬 Future Development

Possible extensions include:

- Predictive tracking for moving UAV/FSOC terminals
- Optical-flow-assisted beacon tracking
- Adaptive beacon detection under changing illumination
- Kalman or other state-estimation filters
- 3D UAV-to-UAV simulation
- Pointing and acquisition control using azimuth/elevation errors
- Line-of-sight (LOS) obstruction modeling
- Atmospheric attenuation and turbulence models
- FSOC link-budget calculation
- Received optical power estimation
- Link establishment / loss visualization
- Hardware-in-the-loop testing
- More realistic camera and sensor models

---

## 🛠️ Technology Stack

- **Python**
- **PyQt5** — desktop GUI
- **OpenCV** — computer vision / image processing
- **PyTorch** — deep learning
- **Ultralytics YOLO** — object detection
- **NumPy** — numerical computation
- **Matplotlib / PyQtGraph** — visualization
- **NetworkX** — graph-related utilities where applicable

The repository's `requirements.txt` contains the project's pinned Python dependencies.

---

## 📌 Project Status

This repository contains a working simulation-oriented prototype for demonstrating **FSOC beacon detection and coarse alignment/tracking**.

The included screenshots demonstrate operation under different motion and disturbance configurations. The system is intended as a software-based environment for developing and testing tracking concepts before integration with more detailed 3D, optical, or hardware environments.

---

## 👥 Contribution

Contributions are welcome.

A typical contribution workflow:

```bash
git checkout -b feature/my-feature
git add .
git commit -m "Add my feature"
git push origin feature/my-feature
```

Then open a Pull Request on GitHub.

---

## 📄 License

No license is currently specified in the repository. If this project is intended for public reuse, add an appropriate open-source license (for example, MIT) to the repository.

---

## 🔗 Repository

**GitHub:** https://github.com/riyyaa28/fsoc-tracker
