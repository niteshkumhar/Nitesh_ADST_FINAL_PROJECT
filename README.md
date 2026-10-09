# 🔍 PCB-Sentry AOI: Automated Optical Inspection System

An AI-powered Automated Optical Inspection (AOI) web platform designed for SMT (Surface Mount Technology) electronics manufacturing lines. It leverages computer vision and deep learning models trained on the benchmark **DeepPCB dataset** to detect sub-millimeter printed circuit board defects in real time.

---

## 🚀 Key Features

* **Automated Defect Detection:** Identifies critical manufacturing anomalies including **Shorts, Spurs, Open Circuits, Mousebites, and Pinholes**.
* **Template Alignment & Anomaly Isolation:** Uses advanced image processing (OpenCV & NumPy) to align test boards against golden reference templates and isolate defects with precision bounding boxes.
* **Industrial Glassmorphic UI:** Built with Flask and Tailwind CSS featuring a high-contrast dark theme optimized for factory floor monitoring and operator consoles.
* **Centralized Telemetry & Audit Logs:** Managed via SQLite and SQLAlchemy to log shift history, inspection pass/fail statuses, and operator activity records securely.

---

## 🛠️ Technology Stack

* **Backend:** Python, Flask, SQLAlchemy
* **Computer Vision / AI:** OpenCV, NumPy, YOLOv8
* **Database:** SQLite
* **Frontend:** HTML5, Tailwind CSS, Google Fonts (*Chakra Petch*)
* **Version Control:** Git, GitHub

---

## 📁 Project Directory Structure

```text
Auto-Defect System in PCB/
│
├── instance/               # Local SQLite database (ignored in git)
├── model/                  # Model definition and weights configuration
├── runs/detect/             # YOLO inference and trained model outputs
├── static/                 # CSS, JavaScript, logo images, and UI assets
├── templates/              # HTML templates (index, login, signup, dashboard)
├── app.py                  # Main Flask application server & routing
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```
---

## ⚙️ Installation & Local Setup
Follow these steps to set up and run the project locally on your machine:

*1. Clone the Repository
   git clone [https://github.com/niteshkumhar/Nitesh_ADST_FINAL_PROJECT.git](https://github.com/niteshkumhar/Nitesh_ADST_FINAL_PROJECT.git))

*2. Create and Activate a Virtual Environment
    python -m venv pcb_env
      # On Windows:
            pcb_env\Scripts\activate
      # On macOS/Linux:
             source pcb_env/bin/activate   

*3. Install Dependencies
   pip install -r requirements.txt

*4. Run the Flask Application
   python app.py

---

## 💡 Usage Workflow

*1. Operator Onboarding / Sign-In: Access the secure glassmorphic authentication console (/login or /signup).

*2. Live Feed & Inspection Dashboard: Monitor production line metrics, active SMT nodes, and real-time inference streams.

*3. Audit History & Reporting: Review historical batch inspection logs, bounding-box outputs, and compliance reports stored in the database.

---

## 📄 License
This project is developed for academic presentation and industrial research purposes.

Developed for advanced SMT quality control and automated manufacturing inspection.  
    
