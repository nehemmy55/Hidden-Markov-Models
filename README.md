<div align="center">

# 🏃 Human Activity Recognition with a Hidden Markov Model

**Standing · Walking · Jumping · Still**, decoded from phone motion sensors

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/HMM-from%20scratch%20(NumPy)-013243?logo=numpy&logoColor=white)
![Accuracy](https://img.shields.io/badge/unseen%20accuracy-100%25%20(33%20windows)-2ea44f)
![Tests](https://img.shields.io/badge/tests-12%20passing-2ea44f)

</div>

## 📌 Overview
A phone records movement all the time, but the activity behind the signal (walking or jumping?) is hidden in noisy sensor data. This project uses a Hidden Markov Model to recover that hidden activity from a phone's accelerometer and gyroscope.

The use case is **exercise and rehabilitation monitoring**, where telling walking from jumping matters.

## 🔬 What I did
| Step | Details |
|---|---|
| 📱 **Collect** | Sensor Logger on an iPhone 11: 53 recordings of 5 to 8 s, about 100 Hz, resampled to 50 Hz |
| 🧹 **Clean** | Merge sensors, remove bad rows, trim start and end, 2 s windows with 50% overlap |
| 📊 **Features** | 19 time-domain and FFT features per window, z-scored with the training data only |
| 🧠 **Model** | HMM from scratch in NumPy: Baum-Welch for training, Viterbi for decoding |
| ✅ **Test** | Unseen recordings: sensitivity, specificity, precision, F1, accuracy, confusion matrix |

## 🎯 Result
| Activity | Windows | Sensitivity | Specificity |
|---|---:|---:|---:|
| Standing | 9 | 1.00 | 1.00 |
| Walking | 8 | 1.00 | 1.00 |
| Jumping | 8 | 1.00 | 1.00 |
| Still | 8 | 1.00 | 1.00 |

This is a small test from one person, one phone and one day, so it needs more data before it can be trusted widely.

<p align="center">
  <img src="figures/transition_heatmap.png" width="48%" alt="Transition matrix heatmap">
  <img src="figures/confusion_matrix.png" width="30%" alt="Confusion matrix on unseen data">
</p>
<p align="center">
  <img src="figures/decoded_sequences.png" width="78%" alt="Decoded activity sequence against the true one">
</p>

## 📁 Repository
```
├── hmm_activity_recognition_final.ipynb   full code, plots and results
├── data/                                  train/ and test/ CSV recordings
├── raw/  +  prepare_dataset.py            original exports and the script that builds data/
├── figures/  tests/  requirements.txt
└── Report.pdf                             the 5-page report
```

## ▶️ Run it
```
pip install -r requirements.txt
jupyter notebook hmm_activity_recognition_final.ipynb      # Restart and Run All
```

## 🎥 Videos
[Standing](https://drive.google.com/file/d/1fgJnANDfjqRZ35PLkdA3MfsvnkB807pz/view?usp=drive_link) · [Walking](https://drive.google.com/file/d/1F0WzvYTYVZfdjYsy_nc4VlF0ii9tCYWN/view?usp=drive_link) · [Jumping](https://drive.google.com/file/d/1HEI5BhoGEFdu5oM_bxG0uACZN-lbGa-U/view?usp=drive_link) · [Still](https://drive.google.com/file/d/13hjMlRoVJ4XSaRHSXrhMx0OnRFVN-Vqj/view?usp=drive_link)
