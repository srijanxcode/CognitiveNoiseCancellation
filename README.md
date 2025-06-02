# 🧠 Cognitive Active Noise Cancellation

A Streamlit-based demo for **Cognitive Active Noise Cancellation (ANC)** using:

- 🎚️ LMS Adaptive Filtering (DSP-based)
- 🤖 Machine Learning Denoising
- 🔁 Hybrid Signal Fusion for optimal audio clarity

This app allows users to upload noisy audio, visualize the processing, apply different ANC techniques, and compare the results through audio playback.

---

## 🚀 Features

- 🔊 **Audio Upload** (WAV format)
- 📉 **LMS Filtering** – Classic real-time ANC
- 🤖 **ML Denoising** – Deep learning-based signal cleaning
- 🔁 **Hybrid Mode** – Weighted blend of LMS & ML outputs
- 📊 **Waveform & Spectrogram** visualizations
- 🎧 **Streamlit Audio Player** for A/B comparison

---

## 🧠 Techniques Explained

### LMS Filtering (DSP)
A classical adaptive filtering method that minimizes the error between noisy input and a noise reference using the Least Mean Squares algorithm.

### ML-Based Denoising
A pretrained neural network (e.g., Denoising Autoencoder, UNet, or WaveNet) removes noise from the input audio signal.

### Hybrid ANC
Combines the outputs from LMS and ML using weighted signal fusion:


The parameter `α` can be tuned for balance between real-time filtering and deep noise reduction.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/cognitive-anc.git
cd cognitive-anc
pip install -r requirements.txt
streamlit run app.py
cognitive-anc/
├── app.py                  # Streamlit app
├── lms_filter.py           # LMS filtering logic
├── ml_denoiser.py          # ML-based denoising model
├── utils.py                # Audio utilities
├── example_audio/          # Sample noisy audio files
├── requirements.txt
└── README.md
