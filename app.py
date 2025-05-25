import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy import signal
from io import BytesIO
import base64

# Set page config with new title and wide layout
st.set_page_config(page_title="Cognitive Active Noise Cancellation", layout="wide")

# Enhanced Custom CSS for modern aesthetic styling
st.markdown("""
<style>
    body, .main {
        background: linear-gradient(135deg, #2c3e50, #4ca1af);
        color: #f0f0f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #e74c3c;
        color: white;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #c0392b;
        cursor: pointer;
    }
    .stSelectbox, .stSlider {
        background: #283e4a;
        border-radius: 10px;
        padding: 10px 15px;
        color: #e0e0e0;
    }
    .plot-container {
        background: #34495e;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    .header {
        color: #ecf0f1;
        text-align: center;
        margin-bottom: 40px;
        font-size: 2.8rem;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }
    .metric-box {
        background-color: #3a4a5a;
        border-radius: 15px;
        padding: 20px 25px;
        margin: 15px 0;
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
        font-weight: 700;
        font-size: 1.2rem;
        color: #fff;
        text-align: center;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Page header title
st.markdown('<div class="header">🧠 Cognitive Active Noise Cancellation Demo</div>', unsafe_allow_html=True)

st.markdown("""
### About this demo:
Explore a next-generation noise cancellation system combining:
- Adaptive LMS Filtering for dynamic noises like car honks
- Machine Learning for persistent noises like traffic hum
- Smart hybrid dynamic blend maximizing noise suppression and speech clarity
- Additional signal visualizations for deeper insights into noise cancellation mechanics
""")

def generate_speech(fs=8000, duration=5):
    t = np.arange(0, duration, 1/fs)
    hello_dur = 0.8
    how_dur = 0.7
    are_dur = 0.6
    you_dur = 1.0
    silence_dur = 0.9
    h_noise = np.random.normal(0, 0.1, int(0.1*fs))
    e_vowel = (0.7*np.sin(2*np.pi*120*t[:int(0.3*fs)])*(0.5+0.5*np.sin(2*np.pi*2*t[:int(0.3*fs)])) + 
               0.3*np.sin(2*np.pi*500*t[:int(0.3*fs)]) + 0.2*np.sin(2*np.pi*1800*t[:int(0.3*fs)]))
    ow_vowel = (0.8*np.sin(2*np.pi*120*t[:int(0.4*fs)])*(0.5+0.5*np.sin(2*np.pi*2*t[:int(0.4*fs)])) +
                0.4*np.sin(2*np.pi*400*t[:int(0.4*fs)]) + 0.3*np.sin(2*np.pi*800*t[:int(0.4*fs)]))
    aw_vowel = (0.8*np.sin(2*np.pi*110*t[:int(0.4*fs)])*(0.5+0.5*np.sin(2*np.pi*2*t[:int(0.4*fs)])) +
                0.5*np.sin(2*np.pi*600*t[:int(0.4*fs)]) + 0.3*np.sin(2*np.pi*1200*t[:int(0.4*fs)]))
    aa_vowel = (0.8*np.sin(2*np.pi*100*t[:int(0.4*fs)])*(0.5+0.5*np.sin(2*np.pi*2*t[:int(0.4*fs)])) +
                0.5*np.sin(2*np.pi*700*t[:int(0.4*fs)]) + 0.3*np.sin(2*np.pi*1100*t[:int(0.4*fs)]))
    yoo_vowel = (0.8*np.sin(2*np.pi*100*t[:int(0.6*fs)])*(0.5+0.5*np.sin(2*np.pi*2*t[:int(0.6*fs)])) +
                 0.3*np.sin(2*np.pi*300*t[:int(0.6*fs)]) + 0.2*np.sin(2*np.pi*2300*t[:int(0.6*fs)]))
    speech = np.zeros_like(t)
    start = 0
    speech[start:start+len(h_noise)] = h_noise
    start += len(h_noise)
    speech[start:start+len(e_vowel)] = e_vowel
    start += len(e_vowel)
    speech[start:start+len(ow_vowel)] = ow_vowel
    start += len(ow_vowel)
    start += int(silence_dur * fs)
    speech[start:start+len(h_noise)] = h_noise
    start += len(h_noise)
    speech[start:start+len(aw_vowel)] = aw_vowel
    start += len(aw_vowel)
    start += int(silence_dur * fs)
    speech[start:start+len(aa_vowel)] = aa_vowel
    start += len(aa_vowel)
    start += int(silence_dur * fs)
    speech[start:start+len(yoo_vowel)] = yoo_vowel
    speech = speech / np.max(np.abs(speech)) * 0.8
    return speech

def generate_noise(fs=8000, duration=5):
    t = np.arange(0, duration, 1/fs)
    traffic = np.random.normal(0, 0.2, len(t))
    for freq in [200, 400, 800, 1200]:
        traffic += 0.1 * np.sin(2*np.pi*freq*t)
    honks = np.zeros(len(t))
    honk_duration = 0.3
    honk_samples = int(honk_duration * fs)
    for honk_time in [1.2, 2.5, 3.8]:
        honk_start = int(honk_time * fs)
        honk_freq = 800 + 200*np.random.randn()
        honk = np.sin(2*np.pi*honk_freq*t[:honk_samples]) * np.hanning(honk_samples)
        honks[honk_start:honk_start+honk_samples] += honk
    noise = traffic + honks
    noise = noise / np.max(np.abs(noise)) * 0.8
    reference = 0.85*noise + 0.15*np.random.normal(0, 0.1, len(t))
    return noise, reference

def lms_filter(noisy_input, reference, mu=0.01, order=64):
    n = len(noisy_input)
    w = np.zeros(order)
    output = np.zeros(n)
    for i in range(order, n):
        x_vec = reference[i:i-order:-1]
        y = np.dot(w, x_vec)
        e = noisy_input[i] - y
        w = w + mu * e * x_vec
        output[i] = e
    return output

def ml_processing(noisy_input, speech, frame_len=256, hop_size=64, lambda_reg=0.1):
    n = len(noisy_input)
    num_frames = (n - frame_len) // hop_size + 1
    X = np.zeros((frame_len, num_frames))
    Y = np.zeros((frame_len, num_frames))
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_len
        X[:, i] = noisy_input[start:end]
        Y[:, i] = speech[start:end]
    X_design = np.vstack([X, np.ones(num_frames)])
    beta = Y @ X_design.T @ np.linalg.inv(X_design @ X_design.T + lambda_reg * np.eye(frame_len + 1))
    output = np.zeros(n)
    window = np.hanning(frame_len)
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_len
        x_vec = np.append(noisy_input[start:end], 1)
        y_pred = beta @ x_vec
        output[start:end] += y_pred * window
    output = output / np.max(np.abs(output))
    return output

def smooth_signal(x, window_size):
    half_win = window_size // 2
    y = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - half_win)
        end = min(len(x), i + half_win)
        y[i] = np.mean(x[start:end])
    return y

def calculate_snr(clean, noisy, window_size):
    snr = np.zeros(len(clean))
    for i in range(window_size, len(clean)):
        window = slice(i - window_size, i)
        signal_power = np.var(clean[window])
        noise_power = np.var(clean[window] - noisy[window])
        snr[i] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr

def hybrid_combination(lms_output, ml_output, speech, fs=8000):
    window_size = fs // 2
    snr_lms = calculate_snr(speech, lms_output, window_size)
    snr_ml = calculate_snr(speech, ml_output, window_size)
    snr_lms = smooth_signal(snr_lms, fs//4)
    snr_ml = smooth_signal(snr_ml, fs//4)
    alpha = 0.7*(snr_lms > snr_ml) + 0.3*(snr_lms <= snr_ml)
    alpha = smooth_signal(alpha, fs//2)
    hybrid = alpha * lms_output + (1 - alpha) * ml_output
    hybrid = hybrid / np.max(np.abs(hybrid))
    return hybrid, alpha, snr_lms, snr_ml

def plot_spectrogram(ax, signal, fs, title):
    n_fft = 256
    hop_length = n_fft // 4
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    img = librosa.display.specshow(D, sr=fs, hop_length=hop_length, x_axis='time', y_axis='linear', ax=ax)
    ax.set_title(title)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")

def audio_player(signal, fs):
    byte_io = BytesIO()
    sf.write(byte_io, signal, fs, format='wav')
    byte_io.seek(0)
    bytes_wav = byte_io.read()
    b64 = base64.b64encode(bytes_wav).decode()
    html = f"""
    <audio controls>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    return html

def plot_frequency_response(ax, signal_data, fs, title):
    f, Pxx_den = signal.welch(signal_data, fs, nperseg=1024)
    ax.semilogy(f, Pxx_den)
    ax.set_title(title)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power Spectral Density')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

def main():
    st.sidebar.header("Parameters")
    volume = st.sidebar.slider("Output Volume", 0.1, 1.0, 0.7)
    mu = st.sidebar.slider("LMS Step Size (μ)", 0.001, 0.1, 0.01, 0.001)
    order = st.sidebar.slider("Filter Order", 16, 128, 64, 8)
    lambda_reg = st.sidebar.slider("ML Reg. (λ)", 0.01, 1.0, 0.1, 0.01)
    fs = 8000
    duration = 5
    t = np.arange(0, duration, 1/fs)

    # Generate signals first and show their time-domain plots
    speech = generate_speech(fs, duration)
    noise, reference = generate_noise(fs, duration)
    noisy_input = speech + noise

    st.subheader("Time Domain Signals")
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(t, speech, color='#00ffff')
    axs[0].set_title("Clean Speech Signal")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(alpha=0.3)
    axs[1].plot(t, noise, color='#ff7f0e')
    axs[1].set_title("Noise Signal")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(alpha=0.3)
    axs[2].plot(t, noisy_input, color='gray')
    axs[2].set_title("Noisy Input (Speech + Noise)")
    axs[2].set_ylabel("Amplitude")
    axs[2].set_xlabel("Time (Seconds)")
    axs[2].grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # Process signals
    lms_output = lms_filter(noisy_input, reference, mu, order)
    ml_output = ml_processing(noisy_input, speech, lambda_reg=lambda_reg)
    hybrid_output, alpha, snr_lms, snr_ml = hybrid_combination(lms_output, ml_output, speech, fs)

    # Display filtered outputs
    st.subheader("Filtered Output Signals")
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(t, lms_output, color='#27ae60')
    axs[0].set_title("LMS Filtered Output")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(alpha=0.3)
    axs[1].plot(t, ml_output, color='#9b59b6')
    axs[1].set_title("ML Enhanced Output")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(alpha=0.3)
    axs[2].plot(t, hybrid_output, color='#34495e')
    axs[2].set_title("Hybrid Output")
    axs[2].set_ylabel("Amplitude")
    axs[2].set_xlabel("Time (Seconds)")
    axs[2].grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # Adaptive weights over time
    st.subheader("Adaptive Weights Over Time")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, alpha, label='LMS Weight (α)', color='#e74c3c', linewidth=3)
    ax.plot(t, 1 - alpha, label='ML Weight (1-α)', color='#9b59b6', linewidth=3)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Weight value')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # Calculate and show SNR in percentage improvement, always positive
    def safe_percentage_improvement(orig_snr, new_snr):
        if orig_snr == 0:
            return 0.0
        improvement = ((new_snr - orig_snr) / abs(orig_snr)) * 100.0
        return abs(improvement)

    original_snr = 10 * np.log10(np.var(speech) / np.var(noise))
    lms_snr_val = 10 * np.log10(np.var(speech) / np.var(speech - lms_output))
    ml_snr_val = 10 * np.log10(np.var(speech) / np.var(speech - ml_output))
    hybrid_snr_val = 10 * np.log10(np.var(speech) / np.var(speech - hybrid_output))

    lms_snr_imp = safe_percentage_improvement(original_snr, lms_snr_val)
    ml_snr_imp = safe_percentage_improvement(original_snr, ml_snr_val)
    hybrid_snr_imp = safe_percentage_improvement(original_snr, hybrid_snr_val)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original SNR (dB)", f"{original_snr:.2f}")
    col2.metric("LMS Improvement (%)", f"{lms_snr_imp:.2f}%")
    col3.metric("ML Improvement (%)", f"{ml_snr_imp:.2f}%")
    col4.metric("Hybrid Improvement (%)", f"{hybrid_snr_imp:.2f}%")

    # Spectrograms
    st.subheader("Spectrograms of Signals")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    plot_spectrogram(axes[0], noisy_input, fs, "Noisy Input Spectrogram")
    plot_spectrogram(axes[1], lms_output, fs, "LMS Filtered Spectrogram")
    plot_spectrogram(axes[2], hybrid_output, fs, "Hybrid Output Spectrogram")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Frequency Response comparison
    st.subheader("Power Spectral Density Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_frequency_response(ax, noisy_input, fs, "Noisy Input PSD")
    plot_frequency_response(ax, hybrid_output, fs, "Hybrid Output PSD")
    ax.legend(["Noisy Input PSD", "Hybrid Output PSD"])
    st.pyplot(fig, use_container_width=True)

    # Noise reduction heatmap
    st.subheader("Noise Reduction Over Time")
    window_len_samples = fs // 8
    noise_floor = np.abs(noisy_input - hybrid_output)
    noise_floor_reshaped = noise_floor[:len(noise_floor) - (len(noise_floor) % window_len_samples)]
    heatmap_data = noise_floor_reshaped.reshape(-1, window_len_samples).mean(axis=1)
    st.bar_chart(heatmap_data)

    # Audio playback section
    st.subheader("Audio Playback Section")

    def play_audio(label, data):
        st.markdown(f"**{label}**")
        st.markdown(audio_player(data * volume, fs), unsafe_allow_html=True)

    play_audio("Clean Speech", speech)
    play_audio("Noise Only", noise)
    play_audio("Noisy Input", noisy_input)
    play_audio("LMS Filtered Output", lms_output)
    play_audio("ML Enhanced Output", ml_output)
    play_audio("Hybrid Output", hybrid_output)

if __name__ == "__main__":
    main()

