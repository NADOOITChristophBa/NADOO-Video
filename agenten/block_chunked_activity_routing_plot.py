import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

def compute_times(B=8, C=16, k=4, t_mean=1.0):
    # Simuliere Rechenzeiten pro Chunk als konstant
    tji = np.ones((B, C)) * t_mean
    T_full = np.sum(tji)
    T_comb = np.sum(tji[:, :k])
    return T_full, T_comb

def plot_efficiency(B=8, C=16, t_mean=1.0):
    ks = np.arange(1, C+1)
    T_full = B * C * t_mean
    T_combs = B * ks * t_mean
    speedup = T_full / T_combs
    plt.figure(figsize=(6,4))
    plt.plot(ks, T_combs, label='T_comb (Block-Chunked)')
    plt.hlines(T_full, 1, C, colors='r', linestyles='dashed', label='T_full (klassisch)')
    plt.xlabel('k (aktive Chunks pro Block)')
    plt.ylabel('Gesamtrechenzeit')
    plt.title('Effizienz von Block-Chunked Activity Routing')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

def gradio_interface(B, C, t_mean):
    fig = plot_efficiency(int(B), int(C), float(t_mean))
    return gr.Plot(fig)

with gr.Blocks() as demo:
    gr.Markdown("""
    # Visualisierung: Block-Chunked Activity Routing
    Zeigt die Gesamtrechenzeit $T_{comb}$ im Vergleich zum klassischen Ansatz $T_{full}$
    """)
    with gr.Row():
        B = gr.Slider(2, 32, value=8, step=1, label="Blöcke (B)")
        C = gr.Slider(2, 64, value=16, step=1, label="Chunks pro Block (C)")
        t_mean = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Rechenzeit pro Chunk (t)")
    plot = gr.Plot()
    B.change(gradio_interface, [B, C, t_mean], plot)
    C.change(gradio_interface, [B, C, t_mean], plot)
    t_mean.change(gradio_interface, [B, C, t_mean], plot)
    gradio_interface(B.value, C.value, t_mean.value)

demo.launch()
