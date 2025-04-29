import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import io
from PIL import Image

def plot_async_distributed(n_devices=8, t_min=1.0, t_max=3.0):
    T_i = np.linspace(t_min, t_max, n_devices)
    R = np.sum(1 / T_i)
    R_ser = 1 / np.max(T_i)
    plt.figure(figsize=(5,3))
    plt.bar(range(1, n_devices+1), T_i, label='Rechenzeit pro Gerät')
    plt.axhline(1/R, color='g', linestyle='--', label='Paralleler Durchsatz')
    plt.axhline(1/R_ser, color='r', linestyle=':', label='Serieller Durchsatz')
    plt.xlabel('Gerät')
    plt.ylabel('Rechenzeit (s)')
    plt.title('Asynchrone Verteilung: Durchsatzvergleich')
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def plot_activity_routing(M=8, k=3, t=1.0):
    T_full = M * t
    T_dyn = k * t
    plt.figure(figsize=(5,3))
    plt.bar(['Voll', 'Dynamisch'], [T_full, T_dyn], color=['red', 'green'])
    plt.ylabel('Gesamtrechenzeit')
    plt.title('Aktivitätsbasiertes Routing')
    plt.tight_layout()
    return plt.gcf()

def plot_block_chunked(B=6, C=12, k=3, t=1.0):
    T_full = B * C * t
    T_comb = B * k * t
    ks = np.arange(1, C+1)
    plt.figure(figsize=(5,3))
    plt.plot(ks, B*ks*t, label='T_comb (Block-Chunked)')
    plt.hlines(T_full, 1, C, colors='r', linestyles='dashed', label='T_full (klassisch)')
    plt.xlabel('k (aktive Chunks pro Block)')
    plt.ylabel('Gesamtrechenzeit')
    plt.title('Block-Chunked Activity Routing')
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def plot_quantisierung(N=32, q_high=32, q_mid=8, q_low=1, p_high=0.2, p_mid=0.5):
    n_high = int(N * p_high)
    n_mid = int(N * p_mid)
    n_low = N - n_high - n_mid
    S = n_high*q_high + n_mid*q_mid + n_low*q_low
    plt.figure(figsize=(5,3))
    plt.bar(['High', 'Mid', 'Low'], [n_high*q_high, n_mid*q_mid, n_low*q_low], color=['blue','orange','grey'])
    plt.ylabel('Gesamtspeicherbedarf (Bit)')
    plt.title(f'Dynamische Quantisierung\nGesamt: {S} Bit')
    plt.tight_layout()
    return plt.gcf()

def plot_combined(B=6, C=12, k=3, t=1.0, q_high=32, q_mid=8, q_low=1, p_high=0.2, p_mid=0.5):
    # Kombiniere Block-Chunking und Quantisierung
    N = B*k
    n_high = int(N * p_high)
    n_mid = int(N * p_mid)
    n_low = N - n_high - n_mid
    S = n_high*q_high + n_mid*q_mid + n_low*q_low
    T_comb = B*k*t
    plt.figure(figsize=(6,3))
    plt.bar(['Rechenzeit','Speicherbedarf'], [T_comb, S], color=['green','blue'])
    plt.title('Kombinierte Effizienz (Block-Chunk + Quantisierung)')
    plt.tight_layout()
    return plt.gcf()

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

def gradio_interface(algorithm, **kwargs):
    if algorithm == 'Asynchrone Verteilung':
        params = {k: kwargs[k] for k in ['n_devices', 't_min', 't_max']}
        fig = plot_async_distributed(**params)
    elif algorithm == 'Aktivitätsbasiertes Routing':
        params = {k: kwargs[k] for k in ['M', 'k', 't']}
        fig = plot_activity_routing(**params)
    elif algorithm == 'Block-Chunked Routing':
        params = {k: kwargs[k] for k in ['B', 'C', 'k', 't']}
        fig = plot_block_chunked(**params)
    elif algorithm == 'Dynamische Quantisierung':
        params = {k: kwargs[k] for k in ['N', 'q_high', 'q_mid', 'q_low', 'p_high', 'p_mid']}
        fig = plot_quantisierung(**params)
    elif algorithm == 'Kombiniert':
        params = {k: kwargs[k] for k in ['B', 'C', 'k', 't', 'q_high', 'q_mid', 'q_low', 'p_high', 'p_mid']}
        fig = plot_combined(**params)
    else:
        fig = plt.figure()
    img = fig_to_image(fig)
    return img

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🚀 KI-Effizienz-Visualisierung – Für junge Tüftler!
    
    Willkommen! Hier kannst du ausprobieren, wie clevere Tricks KI-Modelle viel schneller und sparsamer machen. Wähle unten einen Algorithmus aus und spiele mit den Reglern – du siehst sofort, wie sich das auf die Rechenzeit und den Speicher auswirkt!
    
    **Was bedeutet das?**
    - Je kleiner die Balken, desto schneller und sparsamer ist die KI.
    - Mit diesen Tricks kann sogar ein Handy Sachen machen, die sonst nur Supercomputer konnten!
    
    **Was macht welcher Algorithmus?**
    - **Asynchrone Verteilung:** Viele Geräte rechnen gleichzeitig – wie ein großes Team.
    - **Aktivitätsbasiertes Routing:** Die KI rechnet nur da, wo wirklich was passiert.
    - **Block-Chunked Routing:** Die KI lädt immer nur kleine Teile und rechnet nur die wichtigsten davon.
    - **Dynamische Quantisierung:** Unwichtige Teile werden grob gerechnet, wichtige in hoher Qualität.
    - **Kombiniert:** Alle Tricks zusammen – maximaler Turbo!
    """)
    algo = gr.Dropdown([
        'Asynchrone Verteilung',
        'Aktivitätsbasiertes Routing',
        'Block-Chunked Routing',
        'Dynamische Quantisierung',
        'Kombiniert'],
        label="Algorithmus",
        info="Wähle einen Trick – Tipp: Kombiniert ist der Super-Turbo!"
    )
    with gr.Row():
        B = gr.Slider(2, 16, value=6, step=1, label="Blöcke (B)", info="Wie viele große Teile hat das Modell?")
        C = gr.Slider(2, 32, value=12, step=1, label="Chunks pro Block (C)", info="Wie viele kleine Stücke pro Block?")
        k = gr.Slider(1, 12, value=3, step=1, label="Aktive Chunks (k)", info="Wie viele Stücke werden wirklich gerechnet?")
        t = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Rechenzeit pro Chunk (t)", info="Wie lange dauert ein kleines Stück?")
        N = gr.Slider(8, 128, value=32, step=1, label="Anzahl Chunks (N)", info="Wie viele Stücke insgesamt?")
        q_high = gr.Slider(1, 32, value=32, step=1, label="q_high (Bit)", info="Qualität wichtiger Teile (z.B. 32 = super genau)")
        q_mid = gr.Slider(1, 32, value=8, step=1, label="q_mid (Bit)", info="Qualität mittlerer Teile")
        q_low = gr.Slider(1, 32, value=1, step=1, label="q_low (Bit)", info="Qualität unwichtiger Teile (1 = ganz grob)")
        p_high = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="% High Aktivität", info="Wie viel ist super wichtig?")
        p_mid = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="% Mid Aktivität", info="Wie viel ist mittel wichtig?")
        n_devices = gr.Slider(2, 32, value=8, step=1, label="Geräte (Async)", info="Wie viele Geräte rechnen mit?")
        t_min = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Min. Rechenzeit (Async)", info="Schnellstes Gerät")
        t_max = gr.Slider(0.1, 10.0, value=3.0, step=0.1, label="Max. Rechenzeit (Async)", info="Langsamstes Gerät")
    with gr.Row():
        plot = gr.Image(label="Effizienz-Graph", elem_id="main_plot", type="pil")
        # download_btn = gr.Button("📥 Bild herunterladen", elem_id="download_btn")
    def update(algo, B, C, k, t, N, q_high, q_mid, q_low, p_high, p_mid, n_devices, t_min, t_max):
        kwargs = dict(B=B, C=C, k=k, t=t, N=N, q_high=q_high, q_mid=q_mid, q_low=q_low, p_high=p_high, p_mid=p_mid, n_devices=n_devices, t_min=t_min, t_max=t_max)
        return gradio_interface(algo, **kwargs)
    inputs = [algo, B, C, k, t, N, q_high, q_mid, q_low, p_high, p_mid, n_devices, t_min, t_max]
    gr.Interface(fn=update, inputs=inputs, outputs=plot, live=True)
    
    # Custom CSS für größere Anzeige und "Vollbild"
    gr.HTML("""
    <style>
    #main_plot {
        min-width: 900px !important;
        min-height: 550px !important;
        max-width: 98vw !important;
        max-height: 80vh !important;
        margin: auto;
        border: 2px solid #3b82f6;
        border-radius: 10px;
        background: #f9fafb;
    }
    </style>
    """)

    gr.Markdown("""
    ---
    ## 📊 Typische Einstellungen & Beispiele
    
    Hier findest du Vorschläge, wie du die Regler für verschiedene Geräte einstellen kannst – und warum:
    
    | Gerät           | Blöcke (B) | Chunks (C) | Aktive Chunks (k) | Genauigkeit (q_high/q_mid/q_low) | Geräte (Async) | Erklärung |
    |-----------------|------------|------------|-------------------|-------------------------------|-----------------|-----------|
    | 🕹️ Handy        | 3–5        | 6–8        | 2–3               | 8 / 4 / 1                     | 1–2             | Wenig Speicher, wenig Power. KI muss sparsam sein! |
    | 💻 PC/Laptop    | 8–12       | 12–16      | 4–6               | 16 / 8 / 2                    | 2–4             | Mehr Power, kann mehr rechnen und speichern.      |
    | 🖥️ Supercomputer| 16–32      | 24–32      | 8–12              | 32 / 16 / 4                   | 8–32            | Mega viel Speicher und Power, alles geht schnell! |
    
    **Was bedeutet das für die Grafik?**
    - Handy: Balken werden kleiner, wenn du weniger Blöcke/Chunks und grobe Genauigkeit einstellst.
    - Supercomputer: Kann alles auf Maximum stellen – trotzdem bleibt die Grafik klein, weil viel Power da ist.
    
    **Tipp:**
    - Wenn du ausprobieren willst, wie eine KI auf dem Handy läuft, stell die Werte klein und die Genauigkeit niedrig.
    - Für einen Supercomputer dreh alles auf Maximum – dann siehst du, wie schnell und genau es gehen kann!
    
    ---
    ## 🤔 Begriffe einfach erklärt
    
    **Künstliche Intelligenz (KI):**
    Computer, die Dinge können wie Menschen – z.B. Bilder erkennen, Texte schreiben oder Spiele spielen.
    
    **Algorithmus:**
    Eine genaue Anleitung, wie der Computer etwas machen soll – wie ein Rezept beim Kochen.
    
    **Block:**
    Ein großer Teil im KI-Modell, der etwas Bestimmtes rechnet.
    
    **Chunk:**
    Ein kleines Stück von einem Block – wie ein Puzzleteil.
    
    **Aktivität:**
    Zeigt, wie wichtig oder "fleißig" ein Teil gerade ist. Viel Aktivität = da passiert was!
    
    **Quantisierung:**
    Die Werte werden "grob" gemacht, damit der Computer weniger rechnen muss – wie beim Malen mit dicken oder dünnen Pinseln.
    
    **Was zeigt die Grafik?**
    Sie zeigt, wie viel schneller oder sparsamer die KI mit den Tricks wird. Je kleiner die Balken, desto besser!
    
    ---
    **Tipp:** Wenn du keinen Graphen siehst, prüfe, ob Python-Pakete wie `matplotlib` installiert sind oder lade die Seite neu. Manchmal hilft es auch, einen anderen Algorithmus auszuwählen.
    """)

demo.launch()
