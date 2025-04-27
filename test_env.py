import sys
import importlib
import subprocess
import socket
import time

REQUIRED = [
    ('torch', 'torch'),
    ('diffusers', 'diffusers'),
    ('transformers', 'transformers'),
    ('gradio', 'gradio'),
    ('sentencepiece', 'sentencepiece'),
    ('PIL', 'pillow'),
    ('av', 'av'),
    ('numpy', 'numpy'),
    ('scipy', 'scipy'),
    ('requests', 'requests'),
    ('einops', 'einops'),
    ('cv2', 'opencv-contrib-python'),
    ('safetensors', 'safetensors')
]

def test_imports():
    failed = []
    for import_name, pip_name in REQUIRED:
        try:
            importlib.import_module(import_name)
        except ImportError:
            failed.append(pip_name)
    if failed:
        print("Missing packages:", ', '.join(failed))
        return False
    print("All required packages import successfully.")
    return True

def test_gradio_launch():
    # Try to launch a minimal Gradio app and check port
    import gradio as gr
    def dummy(x): return x
    iface = gr.Interface(dummy, gr.Textbox(), gr.Textbox())
    try:
        iface.launch(server_name='127.0.0.1', server_port=7861, share=False, prevent_thread_lock=True)
        time.sleep(2)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex(('127.0.0.1', 7861))
        s.close()
        iface.close()
        if result == 0:
            print("Gradio test interface launched successfully.")
            return True
        else:
            print("Gradio interface did not launch on expected port.")
            return False
    except Exception as e:
        print("Gradio launch failed:", e)
        return False

def main():
    ok = test_imports()
    if not ok:
        sys.exit(1)
    ok = test_gradio_launch()
    if not ok:
        sys.exit(2)
    print("Environment tests passed.")

if __name__ == "__main__":
    main()
