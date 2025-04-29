import torch
import threading
import queue
import time

class AsyncDistributedInference:
    """
    Simuliert asynchrone, verteilte Inferenz über mehrere Devices/Worker.
    Jeder Worker verarbeitet Aufgaben unabhängig und legt die Ergebnisse in eine gemeinsame Queue.
    """
    def __init__(self, model, devices):
        self.model = model
        self.devices = devices
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.threads = []
        for device in devices:
            t = threading.Thread(target=self.worker, args=(device,))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def _to_cpu(self, obj):
        if torch.is_tensor(obj):
            return obj.cpu()
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._to_cpu(o) for o in obj)
        elif isinstance(obj, dict):
            return {k: self._to_cpu(v) for k, v in obj.items()}
        else:
            return obj

    def worker(self, device):
        while True:
            item = self.task_queue.get()
            if item is None:
                break
            idx, x = item
            with torch.no_grad():
                x = x.to(device)
                out = self.model(x)
                out = self._to_cpu(out)
            self.result_queue.put((idx, out))
            self.task_queue.task_done()

    def infer(self, inputs):
        # Inputs: Liste von Tensoren
        for idx, x in enumerate(inputs):
            self.task_queue.put((idx, x))
        results = [None] * len(inputs)
        for _ in range(len(inputs)):
            idx, out = self.result_queue.get()
            results[idx] = out
        return results

# Beispiel/Test
if __name__ == "__main__":
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            time.sleep(0.2)
            return x * 2
    model = DummyModel()
    devices = [torch.device('cpu')]
    async_inf = AsyncDistributedInference(model, devices)
    batch = [torch.tensor([i], dtype=torch.float32) for i in range(5)]
    outs = async_inf.infer(batch)
    print("Ergebnisse:", [o.item() for o in outs])
