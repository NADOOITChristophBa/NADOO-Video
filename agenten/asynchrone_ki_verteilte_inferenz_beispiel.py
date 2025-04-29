"""
Beispiel: Asynchrone & verteilte KI-Inferenz
Mehrere Geräte bearbeiten Aufgaben unabhängig und asynchron.
"""
import threading
import queue
import time
import random

def worker(device_id, task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break
        # Simuliere variable Rechenzeit
        t = random.uniform(0.5, 2.0)
        print(f"Gerät {device_id} bearbeitet Aufgabe {task} ({t:.2f}s)")
        time.sleep(t)
        result_queue.put((device_id, task, t))
        task_queue.task_done()

def main():
    num_devices = 4
    num_tasks = 8
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    threads = []
    for i in range(num_devices):
        t = threading.Thread(target=worker, args=(i, task_queue, result_queue))
        t.start()
        threads.append(t)
    for i in range(num_tasks):
        task_queue.put(i)
    task_queue.join()
    for _ in threads:
        task_queue.put(None)
    for t in threads:
        t.join()
    while not result_queue.empty():
        device_id, task, t = result_queue.get()
        print(f"Ergebnis: Gerät {device_id} löste Aufgabe {task} in {t:.2f}s")

if __name__ == "__main__":
    main()
