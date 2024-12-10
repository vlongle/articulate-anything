from typing import List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress
import logging
import traceback
import os
from threading import Lock
from collections import deque
import GPUtil
import datetime
import time


class GPUAllocator:
    def __init__(self, max_load_per_gpu=5):
        # max_load_per_gpu : max no. of tasks that can be allocated to a single gpu
        self.gpus = deque(
            GPUtil.getAvailable(order="memory", maxLoad=0.8,
                                maxMemory=0.8, limit=100)
        )
        self.gpu_loads = {gpu: 0 for gpu in self.gpus}
        self.lock = Lock()
        self.max_load_per_gpu = max_load_per_gpu

    def get_available_gpu(self):
        with self.lock:
            if not self.gpus:
                return None
            for _ in range(len(self.gpus)):
                gpu = self.gpus[0]
                self.gpus.rotate(-1)
                if self.gpu_loads[gpu] < self.max_load_per_gpu:
                    return gpu
            return None

    def allocate_gpu(self, gpu_id):
        with self.lock:
            self.gpu_loads[gpu_id] += 1

    def release_gpu(self, gpu_id):
        with self.lock:
            self.gpu_loads[gpu_id] = max(0, self.gpu_loads[gpu_id] - 1)


def process_tasks(tasks: List[Any],
                  process_func: Callable,
                  num_workers: int = 1,
                  max_load_per_gpu: int = 5,
                  *args, **kwargs) -> None:
    start = time.time()
    logging.info("Starting the main process")
    total_tasks = len(tasks)
    gpu_allocator = GPUAllocator(max_load_per_gpu=max_load_per_gpu)

    with Progress() as progress:
        task_progress = progress.add_task(
            f"[green]Processing tasks... 0/{total_tasks} (0.00%)",
            total=total_tasks
        )

        def update_progress():
            progress.update(task_progress, advance=1)
            task = progress.tasks[task_progress]
            progress.update(
                task_progress,
                description=f"[green]Processing tasks... {task.completed}/{task.total} ({task.percentage:.2f}%)"
            )

        def wrapped_process_func(task, *args, **kwargs):
            gpu_id = gpu_allocator.get_available_gpu()
            wait_time = 60
            while gpu_id is None:
                gpu_id = gpu_allocator.get_available_gpu()
                if gpu_id is None:
                    logging.warning(
                        f"No available GPU for task {task}. Waiting for {wait_time} seconds")
                    time.sleep(wait_time)
            try:
                gpu_allocator.allocate_gpu(gpu_id)
                logging.info(f"Running task {task} on GPU {gpu_id}")
                return process_func(task, str(gpu_id), *args, **kwargs)
            finally:
                gpu_allocator.release_gpu(gpu_id)

        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=min(num_workers, total_tasks)) as executor:
                futures = [executor.submit(
                    wrapped_process_func, task, *args, **kwargs) for task in tasks]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        logging.error(f'Task generated an exception:')
                        logging.error(traceback.format_exc())
                    finally:
                        update_progress()
        else:
            for task in tasks:
                try:
                    wrapped_process_func(task, *args, **kwargs)
                except Exception as exc:
                    logging.error(f'Task generated an exception:')
                    logging.error(traceback.format_exc())
                finally:
                    update_progress()

        end = time.time()
        logging.info(
            f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
