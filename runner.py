import torch

class ExperimentRunnerBase:
    def __init__(self, gpu_quotas=None):
        self.GPU_QUOTAS = gpu_quotas or {6: 3, 7: 3}

    def run_experiment(self):
        return NotImplementedError()

    def get_available_gpu(self, gpu_remained):
        available_gpus = [gpu_id for gpu_id in gpu_remained.keys() 
                            if gpu_remained[gpu_id] > 0]
        if not available_gpus:
            return None
        selected_gpu = max(available_gpus, key=lambda gpu_id: gpu_remained[gpu_id])
        gpu_remained[selected_gpu] -= 1
        return selected_gpu

    def release_gpu(self, gpu_id, gpu_remained):
        assert gpu_remained[gpu_id] < self.GPU_QUOTAS[gpu_id]
        gpu_remained[gpu_id] += 1
        torch.cuda.empty_cache()

    def run_parallel_experiment(self, all_experiments):
        return NotImplementedError()

import copy
import time
import threading


class ExperimentRunnerThread(ExperimentRunnerBase):
    def run_single_experiment_thread(self, exp, gpu_remained):
        gpu_id = None
        while gpu_id is None:
            with self.lock:
                gpu_id = self.get_available_gpu(gpu_remained)
            if gpu_id is None:
                time.sleep(2)
        self.run_experiment(exp, gpu_id)
        with self.lock:
            self.release_gpu(gpu_id, gpu_remained)

    def run_parallel_experiment(self, all_experiments):
        self.lock = threading.Lock()
        num_threads = min(sum([self.GPU_QUOTAS[n] for n in self.GPU_QUOTAS.keys()]), len(all_experiments))
        print(f"Number of experiments: {len(all_experiments)}, {self.GPU_QUOTAS=}, "
              f"Starting {num_threads} threads")
        # Shared GPU usage dictionary (threads share memory)
        gpu_usage = copy.deepcopy(self.GPU_QUOTAS)
        threads = []
        results = []
        # Create a semaphore to limit concurrent threads
        semaphore = threading.Semaphore(num_threads)
        def worker(exp):
            with semaphore:
                result = self.run_single_experiment_thread(exp, gpu_usage)
                results.append(result)
        # Start all threads
        for exp in all_experiments:
            thread = threading.Thread(target=worker, args=(exp,))
            threads.append(thread)
            thread.start()
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        return results
    
import multiprocessing as mp

class ExperimentRunnerProcess(ExperimentRunnerBase):
    def init_pool_processes(self, the_lock):
        global lock
        lock = the_lock

    def pair_experiment_with_gpu(self, config):
        exp, gpu_remained = config
        with lock:
            gpu_id = self.get_available_gpu(gpu_remained)
        result = self.run_experiment(exp, gpu_id)
        with lock:
            self.release_gpu(gpu_id, gpu_remained)

        return result    

    def run_parallel_experiment(self, all_experiments):
        mp.set_start_method('spawn', force=True)
        num_processes = min(sum([self.GPU_QUOTAS[n] for n in self.GPU_QUOTAS.keys()]), len(all_experiments))
        print(f"Number of experiments: {len(all_experiments)}, {self.GPU_QUOTAS=}, "
              f"Starting {num_processes} processes")
        
        with mp.Manager() as manager:
            lock = mp.Lock()
            gpu_usage = manager.dict(copy.deepcopy(self.GPU_QUOTAS))
            exp_gpu_usages = [(exp, gpu_usage) for exp in all_experiments]
            with mp.Pool(
                processes=num_processes,
                initializer=self.init_pool_processes, initargs=(lock,)
            ) as pool:
                pool.map(self.pair_experiment_with_gpu, exp_gpu_usages)


__all__ = ['ExperimentRunnerThread', 'ExperimentRunnerProcess']


