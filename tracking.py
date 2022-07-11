#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs. 
# You have to use another script to generate the lists of commands
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.


import threading
import time
import os
import numpy as np


import gpustat
import logging

import itertools
import argparse

parser = argparse.ArgumentParser(description='PyTorch File Tracking Available GPUs')
parser.add_argument('--target-gpus', type=int, nargs='+', default=[0,1,2,3,4,5,6,7],
                    help='Which GPUs can you use?')
parser.add_argument('--memory-threshold', type=int, default = 200, help='memory threshold')
#TODO: This part needs to be updated!
parser.add_argument('--wait-cluster', dest='wait_cluster', default = False, action='store_true',
                        help='Do we need to wait for the targeted cluster to be clean and start our experiment?')
parser.add_argument('--command-file', type=str, default = './command_list_files/command_list.txt', help='The commands to run')
parser.add_argument('--gpu-query-time', type=int, default = 10)
parser.add_argument('--submit-wait-time', type=int, default = 10)
parser.add_argument('--one-task-per-gpu', dest='one_task_per_gpu', default = False, action='store_true',
                        help='Do we enforce one task per gpu?')

args = parser.parse_args()


FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
FORMAT_MINIMAL = '%(message)s'

logger = logging.getLogger('runner')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


exitFlag = 0
GPU_MEMORY_THRESHOLD = args.memory_threshold # in MB

## If we need to wait for the entire clean cluster to start, select False here

if args.wait_cluster:
    all_empty = {"ind": False}
else:
    all_empty = {"ind": True}
    
target_GPUs = args.target_gpus
num_target_GPUs = len(target_GPUs)
used_GPUs = set()

def num_available_GPUs(gpus):
    
    sum_i = 0
    for i, stat in enumerate(gpus):
        if stat['memory.used'] < 100 and i in target_GPUs:
            sum_i += 1
    return sum_i


def get_free_gpu_indices():
    '''
        Return an available GPU index.
    '''
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        # print('stats length: ', len(stats))
        return_list = []
        
        if num_available_GPUs(stats.gpus) >= num_target_GPUs and all_empty["ind"] == False:
            logger.info("Previous experiments all finished, try waiting for 30s!")
            time.sleep(30)
            if num_available_GPUs(stats.gpus) >= num_target_GPUs:
                logger.info("Previous experiments all finished, and have waited for 30s! Start experiments.")
                all_empty["ind"] = True
            else:
                logger.info("Previous experiments not finished...")
                time.sleep(10)
                continue
            
        if not all_empty["ind"]:
            logger.info("Previous experiments not finished...")
            time.sleep(10)
            continue
        
        for i, stat in enumerate(stats.gpus):
            memory_used = stat['memory.used']
            if memory_used < GPU_MEMORY_THRESHOLD and i in target_GPUs:
                if not args.one_task_per_gpu:
                    return i
                elif f'{i}' not in used_GPUs:
                    return i

        logger.info("Waiting on GPUs")
        time.sleep(args.gpu_query_time)

        
class DispatchThread(threading.Thread):
    def __init__(self, threadID, name, counter, bash_command_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.bash_command_list = bash_command_list

    def run(self):
        logger.info("Starting " + self.name)
        # print_time(self.name, self.counter, 5)
        threads = []
        for i, bash_command in enumerate(self.bash_command_list):
             
            import time
                
            time.sleep(0.1)
            
            cuda_device = get_free_gpu_indices()
            thread1 = ChildThread(1, f"{i}th + {bash_command}", 1, cuda_device, bash_command)
            thread1.start()
            
            time.sleep(args.submit_wait_time)
            threads.append(thread1)

        # join all.
        for t in threads:
            t.join()
        logger.info("Exiting " + self.name)


class ChildThread(threading.Thread):
    def __init__(self, threadID, name, counter, cuda_device, bash_command):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.cuda_device = cuda_device
        self.bash_command = bash_command

    def run(self):
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device[0]},{self.cuda_device[1]}'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device}'
        bash_command = self.bash_command

        logger.info(f'executing {bash_command} on GPU: {self.cuda_device}')
        used_GPUs.add(f'{self.cuda_device}')
        logger.info(f"Currently used GPUs: {used_GPUs}")
        # ACTIVATE
        os.system(bash_command)
        import time
        import random
        
        time.sleep(random.random() % 5)
        
        logger.info("Finishing " + self.name)
        used_GPUs.remove(f'{self.cuda_device}')
        logger.info(f"Removing {self.cuda_device} from used GPUs. Currently used GPUs: {used_GPUs}")


BASH_COMMAND_LIST = []

with open(args.command_file) as f:
    lines = f.readlines()
    for line in lines:
        if line and not line.isspace(): # Sometimes the script contains rows that are completely empty
            BASH_COMMAND_LIST.append(line)
        else:
            logger.info("Skipping empty lines in the command list.")


# Create new threads
dispatch_thread = DispatchThread(2, "Thread-2", 4, BASH_COMMAND_LIST[:])

# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")