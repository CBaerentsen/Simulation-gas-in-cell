# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:55:43 2020

@author: chris
"""
from montecarlo_thermal_cell_class import montecarlo_thermal_cell
from multiprocessing import Process, Queue
from combine_workers_simulation import combine_workers
import numpy as np
import os
import datetime
import psutil
import time
import sys
import signal

def run(workers,sound,cores):
    simulation = montecarlo_thermal_cell(cores)
    simulation.run_multithread(workers,sound)
    return
    
if __name__ == '__main__':
    sound=False
    cores = input("Choose the number of processes/cores to be used: ")
    cores = int(cores)
    jobs=[]
    for i in range(cores):
        p = Process(target=run, args=(i,sound,cores))
        jobs.append(p)
        p.start()
        
    for job in jobs:
        job.join()
    
    time.sleep(3)
        
    
    today = datetime.date.today()
    d1 = today.strftime('%d_%m_%Y')
    i=0
    d = d1+'_'+str(i)
    while os.path.isfile(d) == True:
        i+=1
        d = d1+"_" + str(i)
    nextfile = i
    combine_workers(d1,cores,nextfile)
    print("Simulation has finished")
    response = input("Do you want to run more simulations? [yes] or [no]: ")
    
    if response == "yes" or response == "Yes":
        print("\n")
        exec(open('run_simulation_multithreading.py').read())
        
    if response == "no" or response == "No":
        os.kill(os.getpid(), 9)
    