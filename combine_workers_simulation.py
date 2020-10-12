# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:41:43 2020

@author: chris
"""
import numpy as np
import os
from shutil import copyfile
import pickle



class combine_workers:
    def __init__(self,d1,cores,nextfile):
        
        filename = d1+"_" + str(nextfile-cores)
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict) 
        for j in range(cores-1):
            path = d1+"_" + str(nextfile-cores+j+1)
            function = collect_workers(path)
            s,v,timetrace = function.returnSelfs()
            
            self.v=np.append(self.v,v)
            self.s=np.append(self.s,s)
            self.timetrace+=timetrace
        
        self.timetrace=self.timetrace/np.sqrt(cores)
        self.N_cell=cores*self.N_cell
        for j in range(cores):
            path = d1+"_" + str(nextfile-cores+j)
            path1 = path + "_Parameters" + ".txt"
            path2 = path  + "_Physics" + ".txt"
            os.remove(path)
            os.remove(path1)
            os.remove(path2)
            
        
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
        d2 = filename + "_Parameters_"+str(cores)+"cores" + ".txt"
        copyfile("montecarlo_thermal_cell_class.py", d2)
        d3 = filename  + "_Physics" + ".txt"
        copyfile("montecarlo_thermal_cell_physics.py", d3)
        
        
class collect_workers:
    def __init__(self,filename):

        filename
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 
    
    def returnSelfs(self):
        return self.s,self.v,self.timetrace