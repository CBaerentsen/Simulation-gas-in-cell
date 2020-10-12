# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:39:23 2020

@author: chris
"""
import numpy as np
import scipy.stats as scistat
import time
import datetime
import os.path
from montecarlo_thermal_cell_physics import Quantum_laws, Classical_laws, physics
import pickle
import math
from scipy import interpolate
from playsound import playsound
from scipy.integrate import simps
from shutil import copyfile
from scipy.fft import fft2, ifft2
from scipy.interpolate import RectBivariateSpline

class montecarlo_thermal_cell:
    def __init__(self,cores=1,mode_x=0,mode_y=0):
        print("Setting up the simulation...")
        
        
# =============================================================================
#         Parameters to change:
# =============================================================================
        self.N_cell =int(5*10**(3)/cores)    #Number of atoms        

        self.x_cell = 1*1e-3 / 2   #Length of cell/2 in x- direction
        self.y_cell = 1*1e-3 / 2   #Length of cell/2 in y- direction
        self.z_cell = 40*1e-3 / 2      #Length of cell/2 in z- direction

        self.T = 273.15 + 55   #temperature of the cell
        self.m = 132.905451933 * (1.66053906660*1e-27)   #mass of one cesium atom        
                
        self.lam=self.m/10**-6    #float(10**-21)*1000
        
        self.beam="mode" #can take "gaus","tophat", "mode"
        self.tophat = (0.75*1E-3 )/2
        self.w_0_2 = (900*1/3*1e-6)**2 #the waist squared 
        self.double_pass=True
        
        self.t_step = 2.1853e-8# euler steps
        self.T_0=2e-3 #length of simulation in seconds
        
        self.Nrounds=10    #The number of simulations made before averageing
        self.prerun=1000      #The amount of timesteps taken before data is taken
        
        
        self.lamor=1372.5*10**3*2*np.pi                   #The lamor frequency
        
        # #
        # f = open("Larmor.pkl", 'rb')
        # data = pickle.load(f)
        # f.close()
        
        # self.magnetic_x=(data[0]-85)*1E-3
        # self.magnetic_y=data[1]*1E3 * 2*np.pi*32.7875
        # self.lamor=""
        # #
        
        self.staticdetuning=-3*10**9                         #-10**9 is 1 gigahertz blue detuning assuming no dopplershift
        self.gammacom=0#5*1E3 /2  #9*5*10**19/staticdetuning**2            #1-math.exp(-10**3*t_step)       #The probability of the atom getting a random phase each timestep 
        self.a=3*10**9/self.staticdetuning                      #a1 in Julsgaard
 
        
# =============================================================================
# Cell parameters
# =============================================================================

        self.cell=np.array([self.x_cell,self.y_cell,self.z_cell])

        self.k = 1.380649*1e-23    #Boltzmann constant

    
        A_maxwell = (self.m/(2*np.pi*self.k*self.T))**(1/2)    #Amplitude of the Thermal distribution
        E_maxwell = self.m/(self.k*self.T)  #Exponential amplitude of the Thermal distribution
        self.sigma = (E_maxwell)**(-1/2) #Scaling for random distribution

# =============================================================================
#  probe parameters
# =============================================================================
        
        

        self.Lambda = 852.3650*1e-9  #the wavelength of the beam
    
        self.A_Rayleigh = (self.Lambda/(np.pi*self.w_0_2))**2 #z_R^2 where z_R is the rayleigh range
        
        if self.beam == "mode":
            mode_precision=np.int(1E3)
            a=np.zeros((mode_precision,mode_precision), dtype=complex)
            x=np.linspace(-self.x_cell, self.x_cell,mode_precision)
            y=np.linspace(-self.y_cell, self.y_cell,mode_precision)
            A=mode_precision**2/1E2
            a[mode_x,mode_y]=np.complex(A,0)
            a[0,0]=np.complex(np.sum(a),0)
            f = np.real(ifft2(a))
            self.intensity_mode = RectBivariateSpline(x, y, f)

        

# =============================================================================
#   Simulations parameters
# =============================================================================


        self.T_int=0  #length of scrabling in seconds
    
        self.Pro=0   #the procentage of the the simulation (starting at 0%)
        self.status=True
    

        self.rounds=int(self.T_0/self.t_step)

    
        self.arange = np.arange(self.N_cell)

  
##%% Initial state
        self.s = np.random.rand(self.N_cell,3)-0.5 #position
        self.s[:,0] = self.s[:,0]*self.x_cell * 2  #generation of the atoms x component
        self.s[:,1] = self.s[:,1]*self.y_cell * 2  #generation of the atoms y component
        self.s[:,2] = self.s[:,2]*self.z_cell * 2  #generation of the atoms z component

    
        self.v = np.random.normal(0,self.sigma,(self.N_cell,3)) #velocity vector
    
        self.v = np.random.normal(0,self.sigma,(self.N_cell,3)) #velocity vector
        self.v_init=self.v
    
        #light experienced by the atom
        np.random.default_rng(42)       #seeds the rng
        self.phase=np.random.rand(self.N_cell)*2*np.pi  #The initial phase of the spin of all atoms is set between 0 and 2pi
        self.jz=np.cos(self.phase)                        # jz components of the spin of the atoms
        self.jy=np.sin(self.phase)                        # jz components of the spin of the atoms
    
        self.totalTime=0                     #The amount of time that has passed in the simultaion
        
        

        
        self.intmax=Quantum_laws.Intent(self,0,0,0)            #The maximum intensity
        self.intmin=Quantum_laws.Intent(self,self.x_cell,self.y_cell,self.z_cell)#The minimum intensity
        
        
       # self.inhomogeneity=0.0*self.lamor/self.z_cell**2      #The maximum value of the inhomogeneity
        self.wallscale=1000#-wallgammathing/t_step  #The amount that the gamma factor should be scaled with after a wall colision
        self.wallterm=np.zeros([self.N_cell])           #Contains the atomsthat has colided with the wall
        self.prob_intent=100                        #The proportionality constant between the odds of decay and the fraction of the local intensity versus the average intensity
        
        

    
    
    
    def run_multithread(self,worker,sound=False):
        worker = "Worker " + str(worker) + ": "
        self.setup()
        for prerun in range(self.prerun):
            physics.__init__(self)     #run the simulation prerun times before taking data
        
        print(worker + "Setup completed")
        print(worker + "Running the simulation...")
        self.t_0=0
        self.t = time.time() #for the time of the ongoing similation
        
        for Nrounds in range(self.Nrounds):
            rounds=0
            for int_1 in range(10):
                
                for int_2 in range(int(self.rounds/10)):
                    physics.__init__(self,Nrounds,rounds)
                    rounds+=1
                self.timer(10/self.Nrounds, worker)
        print(worker + "Jobs Done!")
        if sound==True:
            playsound("C:/Users/chris/Music/JobsDone.mp3")
        self.save()
    
    def UTT(self,Nrounds,rounds):
        if Nrounds!=False:
            measurement = np.sqrt(1/self.N_cell)*np.sum(self.a*self.normintent*self.jz)#+571/17365*np.sum(np.random.normal(scale=np.sqrt(normintent)))#ændre?
            self.timetrace[Nrounds,rounds] += measurement
    
    def setup(self):
        self.timetrace = np.zeros((self.Nrounds,int(self.rounds/10)*10))
        
        x = np.linspace(-self.x_cell, self.x_cell, 2000)
        y = np.linspace(-self.y_cell, self.y_cell, 2000)
        z = np.linspace(-self.z_cell, self.z_cell, 30)
        a=time.time()
        volume=0
        for z1 in z:
            I = Quantum_laws.Intent(self,x[:,None],y,z1)
            volume+=simps(simps(I, y), x)
        volume=volume
        self.Intentnormalization=volume/2.5119744175472717e-05
        
        
    
    
    def save(self):
        today = datetime.date.today()
        d1 = today.strftime('%d_%m_%Y')
        i=0
        d = d1+'_'+str(i)
        while os.path.isfile(d) == True:
            i+=1
            d = d1+"_" + str(i)
        self.filename = d
        f = open(self.filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
        d2 = d + "_Parameters" + ".txt"
        copyfile("montecarlo_thermal_cell_class.py", d2)
        d3 = d  + "_Physics" + ".txt"
        copyfile("montecarlo_thermal_cell_physics.py", d3)
        
    def timer(self, w, worker=""):
        if self.status == True:
            time_update = self.t_0
            self.t_0 = time.time()-self.t             # Calculate time since start
            self.Pro=self.Pro+w                       # Calculate procentage of compleation
            
            if self.Pro < 10:
                a='  '
            elif self.Pro >= 10:
                a=' '
            timeleft = (self.t_0-time_update)*(100-self.Pro)/w
            timeleft_hours = math.floor(timeleft/3600)
            timeleft_min = math.floor((timeleft-3600*timeleft_hours)/60)
            timeleft_s = math.floor((timeleft-3600*timeleft_hours-60*timeleft_min))
            
            t_0_hours = math.floor(self.t_0/3600)
            t_0_min = math.floor((self.t_0-3600*t_0_hours)/60)
            t_0_s = math.floor((self.t_0-3600*t_0_hours-60*t_0_min))
            # Print procentages
            print(worker + 'Simulated: %.1f%% ' % self.Pro + a + ' Time: %02d:' \
                  % t_0_hours + "%02d:" % t_0_min + "%02d" % t_0_s + \
                  "\t Expected time left: %02d:" % timeleft_hours + "%02d:" % timeleft_min + "%02d" % timeleft_s)

