# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:32:44 2020

@author: chris
"""
import numpy as np
import scipy.stats as scistat
class Quantum_laws:
# Evolves the spin of the atom
    def evol(self):
    
        Quantum_laws.prob(self)                                                  #Calculates the normalized intensity
        Quantum_laws.calcdetuning(self)
            
        gamma=self.gammacom#+self.normintent*self.prob_intent+self.wallterm*self.wallscale       #Calculating the decay term taking into account both collision decay and
        
        gamma=gamma*(2*np.pi)
        
        F_y=np.random.normal(size=self.N_cell,scale=np.sqrt(gamma))  #Calculating the langevin forces
        F_z=np.random.normal(size=self.N_cell,scale=np.sqrt(gamma))
            
        magfreq=Quantum_laws.lamorf(self)
        #jm exists to prevent jz from being affected by its own effect on jy. If jz were not there the lamor terms would cause an atenuated sine wave for jz and jy.
        jm=self.jy
        self.jy = -np.sin(magfreq*self.t_step)*self.jz+np.cos(magfreq*self.t_step)*self.jy-gamma*self.jy*self.t_step+F_y*np.sqrt(self.t_step) #+a*np.sqrt(t_step)*np.random.normal(size=N_cell,scale=np.sqrt(prob)) #-t_step*lamor*jz-gamma*jy*t_step+F_y*np.sqrt(t_step)#-np.sin(lamor*t_step)*jz+np.cos(lamor*t_step)*jy#-gamma*jy*t_step+F_y*np.sqrt(t_step)#+np.normal(scale=prob)*jx*delta_t
        self.jz = np.sin(magfreq*self.t_step)*jm+np.cos(magfreq*self.t_step)*self.jz-gamma*self.jz*self.t_step+F_z*np.sqrt(self.t_step) # t_step*lamor*jy-gamma*jz*t_step+F_z*np.sqrt(t_step)# np.sin(lamor*t_step)*jy+np.cos(lamor*t_step)*jz#-gamma*jz*t_step+F_z*np.sqrt(t_step)
    
    def lamorf(self):
        if self.lamor=="":
            return np.interp(self.s[:,2], self.magnetic_x, self.magnetic_y)#+self.inhomogeneity*self.s[:,2]**2#+self.maxstarkshift*self.normintent
        else:
            return self.lamor
        
    def prob(self):
        self.normintent=Quantum_laws.Intent(self,self.s[:,0],self.s[:,1],self.s[:,2])/self.Intentnormalization
        
    
    def Intent(self,x,y,z):
        
        if self.beam=="gaus":
            w_z_2 = 1+z**2*self.A_Rayleigh
            
            return w_z_2*np.exp(-2*(x**2+y**2)/w_z_2/self.w_0_2)
        
        elif self.beam=="tophat":
            x_in = np.where(np.abs(x)>self.tophat,0,1)
            y_in = np.where(np.abs(y)>self.tophat,0,1)
            return x_in*y_in
        
        elif self.beam=="mode":
            return self.intensity_mode(x,y,grid=False)
        
        else:
            print("error intensity")
    
    def calcdetuning(self):
        dopplershift=self.v[:,2]/self.Lambda
        
        detuning=self.staticdetuning+dopplershift
        #gammacom=9*5*10**19./detuning**2            #1-math.exp(-10**3*t_step)       #The probability of the atom getting a random phase each timestep 
        self.a=3*10**9/detuning




class Classical_laws:
    
    def eulers(self):
        self.s=self.s+self.v*self.t_step                #Move atoms

    
    def walls(self):
        #Find which atoms have collided with which walls
        sign,x,x_len=Classical_laws.collisions(self)
        Classical_laws.position(self,x,sign)
        Classical_laws.velocity(self,sign, x, x_len)
        Classical_laws.walldecay(self,x,x_len)

    
        # Finds which atoms have collided with other atoms
    def collisions(self):
        sign=[]
        x=[]
        x_len=np.array([1,1,1])
        for n in range(3):        
            x.append(self.arange[np.absolute(self.s[:,n]) > self.cell[n]] )   #checking the x direction
            x_len[n] = len(x[n])                        #Finding the number of particels that colided with the wall   
            sign.append(np.sign(self.s[x[n],n]))             #Finds which wall they collided with
        return sign,x,x_len
    
    
    #Sets the position of the atoms outside the cell on the border of the cell such that no atom is found outside the cell.
    def position(self,x,sign):

        for n in range(3):
            self.s[x[n],n] = self.cell[n]*sign[n]
            
    
    # Gives a new velocity to the atoms refferenced by x
    def velocity(self,sign,x,x_len):
        for n in range(3):
            #print(x_len[n])
            theta=np.arcsin(np.sqrt(np.random.rand(x_len[n])))     #Finds the angle the atom bounce off at with respect to the x axis
            phi=np.random.rand(x_len[n])*2*np.pi                   #Finds the angle with respect to the z axis 
            absV=scistat.maxwell.rvs(scale=self.sigma,size=x_len[n])    #Finds the magnitude of the velocity
            self.v[x[n],n]=-absV*np.cos(theta)*sign[n]                  #Assigns Velocity
            self.v[x[n],n-2]=absV*np.sin(phi)*np.sin(theta)             #Assigns Velocity. The minus is to make the code work for y and z.
            self.v[x[n],n-1]=absV*np.cos(phi)*np.sin(theta)             #Assigns Velocity
            
            # log1=log1+list(theta)
    
    
    
    # Increases the gamma factor for the atoms that are refferenced by x 
    def walldecay(self,x,x_len):
        self.wallterm[:]=0
        for n in range(3):
            self.wallterm[x[n]]=1
            
    def col(self):
        #lam=1/mu=m/tau
        nu=np.random.normal(size=(self.N_cell,3),scale=np.sqrt(2*self.lam*self.k*self.T))
        self.v=self.v+(-self.lam*self.v*self.t_step+nu*np.sqrt(self.t_step))/self.m
        #print(((-lam*v*t_step+nu*np.sqrt(t_step))/m)[0])
   
        

class physics:
    def __init__(self,Nrounds=False,rounds=False):
        self.totalTime=self.totalTime+self.t_step  #Advance the clock
        Classical_laws.eulers(self)             #Move the pieces
        Classical_laws.col(self)                       #Handel atom-atom collisions     
        Classical_laws.walls(self)                    #Keep them in
        Quantum_laws.evol(self)                      #Change the phase
        
        self.UTT(Nrounds,rounds)                       #Record the phase
        if self.double_pass == True:
            self.v[:,2]=-self.v[:,2]
            Quantum_laws.evol(self)
            self.UTT(Nrounds,rounds)                    #Record the phase
            self.v[:,2]=-self.v[:,2]
