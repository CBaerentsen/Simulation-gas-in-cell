# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:39:23 2020

@author: chris
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scistat
import pickle
import scipy.signal as signal
import math
from lmfit import Model, minimize
from scipy.ndimage import gaussian_filter1d as smoothen

class plotting:
    def __init__(self,filename,color="blue",name=""):
        if name =="":
            name=filename
        self.name=name
        self.color=color
        self.filename=filename
        f = open(self.filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 
        
    def hist_speed(self):
        _,bins,_=plt.hist(np.sqrt(self.v[:,0]**2+self.v[:,1]**2+self.v[:,2]**2), bins=75)
        # _,bins,_=plt.hist(scistat.maxwell.rvs(scale=sigma,size=N_cell), bins=100)
        # _,bins,_=plt.hist(mem, bins=50)
        binsize=bins[1]-bins[0]
    #    plt.xlim([-x_cell-x_cell/20, x_cell+x_cell/20])
    #    plt.ylim([0,300])
    
        x=range(600)
        plt.plot(scistat.maxwell.pdf(x,scale=self.sigma)*self.N_cell*binsize,linewidth=8)
        # plt.legend(('maxwell distribution','atom speed'),loc='best',frameon=False,fontsize=48)
        
        plt.xlabel(r'$\mathrm{Speed}$'+' '+r'$ [\frac{m}{s}]$',fontsize=30)
        plt.ylabel(r'$\mathrm{\# Atoms}$',fontsize=30)
        
        plt.tick_params(axis="x", labelsize=30)
        plt.tick_params(axis="y", labelsize=30)
    
    def at(self):
        print(self.a)
        _,bins,_=plt.hist(self.a, bins=1000, range=[-10,0])
        plt.xlabel(r'$\mathrm{a_1}$',fontsize=30)
        plt.ylabel(r'$\mathrm{\# Atoms}$',fontsize=30)
        
        plt.tick_params(axis="x", labelsize=30)
        plt.tick_params(axis="y", labelsize=30)
        # _,bins,_=plt.hist(scistat.maxwell.rvs(scale=sigma,size=N_cell), bins=100)

    def hist_xy(self):
    
        plt.figure(self.filename + "\t hist_xy")
        plt.hist2d(self.s[:,0], self.s[:,1], bins=(50, 50), cmap=plt.cm.jet)
        plt.xlim([-self.x_cell-self.x_cell/20, self.x_cell+self.x_cell/20])
        plt.ylim([-self.y_cell-self.y_cell/20, self.y_cell+self.y_cell/20])
        plt.colorbar()
        
    
    
    def hist_xz(self):

        plt.figure(self.filename + "\t hist_xz")
        plt.hist2d(self.s[:,0], self.s[:,2], bins=(50, 50), cmap=plt.cm.jet)
        plt.xlim([-self.z_cell-self.z_cell/20, self.z_cell+self.z_cell/20])
        plt.ylim([-self.z_cell-self.z_cell/20, self.z_cell+self.z_cell/20])
        plt.colorbar()
    
    
    def PSD(self,resolution=-1,normalize=1,smooth=False,minus=0):
        self.normalize=normalize
        self.FourierTransform(resolution)
        self.Fourier+=-minus
        if smooth!= False:
            self.Fourier = smoothen(self.Fourier,smooth)
        plt.plot(self.freq,self.Fourier,ls='-',linewidth=2,alpha=0.6, color=self.color, label=self.name)  #Plot fourier transform
        self.Fouriermax=np.argmax(self.Fourier)                      #Find peak
        plt.subplots_adjust(right=0.75)
        #Plot peak and write out the x-component of the peak
        # plt.plot(freq[Fouriermax],Fourier2[Fouriermax],color='red',marker='.',figure=fig, clip_box=matplotlib.transforms.Bbox([[0,0],[0.5,0.5]]))
        # print('Peaket er ved: '+str(self.freq[Fouriermax])+' kHz')
        plt.xlabel('kHz',fontsize=32)                   #Lable axis
        plt.ylabel('PSD',fontsize=32)
        # plt.legend(('Fourier transform','peak'),fontsize=10)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        #plt.tick_params(axis='y',size=24)
        leg= plt.legend()
        plt.rc('legend',fontsize=14) 
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        

        plt.yscale('log')
        plt.show()
        return self.freq,self.Fourier

        
    def fit(self, center=41.79,width=10):
        boundaries=[center-width/2,center+width/2]
        f_model = self.freq[(self.freq>boundaries[0]) & (self.freq<boundaries[1])]
        psd_model = self.Fourier[(self.freq>boundaries[0]) & (self.freq<boundaries[1])]
        
        minimum=np.min(psd_model)
        psd_model=psd_model
        omega0=self.freq[self.Fouriermax]; gamma=0.037; A=5*1E1; B=minimum
        model = Model(plotting.Lorentzian)
        model.set_param_hint('B', value = B, min=0)
        model.set_param_hint('A', value = A, min=0)
        model.set_param_hint('omega0', value = omega0, min=omega0-0.01,max=omega0+0.01)
        model.set_param_hint('gamma', value = gamma, min=0)
        
        weights=(psd_model)**(-1)
        result = model.fit(psd_model, omega=f_model)
        plt.plot(f_model,result.init_fit,'-')
        value = result.values["gamma"]*1E3
        # print(result.values["B"]*minimum)
        # print(value)
        text = "FWFM = %0.1f Hz" % value
        plt.plot(f_model,result.best_fit,'-',color=self.color,label=text)
        plt.legend()
        
        print(result.fit_report())
    
    def broadfit(self):
        center=self.freq[self.Fouriermax]
        width=10
        boundaries=[center-width/2,center+width/2]
        f_model = self.freq[(self.freq<boundaries[0]) | (self.freq>boundaries[1])]/1E3
        psd_model = self.Fourier[(self.freq<boundaries[0]) | (self.freq>boundaries[1])]

        minimum=np.min(psd_model)
        psd_model=psd_model/minimum
        
        omega0=center/1E3; gamma=0.4; A=7*1E4; B=1.4
        model = Model(plotting.Lorentzian)
        model.set_param_hint('B', value = B,min=1.1)
        model.set_param_hint('A', value = A)
        model.set_param_hint('omega0', value = omega0,vary=False)
        model.set_param_hint('gamma', value = gamma, min=0)
   
        
        
        weights=psd_model**(-1)
    
        result = model.fit(psd_model, omega=f_model,weights=weights)
        plt.plot(f_model*1E3,result.init_fit,'-',color=self.color)
        value = result.values["gamma"]*1E3
        print(result.values["B"])
        # print(value)
        text = "FWFM = %0.1f kHz" % value
        plt.plot(f_model*1E3,result.best_fit*minimum,'-',color=self.color,label=text)
        plt.legend()
        
        # y=plotting.Lorentzian(f_model,result.values["omega0"]+1E1,result.values["gamma"],result.values["A"],result.values["B"])
        # plt.plot(f_model*1E3,y)
        
        # print(result.fit_report())
    
    
    def LinearResponse(omega,omega0,gamma,A,B):
        return np.abs(A / (omega0**2-omega**2+(gamma/2)**2+1j*gamma*omega))**2+B
    
    def Lorentzian(omega,omega0,gamma,A,B):
        return A*(gamma/2 / ((omega-omega0)**2+(gamma/2)**2))+B
        
    def FourierTransform(self,resolution):
        factor=2
        self.timetrace=self.timetrace/self.normalize
        if resolution==-1:
            resolution=1/self.T_0
        split=int(resolution*self.T_0)
        if split>=1:
            length=math.floor(len(self.timetrace)/split)
            self.Fourier=np.zeros(math.floor(length/factor))
            for n in range(split):
                T=self.timetrace[0+n*length:(n+1)*length]
                A=np.hanning(len(T))
                A=1
                Fourier=np.fft.fft(T*A)  #Get fourier transform of timetrace
                Fourier=Fourier[0:math.floor(length/factor)]         #Round up so that it includes zero foruneven length
        
                self.Fourier=self.Fourier+np.abs(Fourier)**2
            
            self.Fourier=self.Fourier/length
    
        
            self.freq=np.fft.fftfreq(self.Fourier.size*factor)[0:math.floor(length/factor)]*(1/self.t_step/10**3)
        else:
            max_res=1/self.T_0
            print("The resolution is too good, max resolution is: %.0f" % max_res + " Hz")
    
    def tester(self,normalize=1,numbertest=1):
        self.normalize=normalize
        self.FourierTransform1(numbertest)
        
        plt.plot(self.freq,self.Fourier,ls='-',linewidth=8,alpha=0.6, color=self.color, label=self.name)  #Plot fourier transform
        Fouriermax=np.argmax(self.Fourier)                      #Find peak
        plt.subplots_adjust(right=0.75)
        #Plot peak and write out the x-component of the peak
        # plt.plot(freq[Fouriermax],Fourier2[Fouriermax],color='red',marker='.',figure=fig, clip_box=matplotlib.transforms.Bbox([[0,0],[0.5,0.5]]))
        print('Peaket er ved: '+str(self.freq[Fouriermax])+' kHz')
        plt.xlabel('kHz',fontsize=32)                   #Lable axis
        plt.ylabel('PSD',fontsize=32)
        # plt.legend(('Fourier transform','peak'),fontsize=10)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        #plt.tick_params(axis='y',size=24)
        plt.legend()
        plt.yscale('log')
        
    def FourierTransform1(self,numbertest):
        factor=2
        self.timetrace=self.timetrace/self.normalize
        self.Fourier=np.zeros(math.floor(len(self.timetrace[0])/factor))
        n=numbertest

        Fourier=np.fft.fft(self.timetrace[n])  #Get fourier transform of timetrace

        length=len(Fourier)             #Find the length of the array
        Half=math.floor(length/factor)  #Find the midpoint so we only plot positive values                                
        Fourier=Fourier[0:Half]         #Round up so that it includes zero foruneven length

        self.Fourier=self.Fourier+np.abs(Fourier)**2
    
        self.Fourier=self.Fourier/len(self.timetrace)

    
        self.freq=np.fft.fftfreq(self.Fourier.size*factor)[0:Half]*(1/self.t_step/10**6)
        
        
    def tester2(self):
        
        plt.plot(self.magnetic_x, self.magnetic_y/(2*np.pi),color=self.color, label=self.name)
        plt.legend()
        
        
    def Timetrace(self):
        # plt.plot(self.timetrace[0])
        return self.timetrace
        
    def MORSavg(self):
        k=1
        timetraceAverage = self.timetrace[0:self.MORSSteps]
        self.MORSReset = self.MORSSteps
        while self.MORSReset < int(self.rounds/100)*100: 
            timetraceAverage += self.timetrace[self.MORSReset:self.MORSReset+self.MORSSteps]
            k += 1
            self.MORSReset += self.MORSSteps
        timetraceAverage = timetraceAverage/k
        
        return timetraceAverage
        
    def getself(self):
        return self.z_cell
    
    def MORSfitting(self):
        avgtimetrace = self.MORSavg()
        N=500
        omega=np.linspace(min(self.magnetic_y),max(self.magnetic_y),N)
        demod = np.zeros(N)
        t = np.arange(len(avgtimetrace))*self.t_step
        k=0
        for i in omega:
            demod[k] = np.sum(-avgtimetrace*np.cos(i*(t+self.t_step)))
            k += 1
        omega = omega/(2*np.pi)
        plt.plot(omega*1E-3,demod)
        
        def Lorentzian(omega,omega0,gamma,A,B):
            return A*((gamma/2)**2 / ((omega-omega0)**2+(gamma/2)**2))+B
        
        f_model = omega/1E3
        psd_model = demod
        A=np.where(demod>np.max(demod)/2)[0]
        def lin(omega,demod,a,b):
            cof = (demod[b]-demod[a])/(omega[b]-omega[a])
            return (np.max(demod)/2-demod[a])/cof+omega[a]
        
        A2 = lin(omega,demod,A[-1]+1,A[-1])-lin(omega,demod,A[0]-1,A[0])
        
        # A=omega[np.where(demod>np.max(demod)/2)][-1]-omega[np.where(demod>np.max(demod)/2)][0]
        print(A2)
        minimum=np.min(psd_model)
        psd_model=psd_model
        omega0=f_model[np.argmax(psd_model)]; gamma=8/1E3; A=np.max(psd_model); B=minimum
        model = Model(Lorentzian)
        model.set_param_hint('B', value = B, min=0)
        model.set_param_hint('A', value = A, min=0)
        model.set_param_hint('omega0', value = omega0, min=omega0-10,max=omega0+10)
        model.set_param_hint('gamma', value = gamma, min=0)
        
        weights=psd_model
        result = model.fit(psd_model, omega=f_model)
        # plt.plot(f_model,result.init_fit,'-')
        value = result.values["gamma"]*1E3
        # print(result.values["B"]*minimum)
        # print(value)
        text = "FWFM = %0.1f Hz" % value
        plt.plot(f_model,result.best_fit,'-',label=text)
        plt.xlabel('Frequency [kHz]')
        plt.xlabel('Signal [AU]')
        plt.legend()
        
        print(result.fit_report())
    
    def save(self,folder=""):
        filename = folder + "/" + self.filename + ".png"
        plt.savefig(filename)
    
    
    
    