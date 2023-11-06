from neuron import h, gui

class HH_NEURON:
    def __init__(self):
        self.create_sections()
        #self.define_geometry()
        self.define_biophysics()
        self.create_stimulator()
       
    def create_sections(self):
        self.soma = h.Section(name = 'soma')
        self.dend = h.Section(name='dend')
        self.all = [self.soma, self.dend]
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157
        self.dend.L = 200
        self.dend.diam = 1
        
    def define_biophysics(self):
        for sec in self.all: 
                sec.Ra  = 100 #axial resistance Ohm * cm
                sec.cm = 1 #membrane capacitance micro farads/cm^2
        
        #active currents
        self.soma.insert('hh') #applying hodgkin huxley biophysical constrains
        for seg in self.soma: 
            
            #S/cm^2
            seg.hh.gnabar = 0.12 #na conductance
            seg.hh.gkbar = 0.036 #k conductance
            seg.hh.gl = 0.003 #leak conductance
            
            seg.hh.el = -54.3 #reversal potential, mV
        
        #passive currrents
        self.dend.insert('pas')
        for seg in self.dend:
            seg.pas.g = 0.001
            seg.pas.e = -65
            
    def create_sim(self): 
        self.stim = h.IClamp(self.soma(0.5)) #injecting current at center of soma
        self.stim.delay = 10 #time delay before stimulus
        self.stim.dur = 100 #duration of stimulus TUNE THIS
        self.stim.amp = 0.1 #amplitude TUNE THIS