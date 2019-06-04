import numpy as np
import matplotlib.pyplot as plt
import pynlo
from pynlo.media.fibers import fiber

T0_ps    = 0.0284  # pulse duration (ps)
pulseWL = 835   # pulse central wavelength (nm)
dz=1e-3
tau_s=0.00056

GDD     = 0.0    # Group delay dispersion (ps^2)
TOD     = 0.0    # Third order dispersion (ps^3)
pump_power = 1.0e4 # Peak power

Window  = 10   # simulation window (ps)
Steps   = 100 # simulation steps
Points  = 2**13  # simulation points
error   = 0.001



betas=np.array([-11.830e-3, 8.1038e-5, -9.5205e-8, 2.0737e-10, -5.3943e-13, 1.3486e-15, -2.5495e-18, 3.0524e-21,-1.7140e-24])*1e3
# beta2   = 23     # (ps^2/km)
# beta3   = 0.024     # (ps^3/km)
# beta4   = 0.000    # (ps^4/km)
        
Length  = 0.150    # length in mm
    
Alpha   = 0.0     # attentuation coefficient (dB/cm)
Gamma   = 110 # Gamma (1/(W km) 
    
fibWL   = pulseWL # Center WL of fiber (nm)
    
Raman   = True    # Enable Raman effect?
Steep   = True    # Enable self steepening?

alpha = np.log((10**(Alpha * 0.1))) * 100  # convert from dB/cm to 1/m




######## This is where the PyNLO magic happens! ############################

# create the pulse!
pulse = pynlo.light.DerivedPulses.SechPulse(power = pump_power, # Power will be scaled by set_epp
                                            T0_ps                   =T0_ps, 
                                            center_wavelength_nm    = pulseWL, 
                                            time_window_ps          = Window,
                                            GDD=GDD, TOD=TOD, 
                                            NPTS            = Points, 
                                            frep_MHz        = 100, 
                                            power_is_avg    = False)

# create the fiber!
# fiber1 = fiber.FiberInstance() 
# fiber1.load_from_db(Length, 'dudley')

fiber1 = pynlo.media.fibers.fiber.FiberInstance()
fiber1.generate_fiber(Length, center_wl_nm=fibWL, betas=betas,\
                              gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)
                                
# Propagation
evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=error, USE_SIMPLE_RAMAN=False,dz = dz,
                 disable_Raman              = np.logical_not(Raman), 
                 disable_self_steepening    = np.logical_not(Steep),
                tau_s=tau_s,
                suppress_iteration = True)

y, AW, AT, pulse_out = evol.propagate(pulse_in=pulse, fiber=fiber1, n_steps=Steps)

########## That's it! Physics complete. Just plotting commands from here! ################
    
wl = pulse.wl_nm

loWL = 400
hiWL = 1350

loT=-1
hiT=5

iis = np.logical_and(wl>loWL,wl<hiWL)


iisT = np.logical_and(pulse.T_ps>loT,pulse.T_ps<hiT)

xW = wl[iis]
xT = pulse.T_ps[iisT]
zW_in = np.transpose(AW)[:,iis]
zT_in = np.transpose(AT)[:,iisT]

zW = 10*np.log10(np.abs(zW_in)**2)
zT = 10*np.log10(np.abs(zT_in)**2)
mlIW = np.max(zW)
mlIT = np.max(zT)

F = pulse.F_THz     # Frequency grid of pulse (THz)



def dB(num):
    return 10 * np.log10(np.abs(num)**2)
    

y_mm = y * 1e3 # convert distance to mm
# set up plots for the results:
fig = plt.figure(figsize=(10,10))
ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)      
ax0.plot(pulse_out.wl_nm,    dB(pulse_out.AW),  color = 'r')
#ax0.plot(pulse_out.wl_nm,    np.abs(pulse_out.AW)**2/np.max(np.abs(pulse_out.AW)**2),  color = 'r')
ax1.plot(xT,     dB(zT_in[-1,:]),  color = 'r')

ax0.set_ylim( - 60,  0)
#ax0.set_ylim( 0,  1)
ax1.set_ylim( - 40, 40)

ax0.plot(pulse.wl_nm,    dB(pulse.AW),  color = 'b')
#ax0.plot(pulse.wl_nm,    np.abs(pulse.AW)**2/max(np.abs(pulse.AW)**2),  color = 'b')
ax1.plot(xT,     dB(pulse.AT[iisT]),  color = 'b')

ax2.pcolormesh(xW, y_mm, zW,vmin = mlIW - 40.0, vmax = mlIW)
# plt.autoscale(tight=True)
ax2.set_xlim([loWL, hiWL])
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Distance (mm)')


extent = (loT, hiT, np.min(y_mm), Length)
ax3.imshow(zT, extent=extent,vmin=np.max(zT) - 60.0, vmax=np.max(zT), 
           aspect='auto', origin='lower')
ax3.set_xlabel('Delay (ps)')
ax3.set_ylabel('Distance (mm)')

fig, axs = plt.subplots(1,2,figsize=(10,5))

time_steps=1000
for ax, gate_type in zip(axs,('xfrog', 'frog')):
    DELAYS, FREQS, extent, spectrogram = pulse_out.spectrogram(gate_type=gate_type, gate_function_width_ps=0.05, time_steps=time_steps)
    #ax.imshow(spectrogram, aspect='auto', extent=extent,origin='upper')
    tpt=np.logical_and(DELAYS>loT,DELAYS<hiT)
    a_t=int(DELAYS[tpt].shape[0]/pulse_out.NPTS)
    _,WL=np.meshgrid(DELAYS[1],wl)
    wpt=np.logical_and(WL>loWL,WL<hiWL)
    a_w=int(WL[wpt].shape[0]/time_steps)
    poi=np.logical_and(tpt,wpt)
    maIM=np.max(10*np.log10(np.abs(spectrogram)**2))
    if gate_type== 'xfrog':
        ax.pcolormesh(DELAYS[poi].reshape(a_w,a_t),WL[poi].reshape(a_w,a_t),\
                      dB(spectrogram[poi]).reshape(a_w,a_t),vmin=maIM-40,vmax=maIM,cmap='afmhot')
    else:
        ax.pcolormesh(DELAYS[iis,:],WL[iis,:],\
                  dB(spectrogram[iis,:]),vmin=maIM-50,vmax=maIM,cmap='afmhot')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Wavelength (nm)')
    ax.set_title(gate_type)

plt.show()