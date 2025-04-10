import numpy as np
import astropy.units as u
from scipy.optimize import curve_fit
#from ipywidgets import interactive, IntSlider , Layout
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
#from IPython.display import clear_output


plt.rcParams['figure.max_open_warning'] = 0





def dispersion_delay(fstart, fstop, dms = None):
    """
    Returns DM-delay in seconds
    """
    return (
        4148808.0
        * dms
        * (1 / fstart ** 2 - 1 / fstop ** 2)
        / 1000
    )


def dedisperse(wfall, DM, freq, dt, ref_freq="bottom"):

    """
    Dedisperse a waterfaller matrix to DM.
    """

    k_DM = 1. / 2.41e-4
    dedisp = np.zeros_like(wfall)

    # pick reference frequency for dedispersion
    if ref_freq == "top":
        reference_frequency = freq[-1]
    elif ref_freq == "center":
        center_idx = len(freq) // 2
        reference_frequency = freq[center_idx]
    elif ref_freq == "bottom":
        reference_frequency = freq[0]
    else:
        #print "`ref_freq` not recognized, using 'top'"
        reference_frequency = freq[-1]

    shift = (k_DM * DM * (reference_frequency**-2 - freq**-2) / dt).round().astype(int)
    for i,ts in enumerate(wfall):
        dedisp[i] = np.roll(ts, shift[i])


    return dedisp

def gauss(x,a,x0,sigma):

    """
    Simple Gaussian Function.
    """

    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def get_width(time,timeseries):

    """
    Function to get the width of a burst from a Gaussian fit.
    """


    tmax = time[np.argmax(timeseries)]
    par_time_opt,par_time_cov = curve_fit(gauss,time,timeseries, p0=[np.max(timeseries),tmax,0.1 * np.max(time)])

    sigma_t = np.abs(par_time_opt[2])

    W = 2.355 * sigma_t

    return W

def test_FRB(
    nchan = 100,
    nbin = 10000,
    fc = 600 * u.MHz,
    bw = 400 * u.MHz,
    obsduration = 120 * u.s,
    ):
    df = bw / nchan
    dt = obsduration / nbin

    data = np.zeros((nchan, nbin) , dtype = float)

    freqs = np.linspace(fc.value + bw.value / 2 , fc.value - bw.value / 2, nchan)
    times = np.linspace(0, obsduration.value , nbin)

    dm = 349
    delaytot = np.abs(dispersion_delay(freqs[-1], freqs[0], dms = dm))
    tburst = 35.5 * u.s
    fburst = fc * u.MHz
    width_t = np.random.uniform(1,50) * u.ms
    width_f = np.random.uniform(50, bw.value) * u.MHz
    sigma_f = width_f.value / 2.355
    sigma_t = width_t.to(u.s).value / 2.355
    If = gauss(freqs, 1 , fburst.value, sigma_f)
    A = 2


    for chan, f in enumerate(freqs):

        delay = dispersion_delay(f, freqs[0], dms = dm)
        data[chan,:] = A*If[chan]*gauss(times, 1, (tburst.value + delay ) , sigma_t) + np.random.normal(0,0.1, size = data.shape[1])

    rfi_idxs = np.array([5,12,28,32,34])

    for rfi_idx in rfi_idxs:

        data[rfi_idx, :]  = 10 * A

    return tburst,dm,data



def simulate_FRB(
    nchan = 100,
    nbin = 10000,
    fc = 600 * u.MHz,
    bw = 400 * u.MHz,
    obsduration = 120 * u.s,
    ):
    df = bw / nchan
    dt = obsduration / nbin

    data = np.zeros((nchan, nbin) , dtype = float)

    freqs = np.linspace(fc.value + bw.value / 2 , fc.value - bw.value / 2, nchan)
    times = np.linspace(0, obsduration.value , nbin)

    dm = int(np.random.uniform(100,1000))#20 * np.random.noncentral_chisquare(1, 10)
    delaytot = np.abs(dispersion_delay(freqs[-1], freqs[0], dms = dm))
    tburst = np.random.uniform(5, obsduration.value-delaytot) * u.s
    fburst = np.random.uniform(freqs[-1],freqs[0]) * u.MHz
    width_t = np.random.uniform(1,50) * u.ms
    width_f = np.random.uniform(50, bw.value) * u.MHz
    sigma_f = width_f.value / 2.355
    sigma_t = width_t.to(u.s).value / 2.355
    If = gauss(freqs, 1 , fburst.value, sigma_f)
    A = np.random.uniform(0.1,10)


    for chan, f in enumerate(freqs):

        delay = dispersion_delay(f, freqs[0], dms = dm)
        data[chan,:] = A*If[chan]*gauss(times, 1, (tburst.value + delay ) , sigma_t) + np.random.normal(0,0.1, size = data.shape[1])

    rfi_idxs = np.random.choice(np.arange(0, data.shape[0]), np.random.randint(0,10), replace=False)

    for rfi_idx in rfi_idxs:

        data[rfi_idx, :]  = np.random.uniform(1,20) * A


    return tburst,dm,data

class FRBFILE:

    """
    Create a FRBFILE Object

    data : Basic Total Intensity data, a numpy array
    telescope : The telescope  which "recorded" the data
    duration : length of the observation
    fc : central frequency of the observation
    bw : bandwidth of the observation
    df : spectral resolution of the data
    dt : time resolution of the data
    """

    def __init__(
    self,
    data,
    telescope,
    duration,
    fc,
    bw,
    tburst,
    dm,
    df,
    dt,
    ):

        self.telescope = telescope
        self.fc = fc
        self.bw = bw
        self.dm = dm
        self.df = df
        self.dt = dt
        self.data = data
        self.duration = duration
        self.tburst = tburst


    def get_dm(self):

        return self.dm.value

    def header(self):

        string =  f"""


 Observational Specifics \n

 Telescope                            = {self.telescope}
 Central Frequency                    = {self.fc.value} {self.fc.unit}
 Bandwidth                            = {self.bw.value} {self.bw.unit}
 \n
 Data Information \n

 Data length                          = {self.duration.value} {self.duration.unit}
 Time resolution                      = {self.dt.value} {self.dt.unit}
 Frequency resolution                 = {self.df.value} {self.df.unit}
 Data Shape (freq chans, times bins)  = {self.data.shape}

        """

        return string

    def reveal(self):
        DM = self.dm * u.pc / u.cm**3

        string =  f"""


 The FRB Hidden in this File has: \n

 DM = {DM.value} {DM.unit}
 Burst Time  = {self.tburst.value} {self.tburst.unit}

        """

        return string

    def get_data(self):

      return self.data

    def get_lightcurve(self):

      return self.data.mean(0)

    def find_peak(self, start = 0, stop = 100):

      data = self.data
      duration = self.duration.value
      dt = self.dt.value

      lightcurve = self.get_lightcurve()
      times = np.linspace(0, duration, lightcurve.shape[0])
      startbin = int(start / dt)
      stopbin  = int(stop / dt)
      lightcurve = lightcurve[startbin:stopbin]
      times = times[startbin:stopbin]
      apeak = np.argmax(lightcurve)
      peak  = times[apeak]

      print(f"The Peak is at {peak:.3f} s")


    def plot_observation(self, start = 0, stop = 100):

      data = self.data
      fc = self.fc.value
      bw = self.bw.value
      #duration = self.duration.value
      dt = self.dt.value

      plt.figure(figsize = (40,10))
      plt.rc('axes', linewidth = 1)

      widths  = [1]
      heights = [0.3,0.7]

      gs = plt.GridSpec(2,1, hspace = 0.0 , wspace = 0.0,  width_ratios = widths, height_ratios = heights, top = 0.99 , bottom = 0.2, right = 0.9, left = 0.2)

      ax0 = plt.subplot(gs[0,0])
      ax1 = plt.subplot(gs[1,0])

      ax0.set_xticks([])


      size = 20
      ax0.margins(x = 0)
      ax1.margins(x = 0)
      ax0.tick_params(labelsize  = size, length = 5, width = 1)
      ax1.tick_params(labelsize  = size, length = 5, width = 1)

      ax0.set_ylabel(r"Flux (Arbitrary)", size = size)
      ax1.set_xlabel(r"Time (s)", size = size)
      ax1.set_ylabel(r"Frequency (MHz)" , size = size)

      freqs = np.linspace(fc + bw / 2, fc - bw / 2, data.shape[0])

      startbin = int(start / dt)

      stopbin  = int(stop / dt)

      dataplot = data[:, startbin : stopbin]

      times = np.linspace(start, stop, dataplot.shape[1])

      lightcurve = dataplot.mean(0)

      ax0.plot(times, lightcurve, color = "darkblue", linewidth = 2)

      ax1.imshow(dataplot , aspect = "auto", extent = (times[0], times[-1], freqs[-1], freqs[0]))

      ax1.locator_params(axis='x', nbins=25)

      plt.show()

def clean_file(frbfile,channels):

    channels = np.array(channels)

    data = frbfile.data
    fc = frbfile.fc
    bw = frbfile.bw
    dt = frbfile.dt
    df = frbfile.df
    telescope = frbfile.telescope
    tburst = frbfile.tburst
    duration = frbfile.duration
    dm = frbfile.dm

    data[channels,:] = np.nan

    cleanfile = FRBFILE(data,telescope,duration,fc,bw,tburst,dm,df,dt)

    return cleanfile

def dedisp_file(frbfile, dm = 0):

    data = frbfile.data
    fc = frbfile.fc
    bw = frbfile.bw
    #dm = frbfile.dm
    dt = frbfile.dt
    df = frbfile.df
    telescope = frbfile.telescope
    tburst = frbfile.tburst
    duration = frbfile.duration


    freqs = np.linspace(fc.value + bw.value /2 , fc.value - bw.value / 2, data.shape[0])

    dedispdata = dedisperse(data, dm, freqs, dt.value, ref_freq="bottom")

    data = dedispdata


    dedispfile = FRBFILE(data,telescope,duration,fc,bw,tburst,dm,df,dt)


    return dedispfile

def make_observation(
    telescope = "SRT (P-band)",
    fc = 336 * u.MHz,
    bw = 80 * u.MHz,
    df = 0.1 * u.MHz,
    dt = 0.01 * u.s,
    duration = 100 * u.s,
    ):

    nchan = int(bw.value / df.value)
    nbin = int(duration.value / dt.value)

    tburst, dm, data = simulate_FRB(nbin = nbin, nchan = nchan, fc = fc, bw = bw, obsduration = duration)

    r = np.random.uniform(0,1)

    if r >= 0.33:
       srtfile =  FRBFILE(data,telescope,duration,fc,bw,tburst,dm,df,dt)
    else:
       srtfile =  FRBFILE(np.random.normal(0,1,size = data.shape),telescope,duration,fc,bw,0*u.s,0,df,dt)

    return srtfile


def play_observation(frbfile):
    data = frbfile.data.copy()
    fc = frbfile.fc
    bw = frbfile.bw
    dt = frbfile.dt
    duration = frbfile.duration

    freqs = np.linspace(fc.value + bw.value / 2, fc.value - bw.value / 2, data.shape[0])
    t_range = 5  # seconds of data shown
    t_start = 0
    dm = 0

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [1, 3]})
    plt.subplots_adjust(bottom=0.35)
    ax0.margins(x=0)

    def update(val):
        t_start = time_slider.val
        dm = dm_slider.val
        startbin = int(t_start / dt.value)
        stopbin = int((t_start + t_range) / dt.value)

        dedisp = dedisperse(data, dm, freqs, dt.value)
        data_segment = dedisp[:, startbin:stopbin]
        time_segment = np.linspace(t_start, t_start + t_range, data_segment.shape[1])
        lightcurve = np.nanmean(data_segment, axis=0)

        line0.set_data(time_segment, lightcurve)
        line0.axes.set_xlim(time_segment[0], time_segment[-1])
        
        #ymin = np.nanmin(lightcurve)
        #ymax = np.nanmax(lightcurve) 
        #line0.axes.set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))

        #ax0.set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))
        im.set_data(data_segment)
        im.set_extent((t_start, t_start + t_range, freqs[-1], freqs[0]))
        #vmin = np.nanpercentile(data_segment, 2)
        #vmax = np.nanpercentile(data_segment, 98)
        #im.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    # Initial plot
    startbin = int(t_start / dt.value)
    stopbin = int((t_start + t_range) / dt.value)
    dedisp = dedisperse(data, dm, freqs, dt.value)
    data_segment = dedisp[:, startbin:stopbin]
    time_segment = np.linspace(t_start, t_start + t_range, data_segment.shape[1])
    lightcurve = np.nanmean(data_segment, axis=0)

    line0, = ax0.plot(time_segment, lightcurve)
    im = ax1.imshow(data_segment, aspect='auto', extent=(t_start, t_start + t_range, freqs[-1], freqs[0]))

    ax0.set_ylabel("Flux")
    ax1.set_ylabel("Freq (MHz)")
    ax1.set_xlabel("Time (s)")
    ax0.set_title("Let's check the data!")

    # Sliders
    ax_slider_t = plt.axes([0.25, 0.2, 0.5, 0.03])
    time_slider = Slider(ax_slider_t, 'Start Time', 0, duration.value - t_range, valinit=t_start, valstep=dt.value)

    ax_slider_dm = plt.axes([0.25, 0.1, 0.5, 0.03])
    dm_slider = Slider(ax_slider_dm, 'DM', 0, 1000, valinit=0, valstep=0.5)

    time_slider.on_changed(update)
    dm_slider.on_changed(update)

    plt.show()

def rfi_zap(frbfile):
    data = frbfile.data.copy()
    
    fc = frbfile.fc
    bw = frbfile.bw
    dt = frbfile.dt
    df = frbfile.df
    duration = frbfile.duration
    telescope = frbfile.telescope
    tburst = frbfile.tburst
    dm = frbfile.dm

    freqs = np.linspace(fc.value + bw.value / 2, fc.value - bw.value / 2, data.shape[0])
    times = np.linspace(0, duration.value, data.shape[1])

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, aspect="auto", extent=(times[0], times[-1], freqs[-1], freqs[0]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("Click to zap channels")

    def on_click(event):
        if event.inaxes is not ax:
            return
        y = event.ydata
        chan = np.argmin(np.abs(freqs - y))
        print(f"Zapping channel {chan} (Freq ~ {freqs[chan]:.2f} MHz)")
        data[chan, :] = np.nan
        im.set_data(data)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    return FRBFILE(data, telescope, duration, fc, bw, tburst, dm, df, dt)

def make_test(
    telescope = "SRT (P-band)",
    fc = 336 * u.MHz,
    bw = 80 * u.MHz,
    df = 0.1 * u.MHz,
    dt = 0.01 * u.s,
    duration = 100 * u.s,
    ):

    nchan = int(bw.value / df.value)
    nbin = int(duration.value / dt.value)

    tburst, dm, data = test_FRB(nbin = nbin, nchan = nchan, fc = fc, bw = bw, obsduration = duration)

    srtfile =  FRBFILE(data,telescope,duration,fc,bw,tburst,dm,df,dt)

    return srtfile


    data = frbfile.data.copy()
    fc = frbfile.fc
    bw = frbfile.bw
    dt = frbfile.dt
    duration = frbfile.duration
    dm = 0  # Start with zero

    freqs = np.linspace(fc.value + bw.value / 2, fc.value - bw.value / 2, data.shape[0])
    times = np.linspace(0, duration.value, data.shape[1])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 2]})
    plt.subplots_adjust(bottom=0.25)

    # Initial plot range
    t_start = 0
    t_range = 5
    t_stop = t_start + t_range

    startbin = int(t_start / dt.value)
    stopbin = int(t_stop / dt.value)

    dedisp = dedisperse(data, dm, freqs, dt.value)
    data_segment = dedisp[:, startbin:stopbin]
    time_segment = np.linspace(t_start, t_stop, data_segment.shape[1])

    lightcurve = np.nanmean(data_segment, axis=0)
    line0, = ax0.plot(time_segment, lightcurve)
    im = ax1.imshow(data_segment, aspect='auto', extent=(t_start, t_stop, freqs[-1], freqs[0]))

    ax0.set_ylabel("Flux")
    ax1.set_ylabel("Freq (MHz)")
    ax1.set_xlabel("Time (s)")
    ax0.set_title("Dedispersed FRB Observation")

    # Add slider for start time
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Start Time', 0, duration.value - t_range, valinit=t_start, valstep=dt.value)

    def update(val):
        nonlocal data_segment, time_segment
        t_start = slider.val
        t_stop = t_start + t_range
        startbin = int(t_start / dt.value)
        stopbin = int(t_stop / dt.value)
        data_segment = dedisperse(data, dm, freqs, dt.value)[:, startbin:stopbin]
        time_segment = np.linspace(t_start, t_stop, data_segment.shape[1])
        lightcurve = np.nanmean(data_segment, axis=0)
        line0.set_data(time_segment, lightcurve)
        line0.axes.set_xlim(time_segment[0], time_segment[-1])
        im.set_array(data_segment)
        im.set_extent((t_start, t_stop, freqs[-1], freqs[0]))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Handle clicking to zap frequency channels
    def on_click(event):
        if event.inaxes != ax1:
            return
        y_freq = event.ydata
        chan = np.argmin(np.abs(freqs - y_freq))
        print(f"Zapping channel {chan}")
        data[chan, :] = np.nan
        update(slider.val)

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()



if __name__ == '__main__':
    
    mysteryfile = make_test()
    play_observation(mysteryfile)
    mysterycleanfile = rfi_zap(mysteryfile)
    play_observation(mysterycleanfile)