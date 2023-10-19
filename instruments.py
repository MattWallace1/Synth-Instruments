import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

def exp_env(N, sr, mu=3):
    """
    Make an exponential envelope
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    mu: float
        Exponential decay rate: e^{-mu*t}

    Returns
    -------
    ndarray(N): Envelope samples
    """
    return np.exp(-mu*np.arange(N)/sr)

def drum_like_env(N, sr):
    """
    Make a drum-like envelope, according to Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
    t = np.arange(N)/sr
    t1 = t+0.04
    env = (t1)**2*np.exp(-30*t1)
    return env

def wood_drum_env(N, sr):
    """
    Make the wood-drum envelope from Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
    x1 = np.linspace(1, 0, int(sr*0.025))
    x2 = np.linspace(0, 0, int(N-sr*0.025))
    y = np.array([0])
    y = np.concatenate((y, x1, x2))

    
    return y

def brass_env(N, sr):
    """
    Make the brass ADSR envelope from Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    
    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
    
    y = np.array([0])
    
    '''
    attack = np.linspace(0, 1, int(0.1*sr))
    decay = np.linspace(1, 0.8, int(0.1*sr))
    sustain = np.linspace(0.8, 0.7, int(0.7*sr))
    release = np.linspace(0.7, 0, int(0.1*sr))
    '''
    duration = N/sr
    
    if duration > 0.3:
        
        attack = np.linspace(0, 1, int(0.1*sr))
        decay = np.linspace(1, 0.8, int(0.1*sr))
        sustain = np.linspace(0.8, 0.7, int((duration - 0.3)*sr))
        release = np.linspace(0.7, 0, int(0.1*sr))
        y = np.concatenate((y, attack, decay, sustain, release))
    elif duration > 0.2:
        attack = np.linspace(0, 1, int(0.1*sr))
        decay = np.linspace(1, 0.8, int(0.1*sr))
        release = np.linspace(0.8, 0, int((duration - 0.2)*sr))
        y = np.concatenate((y, attack, decay, release))
    elif duration > 0.1:
        attack = np.linspace(0, 1, int(0.1*sr))
        decay = np.linspace(1, 0.8, int((duration - 0.1)*sr))
        y = np.concatenate((y, attack, decay))
    else:
        attack = np.linspace(0, 1, int(duration*sr))
        y = np.concatenate((y, attack))
    
    return y


def dirty_bass_env(N, sr):
    """
    Make the "dirty bass" envelope from Attack Magazine
    https://www.attackmagazine.com/technique/tutorials/dirty-fm-bass/
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    
    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
    
    t = np.arange(N)/sr
    y = np.array([])
    env1 = np.exp(-30 * t)
    env2 = -(env1-1)
    y = np.concatenate((y, env1[0:int(N/2)]))
    y = np.concatenate((y, env2[0:int(N/2)]))

    return y

def fm_plucked_string_note(sr, note, duration, mu=3):
    """
    Make a plucked string of a particular length
    using FM synthesis
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    mu: float
        The decay rate of the note
    
    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: exp_env(N, sr, mu)
    return fm_synth_note(sr, note, duration,
                ratio = 1, I = 8, envelope = envelope,
                amplitude = envelope)

def fm_electric_guitar_note(sr, note, duration, mu=3):
    """
    Make an electric guitar string of a particular length by
    passing along the parameters to fm_plucked_string note
    and then turning the samples into a square wave

    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    mu: float
        The decay rate of the note
    
    Return
    ------
    ndarray(N): Audio samples for this note
    """
    y = np.sign(fm_plucked_string_note(sr, note, duration))
    return y

def fm_brass_note(sr, note, duration):
    """
    Make a brass note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    
    Return
    ------
    ndarray(N): Audio samples for this note
    """
    ## TODO: Fill this in
    envelope = lambda N, sr: brass_env(N, sr)
    return fm_synth_note(sr, note, duration,
                        ratio = 1, I = 10, envelope = envelope,
                        amplitude = envelope)


def fm_bell_note(sr, note, duration, mu = 0.8):
    """
    Make a bell note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    
    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: exp_env(N, sr, mu)
    return fm_synth_note(sr, note, duration,
                ratio = 1.4, I = 2, envelope = envelope,
                amplitude = envelope)


def fm_drum_sound(sr, note, duration, fixed_note = -14):
    """
    Make what Chowning calls a "drum-like sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    ------
    ndarray(N): Audio samples for this drum hit
    """
    envelope = lambda N, sr: drum_like_env(N, sr)
    return fm_synth_note(sr, note, duration,
                        ratio = 1.4, I = 2, envelope = envelope,
                        amplitude = envelope)

def fm_wood_drum_sound(sr, note, duration, fixed_note=-14):
    """
    Make what Chowning calls a "wood drum sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    envelope = lambda N, sr: wood_drum_env(N, sr)
    return fm_synth_note(sr, fixed_note, duration,
                        ratio = 1.4, I = 10, envelope = envelope,
                        amplitude = envelope)

def snare_drum_sound(sr, note, duration):
    """
    Make a snare drum sound by shaping noise
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    x = np.random.rand(int(sr*duration))
    t = np.arange(int(sr*duration))/sr
    
    env = x*np.exp(-20*t)
    
    return env

def fm_dirty_bass_note(sr, note, duration):
    """
    Make a "dirty bass" note, based on 
    https://www.attackmagazine.com/technique/tutorials/dirty-fm-bass/
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    envelope = lambda N, sr: dirty_bass_env(N, sr)
    return fm_synth_note(sr, note, duration,
                        ratio = 1, I = 18, envelope = envelope,
                        amplitude = envelope)

def make_tune(filename, sixteenth_len, sr, note_fn):
    """
    Parameters
    ----------
    filename: string
        Path to file containing the tune.  Consists of
        rows of <note number> <note duration>, where
        the note number 0 is a 440hz concert A, and the
        note duration is in factors of 16th notes
    sixteenth_len: float
        Length of a sixteenth note, in seconds
    sr: int
        Sample rate
    note_fn: function (sr, note, duration) -> ndarray(M)
        A function that generates audio samples for a particular
        note at a given sample rate and duration
    
    Returns
    -------
    ndarray(N): Audio containing the tune
    """
    tune = np.loadtxt(filename)
    notes = tune[:, 0]
    durations = sixteenth_len*tune[:, 1]
    
    # loop through all notes in the tune
    # make one note at a time w calls to note_fn
    # concatenate the samples from note_fn onto the end of a growing array
    y = np.array([0])
    for i in range(len(notes)):
        if np.isnan(notes[i]):
            x = np.zeros(int(durations[i]*sr))
        else:
            x = note_fn(sr, notes[i], durations[i])
        y = np.concatenate((y, x))
        
    return y


def karplus_strong_note(sr, note, duration, decay):
    """
    Parameters
    ----------
    sr: int 
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    decay: float 
        Decay amount (between 0 and 1)

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    N = int(duration*sr)
    y = np.zeros(N)    
    f = get_note_freq(note)
    T = int(sr / f)
    y[0:T] = np.random.rand(T)
    
    for i in range(T, N):
        y[i] = decay * (y[i-T] + y[i-T+1])/2
    
    return y

def get_note_freq(note):
    return 440*2**(note/12)


def fm_synth_note(sr, note, duration, ratio=2, I=2, 
                  envelope = lambda N, sr: np.ones(N),
                  amplitude = lambda N, sr: np.ones(N)):
    """
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    ratio: float
        Ratio of modulation frequency to carrier frequency
    I: float
        Modulation index (ratio of peak frequency deviation to
        modulation frequency)
    envelope: function (N, sr) -> ndarray(N)
        A function for generating an ADSR profile
    amplitude: function (N, sr) -> ndarray(N)
        A function for generating a time-varying amplitude

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    N = int(duration*sr)
    y = np.zeros(N)
    t = np.arange(N)/sr
    f = get_note_freq(note)
    A = amplitude(N, sr)
    modIndex = envelope(N, sr) * I 
    y = A * np.cos(2*np.pi*f*t + modIndex * np.sin(2*np.pi*f*ratio*t))  
    return y