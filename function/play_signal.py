import simpleaudio as sa
import numpy as np



def play_signal(signal,sample_rate):

    # Normalize the signal to the range from -1 to 1
    signal = np.int16(signal * 32767)
    
    #play signal
    play_obj = sa.play_buffer(signal, 1, 2, sample_rate)

    #wait to finish
    play_obj.wait_done()


    

