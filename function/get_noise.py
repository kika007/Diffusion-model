import torch
 


def get_noise(signal,SNR):

    #RMS value of signal
    RMS_s=torch.sqrt(torch.mean(signal**2))
    #RMS values of noise
    RMS_n=torch.sqrt(RMS_s**2/(pow(10,SNR/10)))
    #Additive white gausian noise. Thereore mean=0
    STD_n=RMS_n
    #because mean=0 STD=RMS
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    noise = torch.normal(0, STD_n, (signal.shape[1],))

    return noise

    


