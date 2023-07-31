import numpy as np 
import os,sys


savefile = sys.argv[1]
base_dict = np.load(sys.argv[2],allow_pickle=True).item()
count=0
keys = base_dict.keys()
for afile in sys.argv[3:]:
    ad_dict = np.load(afile,allow_pickle=True).item()
    
    for akey in keys:
        base_dict[akey]+=ad_dict[akey]
    count+=1


for akey in keys:
    base_dict[akey]=base_dict[akey]/(count+1)

np.save(savefile,base_dict)


