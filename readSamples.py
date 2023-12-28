import pickle
from glob import glob

for filename in glob('/home/pi/Data/sample_dmp_*.pkl'):
    with open(filename, 'rb') as f:
        samples = pickle.load(f)
        for sample in samples:
            print(sample)
    print()
