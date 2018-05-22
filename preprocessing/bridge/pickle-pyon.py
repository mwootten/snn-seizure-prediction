import ast
import pickle
import sys

pickle_file = sys.argv[1]
pyon_file = sys.argv[2]

ifh = open(pickle_file, 'rb')
val = pickle.load(ifh)
ifh.close()

ofh = open(pyon_file, 'w')
ofh.write(str(val))
ofh.close()
