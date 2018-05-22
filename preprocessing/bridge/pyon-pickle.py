import ast
import pickle
import sys

pyon_file = sys.argv[1]
pickle_file = sys.argv[2]

ifh = open(pyon_file, 'r')
val = ast.literal_eval(ifh.read())
ifh.close()

ofh = open(pickle_file, 'wb')
pickle.dump(val, ofh)
ofh.close()
