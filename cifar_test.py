import argparse
import sys

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding = 'bytes')
	return dict

def main():
	# parser = argparse.ArgumentParser()
	# args = parser.parse_args()
	file = sys.argv[1]
	data = unpickle(file)
	print(data)
	
if __name__ == "__main__":
    main()
