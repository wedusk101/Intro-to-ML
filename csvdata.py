#imports a csv file 
import pandas as pd
import argparse
import sys

def load_data(file):
	data = pd.read_csv(file)
	return data
	
def data_dict(file):
	data = pd.read_csv(file)
	out = dict(data)
	return out	

def main():
	# parser = argparse.ArgumentParser()
	# args = parser.parse_args()
	file = sys.argv[1]
	data = load_data(file)
	result = data_dict(file)
	print (data)
	# print (result)
	
if __name__ == "__main__":
    main()
