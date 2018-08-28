#imports a csv file 
import pandas as pd
import sys

def load_data(file):
	data = pd.read_csv(file)
	return data
	
def data_dict(file):
	data = pd.read_csv(file)
	
	

def main():
	# parser = argparse.ArgumentParser()
	# args = parser.parse_args()
	file = sys.argv[1]
	data = load_data(file)
	print (data)
	
if __name__ == "__main__":
    main()
