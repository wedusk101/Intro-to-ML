#imports a csv file 
import tensorflow as tf
import pandas as pd
import sys

def load_data(file):
	data = pd.read_csv(file)
	return data
	
def data_dict(file):
	data = pd.read_csv(file)
	
def feature_dict(file):
	featurefile=pd.read_csv(file,header=1,usecols=range(4))
	return featurefile

def label_dict(file):
	labelfile=pd.read_csv(file,header=1,usecols=[4])
	return labelfile

def train_input_fn(features,labels,batch_size):
	dataset=tf.data.Dataset.from_tensor_slices((dict(features),labels))
	
    #dataset = dataset.shuffle(1000).repeat().batch(batch_size)
	
	return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset	
	

def main():
	# parser = argparse.ArgumentParser()
	# args = parser.parse_args()
	file = sys.argv[1]
	file2 = sys.argv[2]
	data = load_data(file)
	train_x= feature_dict(file)
	train_y=label_dict(file)
	test_x=feature_dict(file2)
	test_y=label_dict(file2)
	print (data)
	print (train_x,train_y)
	print (test_x,test_y)
	dataset=train_input_fn(train_x,train_y,0)
	print (dataset)
	
if __name__ == "__main__":
    main()