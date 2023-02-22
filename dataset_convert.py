import os
import argparse
import numpy as np
import pathlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type = str, required = True)
    parser.add_argument("-id", type = int, required=True) # sample id
    parser.add_argument("-mr", type = float, required=True) # missing rate
    parser.add_argument("-size", type = int, required=True) # sample size
    parser.add_argument("-func", type=str, required=True) # choose function from load and convert
    return parser.parse_args()

def imputed_dataset_convert(dataset, indx, mr, size):
    """
    Convert .tsv data into separate imputed datasets
    indx: the index of current sample
    """
    complete_path = os.path.join('train_test_split','{}_groundtruth.tsv'.format(dataset))
    result_path = os.path.join('imputations','{}_imputed.tsv'.format(dataset))

    complete_data = np.loadtxt(complete_path, delimiter='\t')
    result = np.loadtxt(result_path, delimiter='\t')
    result = result.reshape(size, -1, complete_data.shape[1])
    k = result.shape[1]

    
    pathlib.Path('../../MissingData_DL/results/{}/MCAR_{}_{}/vaeac'.format(dataset,mr, size)).mkdir(parents=True, exist_ok=True)
    for id in range(k):
        current_imputed_dataset = result[:,id,:]
        current_imputed_savepath = os.path.join('../../MissingData_DL/results/{}/MCAR_{}_{}/vaeac/imputed_{}_{}.csv'.format(dataset, mr, size, indx, id))
        np.savetxt(current_imputed_savepath, current_imputed_dataset, delimiter=',')

def missing_dataset_convert(dataset, indx, mr, size):
    """
    Convert .csv data into .tsv data for training
    """
    complete_path = os.path.join('../../MissingData_DL/samples/{}/complete_{}_{}/sample_{}.csv'.format(dataset,mr,size,indx))
    input_path = os.path.join('../../MissingData_DL/samples/{}/MCAR_{}_{}/sample_{}.csv'.format(dataset,mr,size,indx)) 
    complete_data = np.loadtxt(complete_path, delimiter=',')
    input_data = np.loadtxt(input_path, delimiter=',')

    groundtruth_path = os.path.join('train_test_split','{}_groundtruth.tsv'.format(dataset))
    train_path = os.path.join('train_test_split','{}_train.tsv'.format(dataset))

    np.savetxt(groundtruth_path, complete_data, delimiter='\t')
    np.savetxt(train_path, input_data, delimiter='\t')

    

if __name__ == '__main__':
    args = parse_args()
    if args.func == 'convert':
        imputed_dataset_convert(args.dataset, args.id, args.mr, args.size)

    elif args.func == 'load':
        missing_dataset_convert(args.dataset, args.id, args.mr, args.size)
    else:
        print("WRONG PARAMETER")