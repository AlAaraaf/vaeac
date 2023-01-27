import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type = str, required = True)
    parser.add_argument("-id", type = int, required=True) # sample id
    parser.add_argument("-mr", type = float, required=True) # missing rate
    parser.add_argument("-size", type = int, required=True) # sample size
    return parser.parse_args()

def imputed_dataset_convert(dataset, indx, mr, size):
    """
    Convert .tsv data into separate imputed datasets
    indx: the index of current sample
    """
    complete_path = os.path.join('train_test_split','{}_groundtruth.tsv'.format(dataset))
    input_path = os.path.join('train_test_split','{}_train.tsv'.format(dataset))
    result_path = os.path.join('imputations','{}_imputed.tsv'.format(dataset))

    complete_data = np.loadtxt(complete_path, delimiter='\t')
    input = np.loadtxt(input_path, delimiter='\t')
    result = np.loadtxt(result_path, delimiter='\t')
    result = result.reshape(input.shape[0], -1, input.shape[1])
    k = result.shape[1]

    current_sample_savepath = os.path.join('../samples/{}/complete_{}_{}/sample_{}.csv'.format(dataset,mr, size,indx))
    current_input_savepath = os.path.join('../samples/{}/MCAR_{}_{}/sample_{}.csv'.format(dataset,mr, size, indx))
    np.savetxt(current_sample_savepath, complete_data, delimiter=',')
    np.savetxt(current_input_savepath, input, delimiter=',')
    for id in range(k):
        current_imputed_dataset = result[:,id,:]
        current_imputed_savepath = os.path.join('../results/{}/MCAR_{}_{}/imputed_{}_{}.csv'.format(dataset, mr, size, indx, id))
        np.savetxt(current_imputed_savepath, current_imputed_dataset, delimiter=',')

if __name__ == '__main__':
    args = parse_args()
    imputed_dataset_convert(args.dataset, args.id, args.mr, args.size)