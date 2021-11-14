import sys
import math

def generate_bins(nums, num_bins):
    bins = [0] * num_bins
    for num in nums:
        bins[math.floor(num_bins / 20 * num)] += 1
    
    return bins

def compare(num_file, num_bins):
    nums = []
    with open(num_file, 'r') as f:
        for line in f:
            nums.extend([int(item) for item in line.split()])
    results = generate_bins(nums, num_bins)
    
    for i, bin in enumerate(results):
        print(f'bin[{i}] = {bin}')

if __name__ == '__main__':
    if compare(sys.argv[1], int(sys.argv[3])):
        print("CORRECT BINS GENERATED")
    else:
        print("!!!INCORRECT BINS GENERATED!!!")