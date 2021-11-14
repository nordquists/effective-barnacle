import sys
import math

def generate_bins(nums, num_bins):
    bin_counts = [0 for bin in range(num_bins)]
    for i in range(1, len(nums)):
        bin_number = nums[i] // (20 / num_bins)
        print(int(bin_number))
        print(nums[i])
        bin_counts[int(bin_number)] += 1
    
    return bin_counts

def compare(num_file, num_bins):
    nums = []
    with open(num_file, 'r') as f:
        for line in f:
            nums.extend([float(item) for item in line.split()])
    results = generate_bins(nums, num_bins)
    
    for i, b in enumerate(results):
        print(b)
        # print(f"bin[{i}] = {b}")

if __name__ == '__main__':
    if compare(sys.argv[1], int(sys.argv[2])):
        print("CORRECT BINS GENERATED")
    else:
        print("!!!INCORRECT BINS GENERATED!!!")