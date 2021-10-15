import sys
def generate_divisors(N, x):
    results = []
    curr = 0

    while curr < N:
        curr += x
        results.append(str(curr))

    return results

def compare(output_file, N, x):
    results = generate_divisors(N, x)
    curr = 0
    other = []
    with open(output_file, 'r') as f:
        for line in f:
            print(line.strip(), results[curr].strip())
            if line.strip() != results[curr].strip():
                return False
            
            other.append(results[curr].strip())
            curr += 1
            
        if curr != len(results) - 1:
            print(results, line)
            print(curr, len(results) - 1)
            return False
    print(results)
    print(other)
    return True

if __name__ == '__main__':
    if compare(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])):
        print("CORRECT DIVISORS GENERATED")
    else:
        print("!!!INCORRECT DIVISORS GENERATED!!!")