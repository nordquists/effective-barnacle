import sys
def generate_divisors(N, x):
    results = []
    curr = 0

    while curr < N:
        curr += x
        results.append(curr)

    return results

def compare(output_file, N, x):
    results = generate_divisors(N, x)
    curr = 0
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip() != results[curr].strip():
                return False
            curr += 1
    return True

if __name__ == '__main__':
    if compare(sys.args[1], sys.args[2], sys.args[3]):
        print("CORRECT DIVISORS GENERATED")
    else:
        print("!!!INCORRECT DIVISORS GENERATED!!!")