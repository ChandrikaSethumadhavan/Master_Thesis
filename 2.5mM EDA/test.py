def empirical_pmf(samples):
    numbers = len(samples)
    # if numbers == 0:
    #     return []
    list1 = []
    unique_values = sorted(set(samples))
    for i in unique_values:
        count_values = samples.count(i)
        pmf = count_values/numbers
        list1.append((i,pmf))
    
    return list1


samples = []
for j in input().split():
    samples.append(int(j))
print(samples[:-1])
print(empirical_pmf(samples))