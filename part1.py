import numpy as np

# 1
matrix = np.loadtxt('matrix.csv', delimiter=',')

# 2
vector = np.load('vector.npy')

# 3
output = []
for row in matrix:
    rowVal = 0
    for idx in range(len(vector)):
        rowVal += row[idx] * vector[idx]
    output += [rowVal]
np.savetxt('output_forloop.csv', output)

# 4
output2 = np.dot(matrix, vector)
np.save('output_dot.npy', output2)

# 5
output_diff = np.array(output) - output2
print(output_diff)
np.save('output_difference.csv', output_diff)
# No, it does not. It only proves that the code is correct for this particular example. We could have just gotten lucky.
# This is only one example where it worked. Proof by example is not good enough to prove that code is correct.
