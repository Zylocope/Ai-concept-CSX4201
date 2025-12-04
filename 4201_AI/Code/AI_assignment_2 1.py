import numpy as np


# Question No.1

#Step 2
print("Step 2 :")
A = np.array([[1,1, 1],
              [1,2,1],
              [1,1,2]])
B = np.array([[10],
              [15],
              [12]])
print(A)
print(B)
print("_"*20)

#Step 3

print("Step 3 :")
# Calculate the inverse of A
A_inv = np.linalg.inv(A)

# Print inverse of A
print("Inverse of A: ")
print(A_inv)


# Check if x is singular and assess its invertibility 
if np.linalg.det(A) == 0:
    print("Matrix A is singular and not invertible")
else:
    print("Matrix A is invertible")
print("_"*20)

# Step 4: Solve for the variables
X = np.dot(A_inv, B)
# print X
print("Price of each item in matrix: ")
print(X)

print("Prices:")
print("Apple Price:", X[0][0])
print("Banana Price:", X[1][0])
print("Cherry Price:", X[2][0])


# Verify the results by substituting
day1_total = X[0][0] + X[1][0] + X[2][0]
day2_total = X[0][0] + 2 * X[1][0] + X[2][0]
day3_total = X[0][0] + X[1][0] + 2 * X[2][0]

print("Verification:")
def verify_totals(day1_total, day2_total, day3_total):
    if day1_total == 10 and day2_total == 15 and day3_total == 12:
        return True
    return False

# Example usage
result = verify_totals(day1_total, day2_total, day3_total)
print("Verification Result:", result)

print(f"Day 1 Total: ${day1_total:.2f}")
print(f"Day 2 Total: ${day2_total:.2f}")
print(f"Day 3 Total: ${day3_total:.2f}")

print("_"*20)
