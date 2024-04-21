# Function to calculate the mean of an array of numbers
def calculate_mean(numbers):
    if not numbers:
        return 0  # Return 0 for an empty array
    return sum(numbers) / len(numbers)

# Test cases
test_arrays = [
    [1, 2, 3, 4, 5],
    [10, 20, 30, 40, 50],
    [],
    [5],
    [-2, -1, 0, 1, 2]
]

for test in test_arrays:
    print(f'Mean of {test}: {calculate_mean(test)}')