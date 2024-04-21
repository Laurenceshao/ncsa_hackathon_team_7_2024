def calculate_mean(numbers):
    if not numbers:  # Check if the list is empty
        return None  # Return None or raise an error as appropriate
    return sum(numbers) / len(numbers)  # Calculate and return the mean
ean

# Test cases for the calculate_mean function
def test_calculate_mean():
    assert calculate_mean([1, 2, 3, 4, 5]) == 3, 'Test with normal array of numbers failed'
    assert calculate_mean([]) is None, 'Test with empty array failed'
    assert calculate_mean([42]) == 42, 'Test with single number array failed'

    print('All test cases passed!')

test_calculate_mean()

# Test cases for the calculate_mean function
def test_calculate_mean():
    assert calculate_mean([1, 2, 3, 4, 5]) == 3, 'Test with normal array of numbers failed'
    assert calculate_mean([]) is None, 'Test with empty array failed'
    assert calculate_mean([42]) == 42, 'Test with single number array failed'

    print('All test cases passed!')

test_calculate_mean()

# Test cases for the calculate_mean function

def print_test_case_result(test_case, expected, actual):
    print(f'Test Case: {test_case}, Expected: {expected}, Actual: {actual}, Pass: {expected == actual}')

test_cases = [
    ([1, 2, 3, 4, 5], 3.0),
    ([], None),
    ([42], 42.0),
    ([-5, -10, -15], -10.0),
    ([3.5, 2.5, 1.0, 4.0], 2.75)
]

for numbers, expected_mean in test_cases:
    result = calculate_mean(numbers)
    print_test_case_result(numbers, expected_mean, result)

# Test cases for the calculate_mean function

def print_test_case_result(test_case, expected, actual):
    print(f'Test Case: {test_case}, Expected: {expected}, Actual: {actual}, Pass: {expected == actual}')

test_cases = [
    ([1, 2, 3, 4, 5], 3.0),
    ([], None),
    ([42], 42.0),
    ([-5, -10, -15], -10.0),
    ([3.5, 2.5, 1.0, 4.0], 2.75)
]

for numbers, expected_mean in test_cases:
    result = calculate_mean(numbers)
    print_test_case_result(numbers, expected_mean, result)

# Test cases for the calculate_mean function

def print_test_case_result(test_case_number, numbers, expected_result):
    result = calculate_mean(numbers)
    print(f'Test Case {test_case_number}:', 'PASS' if result == expected_result else 'FAIL')

# Test Case 1: Typical array of numbers
print_test_case_result(1, [1, 2, 3, 4, 5], 3)

# Test Case 2: Empty array
print_test_case_result(2, [], None)

# Test Case 3: Array with a single number
print_test_case_result(3, [42], 42)

# Test Case 4: Array with negative numbers
print_test_case_result(4, [-3, -2, -1, 0, 1, 2, 3], 0)

# Test Case 5: Array with both integers and floating-point numbers
print_test_case_result(5, [1.5, 2.5, 3.5], 2.5)

# Test cases for the calculate_mean function

def print_test_case_result(test_case, expected, actual):
    print(f'Test Case: {test_case}, Expected: {expected}, Actual: {actual}, Pass: {expected == actual}')

test_cases = [
    ([1, 2, 3, 4, 5], 3.0),
    ([], None),
    ([42], 42.0),
    ([-5, -10, -15], -10.0),
    ([3.5, 2.5, 1.0, 4.0], 2.75)
]

for numbers, expected_mean in test_cases:
    result = calculate_mean(numbers)
    print_test_case_result(numbers, expected_mean, result)
