# Test cases for the calculate_mean function
from calculate_mean import calculate_mean

def print_test_case_result(test_case, expected):
    result = calculate_mean(test_case)
    print(f'Test case: {test_case}, Expected: {expected}, Result: {result}, Pass: {result == expected}')

# Test with a simple array of numbers
print_test_case_result([1, 2, 3, 4, 5], 3)

# Test with an array containing negative numbers
print_test_case_result([-1, -2, -3, -4, -5], -3)

# Test with an array containing floating-point numbers
print_test_case_result([1.5, 2.5, 3.5], 2.5)

# Test with an empty array
print_test_case_result([], 0)
