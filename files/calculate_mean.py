# Function to calculate the mean of an array of numbers
def calculate_mean(numbers):
    if not numbers:  # Check if the list is empty
        return 0  # Return 0 or an appropriate message
    return sum(numbers) / len(numbers)  # Calculate and return the mean

# Test cases for the calculate_mean function
def test_calculate_mean():
    assert calculate_mean([1, 2, 3, 4, 5]) == 3, 'Test with positive integers failed'
    assert calculate_mean([-1, -2, -3, -4, -5]) == -3, 'Test with negative integers failed'
    assert calculate_mean([1.5, 2.5, 3.5]) == 2.5, 'Test with floating-point numbers failed'
    assert calculate_mean([]) == 0, 'Test with empty array failed'

    print('All tests passed!')

test_calculate_mean()

# Test cases for the calculate_mean function
def test_calculate_mean():
    assert calculate_mean([1, 2, 3, 4, 5]) == 3, 'Test with positive integers failed'
    assert calculate_mean([-1, -2, -3, -4, -5]) == -3, 'Test with negative integers failed'
    assert calculate_mean([1.5, 2.5, 3.5]) == 2.5, 'Test with floating-point numbers failed'
    assert calculate_mean([]) == 0, 'Test with empty array failed'
    print('All tests passed!')

test_calculate_mean()