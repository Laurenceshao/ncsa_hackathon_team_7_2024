def calculate_mean(numbers):
    if not numbers:  # Check if the list is empty
        return None
    return sum(numbers) / len(numbers)

# Test cases
print(calculate_mean([1, 2, 3, 4, 5]))  # Should return 3.0
print(calculate_mean([-1, -2, -3, -4, -5]))  # Should return -3.0
print(calculate_mean([10, -10, 20, -20]))  # Should return 0.0
print(calculate_mean([]))  # Should return None