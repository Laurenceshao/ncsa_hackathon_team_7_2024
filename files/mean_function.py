# This function will calculate the mean of an array of numbers.

def calculate_mean(numbers):
    # Check if the input is a valid array of numbers
    if not isinstance(numbers, list) or not all(isinstance(num, (int, float)) for num in numbers):
        raise ValueError('Input must be a list of numbers')

    # Calculate the sum of the numbers
    total_sum = sum(numbers)

    # Calculate the mean
    mean = total_sum / len(numbers) if numbers else 0

    return mean
