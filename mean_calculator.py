def calculate_mean(numbers):
    if not numbers:
        return "The array is empty"
    return sum(numbers) / len(numbers)

# Example usage:
# numbers_array = [1, 2, 3, 4, 5]
# mean = calculate_mean(numbers_array)
# print("The mean is:", mean)
