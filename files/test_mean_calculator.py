import mean_calculator

def test_calculate_mean():
    assert mean_calculator.calculate_mean([1, 2, 3, 4, 5]) == 3, 'Test with positive numbers failed'
    assert mean_calculator.calculate_mean([-1, -2, -3, -4, -5]) == -3, 'Test with negative numbers failed'
    assert mean_calculator.calculate_mean([1, -2, 3, -4, 5]) == 0.6, 'Test with mixed numbers failed'
    assert mean_calculator.calculate_mean([42]) == 42, 'Test with single number failed'
    assert mean_calculator.calculate_mean([]) == None, 'Test with empty array failed'

if __name__ == '__main__':
    test_calculate_mean()
    print('All tests passed!')
