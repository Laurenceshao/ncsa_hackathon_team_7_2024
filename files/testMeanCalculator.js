const calculateMean = require('./meanCalculator');

// Test cases
console.log(calculateMean([1, 2, 3])); // Expected output: 2
console.log(calculateMean([5, -2, 9])); // Expected output: 4
console.log(calculateMean([])); // Expected output: NaN
console.log(calculateMean(['a', 1, 2])); // Expected output: NaN
