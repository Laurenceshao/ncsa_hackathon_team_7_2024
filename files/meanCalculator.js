/**
 * Calculates the mean of an array of numbers.
 * @param {number[]} numbers - An array of numbers.
 * @return {number} The mean of the numbers.
 */
function calculateMean(numbers) {
  if (!Array.isArray(numbers) || numbers.length === 0) {
    return NaN;
  }

  const sum = numbers.reduce((acc, val) => acc + val, 0);
  return sum / numbers.length;
}

module.exports = calculateMean;
