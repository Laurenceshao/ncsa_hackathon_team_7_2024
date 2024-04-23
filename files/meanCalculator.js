/**
 * Calculates the mean of an array of numbers.
 * @param {number[]} numbers - An array of numbers.
 * @return {number|null} The mean of the numbers or null if the array is empty.
 */
function calculateMean(numbers) {
  if (numbers.length === 0) return null;
  const sum = numbers.reduce((acc, val) => acc + val, 0);
  return sum / numbers.length;
}

module.exports = calculateMean;
