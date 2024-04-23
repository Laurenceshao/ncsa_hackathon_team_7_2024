function calculateMean(numbers) {
  if (!Array.isArray(numbers)) {
    throw new Error('Input must be an array of numbers.');
  }
  if (numbers.length === 0) {
    return 0;
  }
  const sum = numbers.reduce((acc, val) => acc + val, 0);
  return sum / numbers.length;
}

module.exports = calculateMean;
