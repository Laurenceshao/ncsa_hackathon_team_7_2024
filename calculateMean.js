function calculateMean(numbersArray) {
  if (numbersArray.length === 0) {
    throw new Error('Array is empty');
  }
  const sum = numbersArray.reduce((acc, val) => acc + val, 0);
  return sum / numbersArray.length;
}

// Example usage:
const mean = calculateMean([1, 2, 3, 4, 5]);
console.log('Mean:', mean);
