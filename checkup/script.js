// Basic calculator script
document.addEventListener('DOMContentLoaded', () => {
  const display = document.getElementById('display');
  const keys = document.querySelectorAll('.key');

  const append = (char) => {
    if (display.value === '0' && char !== '.') {
      display.value = char;
    } else {
      display.value += char;
    }
  };

  const clear = () => {
    display.value = '';
  };

  const backspace = () => {
    display.value = display.value.slice(0, -1);
  };

  const evaluate = () => {
    try {
      // Replace any unicode division sign with slash if present
      const expression = display.value.replace(/÷/g, '/').replace(/×/g, '*');
      // Use Function constructor for safe eval
      // eslint-disable-next-line no-new-func
      const result = Function(`'use strict'; return (${expression})`)();
      display.value = String(result);
    } catch (e) {
      display.value = 'Error';
    }
  };

  keys.forEach(key => {
    key.addEventListener('click', () => {
      const action = key.dataset.action;
      const value = key.dataset.value;
      switch (action) {
        case 'digit':
          append(value);
          break;
        case 'dot':
          // Prevent multiple dots in the current number segment
          const parts = display.value.split(/[^0-9.]/);
          if (!parts[parts.length - 1].includes('.')) {
            append('.');
          }
          break;
        case 'operator':
          // Avoid two consecutive operators
          if (display.value && !/[+\-*/]$/.test(display.value)) {
            append(value);
          }
          break;
        case 'clear':
          clear();
          break;
        case 'backspace':
          backspace();
          break;
        case 'evaluate':
          evaluate();
          break;
        default:
          // No action defined
          break;
      }
    });
  });
});
