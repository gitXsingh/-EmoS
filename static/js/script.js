// Persist dark mode across page loads using localStorage
function setDarkMode(enabled) {
  if (enabled) {
    document.body.classList.add('dark-mode');
    localStorage.setItem('darkMode', '1');
    document.getElementById('darkBtn').innerHTML = '☀️';
  } else {
    document.body.classList.remove('dark-mode');
    localStorage.setItem('darkMode', '0');
    document.getElementById('darkBtn').innerHTML = '🌙';
  }
}

function toggleDarkMode() {
  setDarkMode(!document.body.classList.contains('dark-mode'));
}

window.onload = function() {
  let dark = localStorage.getItem('darkMode');
  setDarkMode(dark === '1');
}; 