document.getElementById('analyze-form').addEventListener('submit', function(e) {
    e.preventDefault();
    fetch('/analyze', { method: 'POST' })
        .then(response => response.text())
        .then(html => {
            document.body.innerHTML = html;
        })
        .catch(error => {
            console.error('Error during analysis:', error);
        });
});