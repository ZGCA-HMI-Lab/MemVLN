window.HELP_IMPROVE_VIDEOJS = false;

function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');

    if (!bibtexElement || !button) {
        return;
    }

    const copyText = button.querySelector('.copy-text');
    const setCopiedState = () => {
        button.classList.add('copied');
        if (copyText) {
            copyText.textContent = 'Copied';
        }

        setTimeout(() => {
            button.classList.remove('copied');
            if (copyText) {
                copyText.textContent = 'Copy';
            }
        }, 2000);
    };

    if (navigator.clipboard) {
        navigator.clipboard.writeText(bibtexElement.textContent)
            .then(setCopiedState)
            .catch(() => fallbackCopyText(bibtexElement.textContent, setCopiedState));
    } else {
        fallbackCopyText(bibtexElement.textContent, setCopiedState);
    }
}

function fallbackCopyText(text, onSuccess) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    onSuccess();
}

function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');

    if (!scrollButton) {
        return;
    }

    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});
