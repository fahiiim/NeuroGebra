/* Keep the built-in Material search and provide its familiar shortcut. */
(function () {
  "use strict";
  document.addEventListener("keydown", function (event) {
    if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "k") return;
    const toggle = document.getElementById("__search");
    const input = document.querySelector('[data-md-component="search-query"]');
    if (!toggle || !input) return;
    event.preventDefault();
    if (!toggle.checked) toggle.click();
    requestAnimationFrame(function () { input.focus(); });
  });
  document.addEventListener("click", async function (event) {
    const button = event.target.closest("[data-ng-copy]");
    if (!button) return;
    const command = button.dataset.ngCopy;
    try {
      await navigator.clipboard.writeText(command);
      button.textContent = "Copied";
      button.setAttribute("aria-label", "Installation command copied");
      setTimeout(function () {
        button.textContent = "Copy";
        button.setAttribute("aria-label", "Copy installation command");
      }, 1800);
    } catch (_) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(button.previousElementSibling);
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = "Select + copy";
      button.setAttribute("aria-label", "Command selected. Press Control C or Command C to copy");
    }
  });
})();
