// Light or dark mode: the browser's setting, unless the visitor picked one with the header
// toggle, which this browser remembers (localStorage). Loaded in <head> without `defer`, so a saved
// choice applies before the page is drawn (no flash of the other theme). Without JS, the CSS still
// follows the browser setting and the toggle stays hidden.
(function () {
  const KEY = "theme";
  const root = document.documentElement;
  const dark = window.matchMedia("(prefers-color-scheme: dark)");

  function saved() {
    try {
      const theme = localStorage.getItem(KEY);
      return theme === "light" || theme === "dark" ? theme : null;
    } catch {
      return null; // storage blocked (private mode, site data off): follow the browser
    }
  }

  function current() {
    return root.dataset.theme || (dark.matches ? "dark" : "light");
  }

  function update(button) {
    const next = current() === "dark" ? "light" : "dark";
    button.setAttribute("aria-label", `Switch to ${next} mode`);
    button.title = `Switch to ${next} mode`;
    button.textContent = next === "dark" ? "☾︎" : "☀︎"; // moon / sun, as text
  }

  const theme = saved();
  if (theme) root.dataset.theme = theme;

  document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("theme-toggle");
    if (!button) return;
    update(button);
    button.hidden = false;
    button.addEventListener("click", () => {
      const next = current() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try {
        localStorage.setItem(KEY, next);
      } catch {
        // not remembered, but still switched for this page
      }
      update(button);
    });
    // With no saved choice, the page follows the browser setting live; keep the label in step.
    dark.addEventListener("change", () => update(button));
  });
})();
