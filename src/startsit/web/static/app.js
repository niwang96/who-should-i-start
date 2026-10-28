// Only the button asks for an answer: Enter in a player box (e.g. to pick a suggested name)
// would otherwise submit the form.
document.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.target.matches("input[name='players']")) {
    event.preventDefault();
  }
});

// htmx leaves the page unchanged when a request fails (an error status, the host's time limit,
// no connection), which looks like nothing happened. Say so instead.
function showError(message) {
  const result = document.getElementById("result");
  const box = document.createElement("div");
  box.className = "error";
  box.setAttribute("role", "alert");
  box.textContent = message;
  result.replaceChildren(box);
}

document.addEventListener("htmx:responseError", () => {
  showError("Something went wrong on our side, or it took too long. Please try again in a minute.");
});
document.addEventListener("htmx:sendError", () => {
  showError("Couldn't reach the site. Check your connection and try again.");
});
