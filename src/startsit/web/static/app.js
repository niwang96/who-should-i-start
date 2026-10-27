// Only the button asks for an answer: Enter in a player box (e.g. to pick a suggested name)
// would otherwise submit the form.
document.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.target.matches("input[name='players']")) {
    event.preventDefault();
  }
});
