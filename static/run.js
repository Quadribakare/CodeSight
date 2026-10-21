// While a stage runs, the form post is a real page load that can take
// minutes, so show something is happening: disable every run button (no
// double submits), swap the clicked one's label, and open the blinking eyes.
(function () {
  var forms = document.querySelectorAll(".run-form");
  forms.forEach(function (form) {
    form.addEventListener("submit", function () {
      document.querySelectorAll(".run-form button").forEach(function (btn) {
        btn.disabled = true;
      });
      var btn = form.querySelector("button");
      if (btn && btn.dataset.running) btn.textContent = btn.dataset.running;
      var eyes = form.querySelector(".eyes");
      if (eyes) eyes.classList.add("blinking");
    });
  });
})();
