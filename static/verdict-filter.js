document.addEventListener("DOMContentLoaded", function () {
  var filterButtons = document.querySelectorAll(".badge-filter");
  var rows = document.querySelectorAll("#verdict-table-body tr");

  filterButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var filter = btn.getAttribute("data-filter");

      filterButtons.forEach(function (b) {
        b.classList.remove("badge-active");
      });
      btn.classList.add("badge-active");

      rows.forEach(function (row) {
        var matches = filter === "all" || row.getAttribute("data-verdict") === filter;
        row.style.display = matches ? "" : "none";
      });
    });
  });
});
