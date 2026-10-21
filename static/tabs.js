// Switches between the Stage A/B/C/D tab panels on the drawing dashboard.
// Remembers the last-active tab in sessionStorage (per drawing) so that
// submitting a "Run Stage X" form -- a real page reload, since this app has
// no AJAX -- lands back on the tab the user was just looking at instead of
// resetting to Stage A every time.
(function () {
  var buttons = document.querySelectorAll(".tab-btn");
  var panels = document.querySelectorAll(".tab-panel");
  if (!buttons.length || !panels.length) return;

  var storageKey = "activeTab:" + window.location.pathname;

  function activate(tab) {
    buttons.forEach(function (btn) {
      btn.classList.toggle("tab-btn-active", btn.dataset.tab === tab);
    });
    panels.forEach(function (panel) {
      panel.classList.toggle("tab-panel-active", panel.dataset.tab === tab);
    });
  }

  buttons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var tab = btn.dataset.tab;
      activate(tab);
      try {
        sessionStorage.setItem(storageKey, tab);
      } catch (e) {
        // storage can throw (private browsing, blocked cookies) -- tab
        // switching still works for this page view, it just won't persist
      }
    });
  });

  var initial = "a";
  try {
    initial = sessionStorage.getItem(storageKey) || "a";
  } catch (e) {
    // fall back to the default tab
  }
  if (!document.querySelector('.tab-panel[data-tab="' + initial + '"]')) {
    initial = "a";
  }
  activate(initial);
})();
