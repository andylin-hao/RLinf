// Expand every top-level navigation section by default, like the LeRobot docs
// where the section groups land open. The pydata theme only opens the branch
// containing the active page, so here we open all first-level sections (their
// immediate children show; deeper sub-trees stay collapsed).
(function () {
  "use strict";

  function init() {
    var nav = document.querySelector(".bd-docs-nav");
    if (!nav) return;
    nav.querySelectorAll("li.toctree-l1.has-children > details").forEach(
      function (section) {
        section.setAttribute("open", "");
      }
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
