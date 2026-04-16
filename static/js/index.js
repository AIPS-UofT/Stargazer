document.addEventListener("DOMContentLoaded", () => {
  const burgers = Array.prototype.slice.call(document.querySelectorAll(".navbar-burger"), 0);

  burgers.forEach((burger) => {
    burger.addEventListener("click", () => {
      const target = burger.dataset.target;
      const menu = document.getElementById(target);

      burger.classList.toggle("is-active");
      if (menu) {
        menu.classList.toggle("is-active");
      }
    });
  });

  document.querySelectorAll(".mascot-callout").forEach((node) => {
    const bubble = node.querySelector(".mascot-bubble");
    const lines = (node.dataset.lines || "")
      .split("|")
      .map((item) => item.trim())
      .filter(Boolean);

    let current = 0;

    const activate = () => node.classList.add("is-active");
    const deactivate = () => node.classList.remove("is-active");
    const advance = () => {
      if (!bubble || lines.length < 2) return;
      current = (current + 1) % lines.length;
      bubble.textContent = lines[current];
    };

    node.addEventListener("mouseenter", activate);
    node.addEventListener("mouseleave", deactivate);
    node.addEventListener("focusin", activate);
    node.addEventListener("focusout", deactivate);
    node.addEventListener("click", advance);
  });
});
