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
});
