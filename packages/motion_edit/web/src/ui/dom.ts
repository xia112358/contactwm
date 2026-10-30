export function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`missing UI element #${id}`);
  return element as T;
}

export function setupTabs(): void {
  const tabs = [...document.querySelectorAll<HTMLButtonElement>('[data-tab]')];
  const pages = [...document.querySelectorAll<HTMLElement>('[data-page]')];
  for (const tab of tabs) {
    tab.addEventListener('click', () => {
      const activePage = tab.dataset.tab;
      for (const item of tabs) item.classList.toggle('active', item === tab);
      for (const page of pages) page.classList.toggle('active', page.dataset.page === activePage);
    });
  }
}
