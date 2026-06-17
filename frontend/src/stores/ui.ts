import { defineStore } from "pinia";

export const useUiStore = defineStore("ui", {
  state: () => ({
    sidebarOpen: true,
  }),
  actions: {
    toggleSidebar() {
      this.sidebarOpen = !this.sidebarOpen;
    },
  },
});
