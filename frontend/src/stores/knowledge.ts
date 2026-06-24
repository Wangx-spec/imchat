import { defineStore } from "pinia";

import {
  getKnowledgeStats,
  listKnowledgeSources,
  rebuildKnowledge,
  searchKnowledge,
  uploadKnowledgeFile,
} from "@/api/knowledge";
import type {
  KnowledgeSearchResponse,
  KnowledgeSourceFile,
  KnowledgeStats,
} from "@/api/types";

export const useKnowledgeStore = defineStore("knowledge", {
  state: () => ({
    stats: null as KnowledgeStats | null,
    sources: [] as KnowledgeSourceFile[],
    searchResult: null as KnowledgeSearchResponse | null,
    loadingStats: false,
    loadingSources: false,
    searching: false,
    uploading: false,
    rebuilding: false,
    error: "" as string,
  }),

  actions: {
    async refreshStats() {
      this.loadingStats = true;
      try {
        this.stats = await getKnowledgeStats();
      } finally {
        this.loadingStats = false;
      }
    },

    async refreshSources() {
      this.loadingSources = true;
      try {
        this.sources = await listKnowledgeSources();
      } finally {
        this.loadingSources = false;
      }
    },

    async refreshAll() {
      this.error = "";
      try {
        await Promise.all([this.refreshStats(), this.refreshSources()]);
      } catch (error) {
        this.error = error instanceof Error ? error.message : String(error);
      }
    },

    async runSearch(query: string, topK: number, includeAnswer: boolean) {
      this.searching = true;
      this.error = "";
      try {
        this.searchResult = await searchKnowledge(query, topK, includeAnswer);
      } catch (error) {
        this.error = error instanceof Error ? error.message : String(error);
      } finally {
        this.searching = false;
      }
    },

    async upload(file: File, rebuild: boolean, pdfParser: "light" | "marker", topic: string) {
      this.uploading = true;
      this.error = "";
      try {
        await uploadKnowledgeFile(file, rebuild, pdfParser, topic);
        await this.refreshAll();
      } catch (error) {
        this.error = error instanceof Error ? error.message : String(error);
      } finally {
        this.uploading = false;
      }
    },

    async rebuild() {
      this.rebuilding = true;
      this.error = "";
      try {
        await rebuildKnowledge();
        await this.refreshAll();
      } catch (error) {
        this.error = error instanceof Error ? error.message : String(error);
      } finally {
        this.rebuilding = false;
      }
    },
  },
});
