import { defineStore } from "pinia";

import {
  createConversation,
  deleteConversation as deleteConversationApi,
  listConversations,
  listMessages,
  resumeHitl,
  streamChat,
  streamMultimodalChat,
} from "@/api/chat";
import { getHealth } from "@/api/system";
import type { ChatMessage, Conversation, HealthStatus, HitlInterrupt, PreviewImage } from "@/api/types";

function newId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function cloneImages(images: PreviewImage[]): PreviewImage[] {
  return images.map((item) => ({ ...item }));
}

export const useChatStore = defineStore("chat", {
  state: () => ({
    sessionId: "" as string,
    conversations: [] as Conversation[],
    messages: [] as ChatMessage[],
    pendingImages: [] as PreviewImage[],
    isStreaming: false,
    activeHitlInterrupt: null as HitlInterrupt | null,
    hitlMessageId: "" as string,
    healthStatus: null as HealthStatus | null,
    currentAudio: null as HTMLAudioElement | null,
  }),

  getters: {
    sessionLabel(state): string {
      return state.sessionId ? `${state.sessionId.slice(0, 8)}...` : "未创建";
    },
    hasPendingImages(state): boolean {
      return state.pendingImages.length > 0;
    },
  },

  actions: {
    async initialize() {
      this.healthStatus = await getHealth();
      await this.refreshConversations();
      if (this.conversations.length > 0) {
        await this.switchSession(this.conversations[0].session_id);
      } else {
        await this.startNewConversation();
      }
    },

    async refreshHealth() {
      this.healthStatus = await getHealth();
    },

    async refreshConversations() {
      this.conversations = await listConversations();
    },

    async startNewConversation() {
      this.sessionId = await createConversation();
      this.messages = [
        {
          id: newId("system"),
          role: "assistant",
          content: "已开启新会话。请描述您的医学问题，回答仅供参考，不能替代专业诊疗。",
          status: "done",
        },
      ];
      await this.refreshConversations();
    },

    async switchSession(sessionId: string) {
      this.sessionId = sessionId;
      const rows = await listMessages(sessionId);
      this.messages = rows.map((row) => ({
        id: newId(row.role),
        role: row.role,
        content: row.content,
        status: "done",
      }));
      if (!this.messages.length) {
        this.messages.push({
          id: newId("system"),
          role: "assistant",
          content: `已切换到会话 ${sessionId.slice(0, 8)}...`,
          status: "done",
        });
      }
      await this.refreshConversations();
    },

    async deleteConversation(sessionId: string) {
      if (!sessionId || this.isStreaming) return;
      const wasActive = sessionId === this.sessionId;
      await deleteConversationApi(sessionId);
      await this.refreshConversations();
      if (!wasActive) return;

      const next = this.conversations[0];
      if (next) {
        await this.switchSession(next.session_id);
      } else {
        await this.startNewConversation();
      }
    },

    addImages(files: File[], maxImages = 3) {
      const imageFiles = files.filter((file) => file.type.startsWith("image/"));
      const remain = maxImages - this.pendingImages.length;
      if (remain <= 0) {
        this.pushSystemMessage(`一次最多支持 ${maxImages} 张图片。`);
        return;
      }
      for (const file of imageFiles.slice(0, remain)) {
        this.pendingImages.push({
          id: newId("image"),
          file,
          objectUrl: URL.createObjectURL(file),
        });
      }
      if (imageFiles.length > remain) {
        this.pushSystemMessage(`已达到图片上限（${maxImages} 张），其余图片已忽略。`);
      }
    },

    removeImage(id: string) {
      const item = this.pendingImages.find((image) => image.id === id);
      if (item) URL.revokeObjectURL(item.objectUrl);
      this.pendingImages = this.pendingImages.filter((image) => image.id !== id);
    },

    clearPendingImages() {
      for (const item of this.pendingImages) URL.revokeObjectURL(item.objectUrl);
      this.pendingImages = [];
    },

    async send(message: string) {
      const trimmed = message.trim();
      const images = cloneImages(this.pendingImages);
      if ((!trimmed && !images.length) || !this.sessionId || this.isStreaming) return;

      this.messages.push({
        id: newId("user"),
        role: "user",
        content: trimmed || "（仅图片）",
        images,
        status: "done",
      });
      this.clearPendingImages();

      const assistantMessage: ChatMessage = {
        id: newId("assistant"),
        role: "assistant",
        content: "",
        status: "streaming",
      };
      this.messages.push(assistantMessage);
      this.isStreaming = true;

      try {
        const handlers = this.buildStreamHandlers(assistantMessage.id);
        if (images.length > 0) {
          await streamMultimodalChat(
            this.sessionId,
            trimmed,
            images.map((image) => image.file),
            handlers,
          );
        } else {
          await streamChat(this.sessionId, trimmed, handlers);
        }
        if (!assistantMessage.content && assistantMessage.status !== "hitl_waiting") {
          assistantMessage.content = "未获取到回答，请稍后重试。";
        }
        if (assistantMessage.status === "streaming") assistantMessage.status = "done";
      } catch (error) {
        assistantMessage.status = "error";
        assistantMessage.content = `错误：${error instanceof Error ? error.message : String(error)}`;
      } finally {
        this.isStreaming = false;
        await this.refreshConversations();
      }
    },

    async submitHitl(decision: "approve" | "reject" | "revise", note: string) {
      if (!this.sessionId || !this.activeHitlInterrupt || !this.hitlMessageId) return;
      const msg = this.messages.find((item) => item.id === this.hitlMessageId);
      if (!msg) return;

      msg.status = "streaming";
      this.isStreaming = true;
      this.activeHitlInterrupt = null;
      try {
        await resumeHitl(this.sessionId, decision, note, this.buildStreamHandlers(msg.id));
        if (msg.status === "streaming") msg.status = "done";
      } catch (error) {
        msg.status = "error";
        msg.content = `错误：${error instanceof Error ? error.message : String(error)}`;
      } finally {
        this.isStreaming = false;
        this.hitlMessageId = "";
        await this.refreshConversations();
      }
    },

    stopAudio() {
      if (!this.currentAudio) return;
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.currentAudio = null;
    },

    setCurrentAudio(audio: HTMLAudioElement | null) {
      this.stopAudio();
      this.currentAudio = audio;
    },

    pushSystemMessage(content: string) {
      this.messages.push({
        id: newId("system"),
        role: "system",
        content,
        status: "done",
      });
    },

    buildStreamHandlers(messageId: string) {
      return {
        onChunk: (text: string, replace: boolean) => {
          const msg = this.messages.find((item) => item.id === messageId);
          if (!msg) return;
          msg.content = replace ? text : msg.content + text;
        },
        onDone: (answer: string) => {
          const msg = this.messages.find((item) => item.id === messageId);
          if (!msg) return;
          if (answer.trim()) msg.content = answer.trim();
          msg.status = "done";
        },
        onError: (detail: string) => {
          const msg = this.messages.find((item) => item.id === messageId);
          if (!msg) return;
          msg.content = `错误：${detail}`;
          msg.status = "error";
        },
        onHitl: (interrupt: HitlInterrupt) => {
          const msg = this.messages.find((item) => item.id === messageId);
          if (!msg) return;
          this.activeHitlInterrupt = interrupt;
          this.hitlMessageId = messageId;
          msg.status = "hitl_waiting";
          msg.meta = { ...(msg.meta || {}), interrupt };
        },
      };
    },
  },
});
