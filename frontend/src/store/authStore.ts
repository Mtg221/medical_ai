import { create } from "zustand";
import type { TokenResponse } from "@/types/api";

interface AuthState {
  token: string | null;
  user: Omit<TokenResponse, "access_token" | "token_type"> | null;
  language: "en" | "fr";
  setAuth: (data: TokenResponse) => void;
  logout: () => void;
  setLanguage: (lang: "en" | "fr") => void;
}

function readStoredAuth(): Pick<AuthState, "token" | "user"> {
  if (typeof window === "undefined") return { token: null, user: null };
  try {
    const raw = window.sessionStorage.getItem("medresearch_auth");
    if (!raw) return { token: null, user: null };
    const parsed = JSON.parse(raw) as Pick<AuthState, "token" | "user">;
    return { token: parsed.token ?? null, user: parsed.user ?? null };
  } catch {
    return { token: null, user: null };
  }
}

const initial = readStoredAuth();

export const useAuthStore = create<AuthState>((set) => ({
  token: initial.token,
  user: initial.user,
  language: "en",
  setAuth: (data) => {
    const user = {
      user_id: data.user_id,
      institution_id: data.institution_id,
      role: data.role,
      permissions: data.permissions,
    };
    try {
      window.sessionStorage.setItem(
        "medresearch_auth",
        JSON.stringify({ token: data.access_token, user })
      );
    } catch {
      // ignore
    }
    set({ token: data.access_token, user });
  },
  logout: () => {
    try {
      window.sessionStorage.removeItem("medresearch_auth");
    } catch {
      // ignore
    }
    set({ token: null, user: null });
  },
  setLanguage: (language) => set({ language }),
}));
