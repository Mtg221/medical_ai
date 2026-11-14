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

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  user: null,
  language: "en",
  setAuth: (data) =>
    set({
      token: data.access_token,
      user: {
        user_id: data.user_id,
        institution_id: data.institution_id,
        role: data.role,
        permissions: data.permissions,
      },
    }),
  logout: () => set({ token: null, user: null }),
  setLanguage: (language) => set({ language }),
}));
