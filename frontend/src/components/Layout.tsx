import {
  Activity,
  AlertTriangle,
  Braces,
  CheckSquare,
  Database,
  FileCheck,
  FileSearch,
  FileText,
  FlaskConical,
  GitBranch,
  Globe,
  History,
  Layers,
  LogOut,
  Moon,
  Settings,
  Sun,
  Upload,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { NavLink, Outlet } from "react-router-dom";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { useAuthStore } from "@/store/authStore";
import { useThemeStore } from "@/store/themeStore";
import { cn } from "@/lib/utils";

export function Layout() {
  const { t, i18n } = useTranslation();
  const { user, language, setLanguage, logout } = useAuthStore();
  const { theme, toggleTheme } = useThemeStore();

  const toggleLanguage = () => {
    const next = language === "en" ? "fr" : "en";
    setLanguage(next);
    void i18n.changeLanguage(next);
  };

  const navItems = [
    { to: "/", label: t("nav.workspace"), icon: FileText },
    { to: "/upload", label: t("nav.upload"), icon: Upload },
    { to: "/pdf", label: "PDF.js", icon: FileSearch },
    { to: "/relations", label: "Relations", icon: GitBranch },
    { to: "/review", label: "Review", icon: CheckSquare },
    { to: "/benchmark", label: t("nav.benchmark"), icon: FlaskConical },
    { to: "/ablation", label: "Ablation", icon: Layers },
    { to: "/errors", label: "Errors", icon: Activity },
    { to: "/corpus", label: "Corpus", icon: Database },
    { to: "/reports", label: "Reports", icon: FileCheck },
    { to: "/timeline", label: t("nav.timeline"), icon: History },
    { to: "/schema", label: "Schema", icon: Braces },
    { to: "/settings", label: "Settings", icon: Settings },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:bg-sky-700 focus:px-3 focus:py-2 focus:text-white"
      >
        Skip to main content
      </a>

      {/* Permanent Clinical Research Disclaimer */}
      <div
        role="status"
        className="bg-amber-50 border-b border-amber-200 px-4 py-2 text-xs text-amber-950 dark:bg-amber-950/60 dark:border-amber-800 dark:text-amber-100 flex items-center justify-center gap-2 font-medium"
      >
        <AlertTriangle className="h-4 w-4 text-amber-700 dark:text-amber-400 shrink-0" aria-hidden="true" />
        <span>{t("disclaimer")}</span>
      </div>

      {/* Header */}
      <header className="bg-white border-b border-slate-200 dark:bg-slate-900 dark:border-slate-800 px-4 py-3">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <span className="h-8 w-8 rounded-md bg-sky-700 text-white font-bold flex items-center justify-center text-sm">
                M
              </span>
              <span className="font-semibold text-slate-900 dark:text-slate-100 tracking-tight">{t("appTitle")}</span>
            </div>

            {user && (
              <nav aria-label="Primary navigation" className="flex flex-wrap items-center gap-1">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  return (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      end={item.to === "/"}
                      className={({ isActive }) =>
                        cn(
                          "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                          isActive
                            ? "bg-sky-50 text-sky-800 dark:bg-sky-900/50 dark:text-sky-200"
                            : "text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
                        )
                      }
                    >
                      <Icon className="h-4 w-4" aria-hidden="true" />
                      {item.label}
                    </NavLink>
                  );
                })}
              </nav>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={toggleTheme}
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            >
              {theme === "dark" ? <Sun className="h-3.5 w-3.5" aria-hidden="true" /> : <Moon className="h-3.5 w-3.5" aria-hidden="true" />}
              {theme === "dark" ? "Light" : "Dark"}
            </Button>

            <Button variant="outline" size="sm" onClick={toggleLanguage} aria-label="Toggle language">
              <Globe className="h-3.5 w-3.5" aria-hidden="true" />
              {language.toUpperCase()}
            </Button>

            {user && (
              <>
                <Badge tone="sky">
                  {user.institution_id} · {user.role}
                </Badge>
                <Button variant="outline" size="sm" onClick={logout}>
                  <LogOut className="h-3.5 w-3.5" aria-hidden="true" />
                  {t("auth.signOut")}
                </Button>
              </>
            )}
          </div>
        </div>
      </header>

      <main id="main-content" className="flex-1 max-w-7xl w-full mx-auto px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
