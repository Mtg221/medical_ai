import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "danger";
  size?: "sm" | "md";
}

export function Button({
  className,
  variant = "primary",
  size = "md",
  ...props
}: ButtonProps) {
  const variants = {
    primary: "bg-sky-700 text-white hover:bg-sky-800 disabled:bg-sky-300",
    secondary: "bg-slate-800 text-white hover:bg-slate-900 disabled:bg-slate-400",
    outline: "border border-slate-300 bg-white text-slate-800 hover:bg-slate-100",
    danger: "bg-rose-700 text-white hover:bg-rose-800 disabled:bg-rose-300",
  };
  const sizes = {
    sm: "px-2.5 py-1.5 text-xs",
    md: "px-4 py-2 text-sm",
  };

  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors cursor-pointer disabled:cursor-not-allowed",
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    />
  );
}
