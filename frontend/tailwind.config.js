/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: { DEFAULT: "#f7f4ee", raised: "#fdfcf9", sunk: "#efebe2" },
        ink: { DEFAULT: "#23211e", soft: "#55514a", faint: "#857f74" },
        rule: { DEFAULT: "#dcd6c9", strong: "#c4bcab" },
        accent: { DEFAULT: "#2f5d62", hover: "#244a4e", tint: "#e3ecea" },
        warn: { DEFAULT: "#8a5a12", tint: "#f6ecd6" },
        danger: { DEFAULT: "#9b2c2c", tint: "#f6e1de" },
      },
      fontFamily: {
        serif: ["Charter", "Iowan Old Style", "Georgia", "Cambria", "serif"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
