/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#11181c",
        muted: "#5c6b73",
        line: "#e3e8ea",
        surface: "#ffffff",
        canvas: "#f6f8f9",
        accent: "#2f6f9f",
        // Status colours are deliberately not red/green pass-fail: a model flag is not a
        // diagnosis, and colouring it like one would overstate what the number means.
        flagged: "#8a5a00",
        flaggedBg: "#fdf4e3",
        clear: "#2f6f5f",
        clearBg: "#eef6f3",
        warn: "#8a3d3d",
        warnBg: "#fbf0f0",
      },
    },
  },
  plugins: [],
};
