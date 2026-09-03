/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // One restrained accent. Everything else is neutral, so the only strong colour on
        // a screen is the thing you are meant to press.
        accent: {
          DEFAULT: "#12716b",
          hover: "#0e5c57",
          soft: "#eef5f4",
          line: "#cfe3e1",
        },
        ink: "#0f1e26",
        body: "#3c4a52",
        muted: "#6b7b84",
        faint: "#9aa8af",
        line: "#e7ecee",
        hairline: "#f1f5f6",
        surface: "#ffffff",
        canvas: "#f7f9fa",
        // Result states are deliberately NOT red/green. A model flag is not a diagnosis,
        // and a pass/fail palette would assert far more than the number supports.
        attention: { DEFAULT: "#8f5a12", soft: "#fdf6ea", line: "#f0dfc2" },
        steady: { DEFAULT: "#3f6b5c", soft: "#f0f5f2", line: "#d7e5dd" },
        caution: { DEFAULT: "#8a4038", soft: "#fdf1ef", line: "#f2d9d5" },
      },
      fontFamily: {
        sans: [
          "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Inter", "Roboto",
          "Helvetica Neue", "Arial", "sans-serif",
        ],
      },
      fontSize: {
        // A deliberate scale: long paragraphs of 12px text were most of what made the
        // first version feel like a dashboard.
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
        xs: ["0.75rem", { lineHeight: "1.15rem" }],
        sm: ["0.8125rem", { lineHeight: "1.3rem" }],
        base: ["0.9375rem", { lineHeight: "1.6rem" }],
        lg: ["1.0625rem", { lineHeight: "1.6rem" }],
        xl: ["1.3125rem", { lineHeight: "1.85rem" }],
        "2xl": ["1.625rem", { lineHeight: "2.1rem" }],
        "3xl": ["2.125rem", { lineHeight: "2.5rem" }],
        "5xl": ["3.25rem", { lineHeight: "1.05" }],
      },
      borderRadius: { xl: "0.875rem", "2xl": "1.25rem" },
      boxShadow: {
        card: "0 1px 2px rgba(15,30,38,0.04), 0 8px 24px -12px rgba(15,30,38,0.08)",
        lift: "0 2px 4px rgba(15,30,38,0.05), 0 18px 40px -18px rgba(15,30,38,0.14)",
      },
      maxWidth: { prose: "62ch" },
    },
  },
  plugins: [],
};
