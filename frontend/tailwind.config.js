/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Ink on paper. The accent carries no hue at all, so the only thing competing for
        // attention on a screen is contrast — which the thing you are meant to press has
        // more of than anything else.
        //
        // Every neutral is warm (a touch of yellow-red), so they sit on the beige ground
        // instead of looking like a cool-grey UI pasted onto it.
        accent: {
          DEFAULT: "#1a1815",
          hover: "#000000",
          soft: "#efeade",
          line: "#d5cdbc",
        },
        ink: "#171512",
        body: "#4a443c",
        muted: "#79726a",
        faint: "#a49c90",
        line: "#e2dbcd",
        hairline: "#efe9dd",
        surface: "#fdfbf6",
        canvas: "#f4f0e7",
        // Result states are deliberately NOT red/green. A model flag is not a diagnosis,
        // and a pass/fail palette would assert far more than the number supports. Their
        // soft tints are warmed to match the beige ground.
        attention: { DEFAULT: "#8a5712", soft: "#f7eeda", line: "#e6d4af" },
        steady: { DEFAULT: "#3f6b5c", soft: "#e9efe8", line: "#d0dfd3" },
        caution: { DEFAULT: "#8a4038", soft: "#f8e7e2", line: "#ecd1c9" },
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
        card: "0 1px 2px rgba(48,38,22,0.05), 0 8px 24px -12px rgba(48,38,22,0.10)",
        lift: "0 2px 4px rgba(48,38,22,0.06), 0 18px 40px -18px rgba(48,38,22,0.16)",
      },
      maxWidth: { prose: "62ch" },
    },
  },
  plugins: [],
};
