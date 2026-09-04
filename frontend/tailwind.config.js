/** @type {import('tailwindcss').Config} */

/**
 * Every colour is a CSS variable, so one set of class names serves both themes and no
 * component has to know which one is active. There is deliberately no `dark:` variant
 * anywhere in the app.
 *
 * The variables hold space-separated RGB channels rather than hex, because that is what
 * lets Tailwind's opacity modifiers keep working — `bg-accent/10`, `ring-accent/45`,
 * `border-line/70` and `bg-canvas/85` are all in use, and they would silently break under
 * a plain `var(--x)`.
 */
const rgb = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // One accent with no hue at all: in light it is near-black on beige, in dark it is
        // the beige on near-black. The only thing competing for attention on a screen is
        // contrast, which the thing you are meant to press has more of than anything else.
        accent: {
          DEFAULT: rgb("accent"),
          hover: rgb("accent-hover"),
          soft: rgb("accent-soft"),
          line: rgb("accent-line"),
          // What sits ON the accent. It inverts with the theme, so a primary button is
          // never white-on-white.
          fg: rgb("accent-fg"),
        },
        ink: rgb("ink"),
        body: rgb("body"),
        muted: rgb("muted"),
        faint: rgb("faint"),
        line: rgb("line"),
        hairline: rgb("hairline"),
        surface: rgb("surface"),
        canvas: rgb("canvas"),
        // Result states are deliberately NOT red/green. A model flag is not a diagnosis,
        // and a pass/fail palette would assert far more than the number supports.
        attention: {
          DEFAULT: rgb("attention"),
          soft: rgb("attention-soft"),
          line: rgb("attention-line"),
        },
        steady: {
          DEFAULT: rgb("steady"),
          soft: rgb("steady-soft"),
          line: rgb("steady-line"),
        },
        caution: {
          DEFAULT: rgb("caution"),
          soft: rgb("caution-soft"),
          line: rgb("caution-line"),
        },
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
      // Also variables: a shadow tuned for dark ink on beige is a grey smear on a dark
      // ground, where separation has to come from the surface being lighter than the page.
      boxShadow: {
        card: "var(--shadow-card)",
        lift: "var(--shadow-lift)",
      },
      maxWidth: { prose: "62ch" },
    },
  },
  plugins: [],
};
