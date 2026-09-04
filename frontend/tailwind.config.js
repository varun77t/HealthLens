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
        // The whole ramp, one step up from where it started. `sm` and `xs` carry about
        // 80% of the words in the application, so those two are what "the text is small"
        // actually meant; the display sizes move with them to keep the ratios intact.
        //
        // `4xl` is defined here rather than inherited: Tailwind's default is 2.25rem,
        // which used to sit *below* the overridden `3xl` of 2.125rem by a hair and would
        // now sit below it outright, so the ramp would run backwards at the two places
        // that use it (the landing headline and the risk band).
        "2xs": ["0.75rem", { lineHeight: "1.05rem" }],
        xs: ["0.8125rem", { lineHeight: "1.2rem" }],
        sm: ["0.875rem", { lineHeight: "1.45rem" }],
        base: ["1rem", { lineHeight: "1.75rem" }],
        lg: ["1.125rem", { lineHeight: "1.75rem" }],
        xl: ["1.375rem", { lineHeight: "1.95rem" }],
        "2xl": ["1.75rem", { lineHeight: "2.25rem" }],
        "3xl": ["2.25rem", { lineHeight: "2.6rem" }],
        "4xl": ["2.625rem", { lineHeight: "1.15" }],
        "5xl": ["3.5rem", { lineHeight: "1.05" }],
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
