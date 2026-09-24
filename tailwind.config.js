/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#f3f4f7',
        surface: '#ffffff',
        raised: '#eef0f4',
        line: '#d7dbe3',
        ink: '#14161c',
        mute: '#586071',
        faint: '#8890a0',
        accent: { DEFAULT: '#3457e6', strong: '#2743e6', ink: '#ffffff' },
        good: '#157f4a',
        warn: '#a15a00',
        bad: '#c92a3e',
        sky: '#0969da',
      },
      fontFamily: { sans: ['"Geist Variable"', 'ui-sans-serif', 'system-ui', 'sans-serif'] },
    },
  },
  plugins: [],
};
